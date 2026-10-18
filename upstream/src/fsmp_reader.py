#!/usr/bin/env python3
'''
把 fsmp PE t 转到波形片段的时间上
'''

import pyarrow.parquet as pq
import h5py
import numpy as np
import numexpr as ne
import sys
from warnings import warn


def initialize(bs_e, bs_ch, CSIZE, i_wave, location, cumc_flat, bs_ech, zs, meta_zs, zchain, meta_zchain):
    '''
    针对 BAPPE 设计。

    在 BAPPE 开始时，我们需要每个通道都有一个采样，从而计算某个 (t, v, E) 下的 likelihood。
    '''
    # 依赖于 NumPy 的 argmax 特别约定： In case of multiple
    # occurrences of the maximum values, the indices corresponding
    # to the first occurrence are returned.
    ri = np.random.randint(CSIZE, size=len(i_wave), dtype=np.uint32)[:, None]

    # 下面代码就是以下语句，只是快 2 倍。
    # (cumc[bs_e, bs_ch] > ri).argmax(1)
    buf = location
    np.take(cumc_flat, bs_ech, axis=0, out=buf, mode='clip')
    place = ne.evaluate("buf > ri")
    choices = place.argmax(1)
    bs = np.s_[bs_e, bs_ch]
    zs[bs] = zchain[bs + (choices,)]
    meta_zs[bs] = meta_zchain[bs + (choices,)][["s0", "nu_lc"]]


def entries(bs_e, bs_ch, offset, N, CSIZE, i_wave, location, cumc_flat, bs_ech, zs, meta_zs, zchain, meta_zchain):
    '''
    针对 BAPPE 设计。

    在 BAPPE 进行中，每一步选取所有通道并从每个链上重新随机取一个 PE 序列。

    参数 offset 用来修正返回的 zchain 时间。
    '''
    for _ in range(N):
        initialize(bs_e, bs_ch, CSIZE, i_wave, location, cumc_flat, bs_ech, zs, meta_zs, zchain, meta_zchain)
        yield zs + offset, meta_zs

class FullReader(object):
    '''大批量一次性读入 FSMP 的 Full-step 结果并重整成张量，给出初值 z 表和随机更新的 z。
    '''
    def __init__(self, file, ch_num, parts=5, select=None):
        '''
        - file: FSMP full 输出的文件路径
        - ch_num: 通道总数
        - parts: 文件分成的块数
        - select: 挑选部分事例
        '''
        channels = list(range(ch_num))
        filters = [("ch", "in", channels)]
        self.parts = parts

        if select:
            selection = np.loadtxt(select, dtype=np.int64)
            filters.append(("eid", "in", selection.reshape(-1)))
        
        self.pe = pq.read_table(
            file,
            columns=["eid", "ch", "offset", "step", "PEt", "nu_lc"],
            filters=filters,
        ).to_pandas()

        # 加回 offset，并取消 offset 列
        self.pe['PEt'] = [
            np.asarray(PEt, dtype=np.float64) + offset
            for PEt, offset in zip(self.pe['PEt'], self.pe['offset'])
        ]
        self.pe['npe'] = self.pe['PEt'].map(len)
        self.pe = self.pe.drop(columns=['offset'])

        # 获取形状信息
        group_eid_ch_step = self.pe.groupby(['eid', 'ch', 'step'])
        group_eid_ch_step_size = group_eid_ch_step['npe'].max()
        self.e_z_length = group_eid_ch_step_size.groupby('eid').max().reset_index(name='max_z_length').sort_values(ascending=True, by='max_z_length')
        self.eids = self.e_z_length['eid'].values.astype(np.int64)
        self.length = len(self.eids)
        print(f"Entries: {len(self)}")

    def __len__(self):
        return self.length

    def __iter__(self):
        '''返回 (eid_block, chs, real_nch, zs, meta_zs)。'''
        # 事例分块
        NPE = self.e_z_length['max_z_length'].values
        NPE_cumsum = np.cumsum(NPE)
        splits = np.abs(NPE_cumsum[None, :] - NPE_cumsum[-1] * np.arange(0, self.parts)[:, None] / self.parts).argmin(axis=-1)
        self.eid_blocks = []
        for i in range(self.parts):
            if i < self.parts - 1:
                # 要保证每个 block 中不少于一个事例
                if splits[i] == splits[i+1]:
                    continue
                self.eid_blocks.append(self.eids[splits[i]:splits[i+1]])
            else:
                self.eid_blocks.append(self.eids[splits[i]:])

        # 分块读取
        for eid_block in self.eid_blocks:
            data = self.pe.query("eid in @eid_block")
            neid = len(eid_block)
            group_eid = data.groupby('eid')
            group_eid_ch_step = data.groupby(['step', 'eid', 'ch'])
            nch = group_eid['ch'].nunique().max()
            nzs = data['npe'].max()
            nzs_alloc = max(nzs, 1)
            nstep = data['step'].nunique()
            print(f"Reading {neid} events with maxium PE count of {nzs} ...")

            # 准备输出张量。chs 仍以 0 padding；real_nch 是判断真实前缀的
            # 唯一依据，因为通道 0 本身也是合法的真实通道。
            chs = np.zeros((neid, nch), dtype=np.int32)
            real_nch = np.zeros(neid, dtype=np.int32)
            zs = np.zeros((nstep, neid, nch, nzs_alloc), dtype=np.float32)
            meta_zs = np.zeros((nstep, neid, nch), dtype=[('s0', 'i2'), ('nu_lc', 'f4')])

            # 创建映射
            eid_to_idx = {eid: idx for idx, eid in enumerate(eid_block)}
            steps = sorted(data['step'].unique())
            step_to_idx = {step: idx for idx, step in enumerate(steps)}
            
            # 填充chs并创建ch映射
            ch_mappings = {}
            for eid, eid_data in group_eid:
                idx = eid_to_idx[eid]
                unique_chs = eid_data['ch'].unique()
                real_nch[idx] = len(unique_chs)
                chs[idx, :real_nch[idx]] = unique_chs
                ch_mappings[eid] = {ch: i for i, ch in enumerate(unique_chs)}
            
            # 填充数据
            for (step, eid, ch), group in group_eid_ch_step:
                s_idx, e_idx, c_idx = step_to_idx[step], eid_to_idx[eid], ch_mappings[eid][ch]

                # 填充PEt数据
                row = group.iloc[0]
                pe_vals = row['PEt']
                npe = len(pe_vals)
                if npe:
                    zs[s_idx, e_idx, c_idx, :npe] = pe_vals
                meta_zs[s_idx, e_idx, c_idx] = (npe, row['nu_lc'])

            yield eid_block, chs, real_nch, zs, meta_zs
                
class Reader(object):
    '''大批量一次性读入 FSMP 结果并 replay 成张量，给出初值 z 表和随机
    更新的 z。
    '''
    def __init__(self, sparsify, fsmp, ch_num, parts=5, select=None):
        '''
        - sparsify: FSMP 链初始态文件路径
        - fsmp: FSMP 输出的文件路径
        - parts: 文件分成的块数
        - N: 随机抽样次数
        - select: 挑选部分事例
        '''
        if select:
            selection = np.loadtxt(select, dtype=np.int64)
            self.sample = pq.read_table(fsmp, filters=[('eid', 'in', selection.reshape(-1))])
        else:
            self.sample = pq.read_table(fsmp)
        with h5py.File(sparsify) as ipt:
            self.s_init = ipt["s"][()]
            self.index = ipt["index"][('eid', 'ch', 'offset')]
            self.time_origin = ipt["tq"]["t_s"][:, 0]
            if select:
                cut = np.isin(self.index['eid'], selection)
                self.s_init = self.s_init[cut]
                self.index = self.index[cut]
                self.time_origin = self.time_origin[cut]
            self.i_arg = np.argsort(self.index)
        # 让分块数量至少小于事例数的一半
        parts = min(parts, len(np.unique(self.index['eid'])) // 2)

        eco = self.sample.select(["eid", "ch", "offset", "flip", "step", "s0"]).to_pandas()
        eco.replace({"flip": {2: 0}}, inplace=True) # 仅用于检查 flip 与 s0 的对应关系
        ei = eco.reset_index()
        gi = ei.groupby(['eid', 'ch'])
        l_s2 = gi["s0"].max()
        zero_chain = l_s2[l_s2 == 0].index
        if len(zero_chain):
            print("pure-zero chains found:\n", zero_chain)
            l_s2.drop(zero_chain, inplace=True)

        noffsets = gi['offset'].nunique()
        assert max(noffsets) == 1, "multiple offsets found!"
        gindex = noffsets.reset_index()[["eid", "ch"]].to_records(index=False)
        self.i_arg = self.i_arg[np.in1d(self.index[self.i_arg][["eid", "ch"]], gindex)]
        assert np.all(self.index[self.i_arg][["eid", "ch"]] == gindex), "indices not aligned"
        time_origins = self.time_origin[self.i_arg][~noffsets.index.isin(zero_chain)]
        offsets = gi['offset'].first().drop(zero_chain)
        starts = gi['index'].first().drop(zero_chain).rename("start")
        ends = gi['index'].last().drop(zero_chain).rename("end") + 1

        step = self.sample["step"].to_numpy()
        count = self.sample["count"].to_numpy()
        total_steps = step[ends-1] + count[ends-1]
        self.TRIALS = np.max(total_steps)
        self.bad_chains = np.where(total_steps < self.TRIALS)[0]
        self.bad_chains = np.append(self.bad_chains,
                                    np.where(gi["s0"].first() + gi["flip"].sum() != gi["s0"].last())[0])
        if len(self.bad_chains):
            print("the following chains failed:\n", noffsets.iloc[self.bad_chains].index)
        self.BURN = self.TRIALS // 2
        self.CSIZE = self.TRIALS - self.BURN

        mids = ei.query(f"step <= {self.BURN}").groupby(['eid', 'ch'])['index'].last().drop(zero_chain).rename("mids")
        self.chain_sizes = ends - mids

        # i_eids: 344816 -> 12360; ival_eids: 12360 -> 344816
        self.eids, ival_eids, i_eids = np.unique(starts.index.get_level_values(0), return_index=True, return_inverse=True)
        print(f"Processing {len(self)} events ...")
        ival_eids = np.append(ival_eids, len(starts))
        self.nch = eco['ch'].nunique()
        if ch_num != self.nch:
            warn(f'''
            Only {self.nch} out of {ch_num} channels have signals. This will waste some memory.
            If the channels are too sparse, consider redefine their indices at the wave_reader.
            ''')
            self.nch = ch_num
        loc = np.zeros((len(self), self.nch),
                       dtype=[("l_s", np.uint16), ("start", np.uint32), 
                              ("mid", np.uint32), ("end", np.uint32)])
        self.offset = np.zeros((len(self), self.nch), dtype=np.float32) # 与 zs 相加
        for i, (i_eid, ch, l_s, start, mid, end, ofst, time_origin) in enumerate(zip(
                i_eids, l_s2.index.get_level_values(1),
                l_s2, starts, mids, ends, offsets, time_origins)):
            if i in self.bad_chains:
                # 对于损坏的链，只取第一个由 lucyDDM 生成的 s，因此 end 是 start+1
                loc[i_eid, ch] = (l_s, start, start, start+1)
                count[start] = self.TRIALS
            else:
                loc[i_eid, ch] = (l_s, start, mid, end) # 344816 -> 142144337
            self.offset[i_eid, ch] = ofst + time_origin

        # 由指定的 parts 把事例按照事例的最大 PE 序列长度为权重分块
        e_ls = loc["l_s"].max(axis=1)
        s_ls = np.argsort(e_ls) # ls sort -> e, 12360 -> 12360
        csum_ls = np.cumsum(e_ls[s_ls])
        splits = tuple(np.unique(np.abs(
            csum_ls[None, :] - csum_ls[-1] * np.arange(1, parts)[:, None] / parts).argmin(axis=-1)))
        splits = (None,) + splits + (None,)

        self.ival_eids, self.step, self.count, self.splits = ival_eids, step, count, splits
        self.s_ls, self.e_ls, self.loc, self.N = s_ls, e_ls, loc, N

    def __len__(self):
        return len(self.eids)
    
    def maxzlength(self):
        return np.max(self.e_ls)

    def maxchlength(self):
        return np.max(np.sum(self.loc['l_s'] > 0, axis=1))

    def __iter__(self):
        self.s0 = self.sample["s0"].to_numpy()
        self.nu_lc = self.sample["nu_lc"].to_numpy()
        self.time = self.sample["time"].to_numpy()
        self.flip = self.sample["flip"].to_numpy()
        self.op = self.sample["op"].to_numpy()

        for s, e in zip(self.splits[:-1], self.splits[1:]):
            s_ev = self.s_ls[s:e] # e, 4259 / 12360
            b_ls = max(self.e_ls[s_ev]) # s size of a block
            print(f"Reading {s}:{e} events with maxium PE count of {b_ls} ...")
            ev = self.loc[s_ev, :]
            ev_hit = ev["l_s"] > 0
            ev_nch = np.sum(ev_hit, axis=1) # number of hit channels for every event
            b_nch = np.max(ev_nch) # max number of hit channels for a event in the block
            ev_chs = np.argsort(~ev_hit, axis=1, kind="stable")[:, :b_nch] # selected wave to (eid, ch) loc block tensor
            bs_e, bs_ch = np.where(ev_hit[np.arange(len(s_ev))[:, None], ev_chs])

            s_wave = np.concatenate([np.arange(*sse) for sse in zip(
                self.ival_eids[s_ev], self.ival_eids[s_ev+1])]) # selected waves
            self.i_wave = ev[ev_hit] # 与 zs[bs_e, bs_ch] 相对应
            # loc -> i_wave -> block loc
            # 我在做的其实是 awkward array
            i_chain = self.i_wave["start"]
            i_chain_end = self.i_wave["end"]
            self.l_chain = np.max(i_chain_end - i_chain) - 1

            # 每个通道的起始状态，float32 7位有效数字，1000 长的波形长度够用
            self.zs = np.full((len(s_ev), b_nch, b_ls), -1, dtype=np.float32)
            ib_ls = min(b_ls, self.s_init.shape[-1])
            self.zs[bs_e, bs_ch, :ib_ls] = self.s_init[self.i_arg[s_wave], :ib_ls]
            self.meta_zs = np.zeros((len(s_ev), b_nch),
                                    dtype=[('s0', 'u2'), ('nu_lc', 'f4')])

            # 每个通道的一条 Markov 链
            self.cur_zchain = np.zeros((len(s_ev), b_nch), dtype=np.uint16)
            l_keep = self.chain_sizes.values[s_wave].max()
            self.zchain = np.empty((len(s_ev), b_nch, l_keep, b_ls), dtype=np.float32)
            self.meta_zchain = np.zeros((len(s_ev), b_nch, l_keep),
                                        dtype=[('s0', 'u2'), ('count', 'u2'), ('nu_lc', 'f4')])
            for mid, end, e, ch in zip(self.i_wave["mid"], self.i_wave["end"], bs_e, bs_ch):
                v = self.meta_zchain[e, ch]
                v[:(end - mid)]["s0"] = self.s0[mid:end]
                v[:(end - mid)]["nu_lc"] = self.nu_lc[mid:end]
                v[:(end - mid)]["count"] = self.count[mid:end]

            self.expand(bs_e, bs_ch, i_chain, i_chain_end)
            self.cumc = self.meta_zchain["count"].cumsum(axis=2, dtype=np.uint32)
            # 下面三行纯粹是为了替代 fancy index 而加速
            self.cumc_flat = self.cumc.reshape(-1, l_keep)
            self.bs_ech = np.ravel_multi_index((bs_e, bs_ch), dims=(len(s_ev), b_nch)) # read-only
            self.location = np.empty((len(bs_e), l_keep), dtype=np.uint32) # pre-allocation for taking self.cumc
            self.initialize(bs_e, bs_ch)
            offset = self.offset[s_ev[:, None, None], ev_chs[:, :, None]]
            yield self.eids[s_ev], ev_chs, ev_nch, self.zs + offset, self.meta_zs, entries(bs_e, bs_ch, offset, self.N, self.CSIZE, self.i_wave, self.location, np.copy(self.cumc_flat), np.copy(self.bs_ech), np.copy(self.zs), np.copy(self.meta_zs), np.copy(self.zchain), np.copy(self.meta_zchain))

    def expand(self, bs_e, bs_ch, i_chain, i_chain_end):
        '''
        replay 注入 zchain, meta_zchain.

        bs_e: 从 1 维数据映射到分块的 eid 地址；
        bs_ch: 从 1 维数据映射到分块的 ch；
        i_chain: 从 1 维数据映射到链的当下位置，从 start 开始；
        i_chain_end: 从 1 维数据映射到链的终止位。
        '''

        def cz(coord):
            return coord + (self.cur_zchain[coord],)

        for _ in range(self.l_chain):
            i_chain += 1

            e_mask = i_chain >= i_chain_end
            # 如果仍然在初始态，说明链只有一步
            b_final = self.flip[i_chain[e_mask]-1] == 0
            single_step = np.any(b_final)
            assert _ == 0 or (not single_step), "flip == 0 is not on the chain head"
            if single_step:
                assert np.all(b_final), "only one case"
                # 48220677, 47164015
                coord = np.s_[bs_e[e_mask], bs_ch[e_mask]]
                self.zchain[cz(coord)] = self.zs[coord]
                self.meta_zchain["count"][cz(coord)] = self.count[i_chain[e_mask]-1] - self.BURN

            # 需要 replay 的链逐渐变少，bs_{e,ch} 变短
            i_chain, i_chain_end, bs_e, bs_ch = (i_chain[~e_mask], i_chain_end[~e_mask],
                                                 bs_e[~e_mask], bs_ch[~e_mask])

            # step0 < burn < step，实际从 burn 开始到 step
            # 补上上一个
            e_final = np.logical_and(self.step[i_chain] > self.BURN, self.step[i_chain-1] < self.BURN)
            coord = np.s_[bs_e[e_final], bs_ch[e_final]]

            self.zchain[cz(coord)] = self.zs[coord]
            self.meta_zchain["count"][cz(coord)] = self.step[i_chain[e_final]] - self.BURN
            self.cur_zchain[coord] += 1

            # 增长链
            e_plus = self.flip[i_chain] == 1
            coord = np.s_[bs_e[e_plus], bs_ch[e_plus]]
            self.zs[coord + (self.s0[i_chain[e_plus]]-1,)] = self.time[i_chain[e_plus]]

            e_minus = self.flip[i_chain] == -1
            coord = np.s_[bs_e[e_minus], bs_ch[e_minus]]
            self.zs[coord + (self.op[i_chain[e_minus]],)] = self.zs[coord + (self.s0[i_chain[e_minus]],)]

            e_move = self.flip[i_chain] == 2
            coord = np.s_[bs_e[e_move], bs_ch[e_move]]
            self.zs[coord + (self.op[i_chain[e_move]],)] = self.time[i_chain[e_move]]

            # 新的输出
            e_final = self.step[i_chain] >= self.BURN
            coord = np.s_[bs_e[e_final], bs_ch[e_final]]
            self.zchain[cz(coord)] = self.zs[coord]
            self.cur_zchain[coord] += 1

    def initialize(self, bs_e, bs_ch):
        initialize(bs_e, bs_ch, self.CSIZE, self.i_wave, self.location, self.cumc_flat, self.bs_ech, self.zs, self.meta_zs, self.zchain, self.meta_zchain)


class FSMPreader:
    '''
    读取 sparsify 的初始时间，在 FSMP 上 replay。
    '''
    def __init__(self, sparsify, fsmp):
        # 此时只能用 pandas，如果它太慢，可以转用 polars
        # 参见 https://github.com/apache/arrow/issues/14882#issuecomment-1800437413
        self.sample = pq.read_table(fsmp).to_pandas()
        with h5py.File(sparsify) as ipt:
            self.s_init = ipt["s"][()]
            self.index = ipt["index"][('eid', 'ch', 'offset')]
            self.mu0 = ipt["mu0"][()]
            self.i_arg = np.argsort(self.index)
            self.time_origin = ipt["tq"]["t_s"][:, 0]

    def __iter__(self):
        '''
        返回 (eid, ch, offset), l_s 和 chain
        '''
        eid0 = -1
        for (eid, ch, offset), body in self.flat_iter():
            if eid0 != eid:
                if eid0 >= 0:
                    yield eid0, ct
                eid0 = eid
                ct = [(ch, offset, body)] # chain table
            else:
                ct.append((ch, offset, body))
        if eid0 >= 0:
            yield eid0, ct

    def flat_iter(self):
        '''
        返回 (eid, ch, offset) 的迭代器
        '''
        i = 0
        # 可以假设 (eid, ch, offset) 良序
        for g, w_sample in self.sample.groupby(["eid", "ch", "offset"]):
            while tuple(self.index[self.i_arg[i]]) != g:
                i += 1
                # 如果找不到，i 会越界
            if "TRIALS" not in vars(self):
                self.TRIALS = w_sample['step'].values[-1] + w_sample['count'].values[-1]
                self.BURN = self.TRIALS // 2
                self.CSIZE = self.TRIALS - self.BURN

            # 读入初始状态
            l_s = w_sample['s0'].max()
            s_seed = np.empty(l_s, dtype=self.s_init.dtype)
            i_init = min(l_s, self.s_init.shape[1])
            s_seed[:i_init] = self.s_init[self.i_arg[i], :i_init]

            eid, ch, offset = g
            full_origin = offset + self.time_origin[self.i_arg[i]]
            yield (eid, ch, full_origin), (
                l_s,
                self.mu0[self.i_arg[i]],
                replay(w_sample.itertuples(index=False), s_seed.copy(), self.BURN, self.TRIALS),
            )
            i += 1

    def get_size(self):
        '''
        获取事例数和最大 z 长度
        '''
        return self.sample['eid'].nunique(), self.sample['s0'].max()

def replay(sample, s, burn, trials):
    '''
    返回 s, s0, nu_lc 和 count
    '''
    _eid, _ch, _offset, step0, flip, s00, _, _, _, _, nu_lc0, count = next(sample)
    assert step0 == 0 and flip == 0
    s0_seed = s00
    nu_lc_seed = nu_lc0
    s_seed = s.copy()
    try:
        for (_eid, _ch, _offset, step, flip, s0, op, time, _, _, nu_lc, count) in sample:
            # step0 < burn < step，实际从 burn 开始到 step
            # 补上上一个
            if step > burn and step0 < burn:
                yield (s, s00, nu_lc0), step - burn
            if flip == 1:
                s[s0-1] = time
                assert s00 + 1 == s0, "NPE overflow"
            elif flip == -1:
                s[op] = s[s0]
                assert s00 - 1 == s0, "cannot kill PE"
            elif flip == 2:
                s[op] = time
            else: # 0 最初的一步
                pass
            s00 = s0
            step0 = step
            nu_lc0 = nu_lc
            if step >= burn:
                yield (s, s0, nu_lc), count
        if flip == 0:
            # 如果仍然在初始态，说明链只有一步
            yield (s, s00, nu_lc0), count - burn
    except AssertionError:
        print(f"The Markov chain at eid {_eid}, ch {_ch}, offset {_offset} is corrupted on step {step}", file=sys.stderr)
        count = trials - max(step, burn)
        print(f"yielding {count} initial samples from LucyDDM")
        yield (s_seed, s0_seed, nu_lc_seed), count

if __name__ == '__main__':
    data = FullReader("test.pq", 17612, 5)
    # eids: 事例编号
    # chs: zs 前两个维度对应的通道号，末尾以 0 padding
    # real_nch: 每个事例在 chs 中的真实通道前缀长度
    # zs: 各事例中点亮通道的 PEt 序列
    # meta_zs: 各事例中点亮通道的 PEt 序列长度和 nu_lc
    for eids, chs, real_nch, zs, meta_zs in data:
        continue
