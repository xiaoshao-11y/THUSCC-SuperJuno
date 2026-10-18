"""This module contains a (proximately) black box.

We don't need to modify it at most times.
"""

import numpy as np
import cupy as cp
import warnings
from . import vstep, update_coeff, sel_add, mmse, vmove1, vmove2, prepare_params, prepare_immediate, prepare_quadratic


def periodic(_h, _lt):
    r"""
    enforce a periodic boundary at 0 and _lt，取两次 _lt 为了防止加法的精度损失。
    np.float32(-7.1525574e-07) % np.float32(19.0) = 19.0
    (np.float32(-7.1525574e-07) % np.float32(19.0)) % np.float32(19.0) = 0
    """
    return (_h % _lt) % _lt


def v_lc(q_s, w_all, loc):
    '''
    q_s is the light curve in 1ns
    '''
    return np.log(q_s[w_all, np.array(loc, dtype=np.uint32)])


def batch(TRIALS, RECAL_STEP, z, index, spe_data, tq, s, elist, mu_t, gain, istar, erandom, flip: np.ndarray, wanders, accepts, full_writer, full_step, enable_mmse):
    r"""
    batch
    ====
    连续时间游走
    tq: proposal and prior distributions。创生与先验，取成同一分布
    s: list of PE locations
    elist: uint8 每个 PE（Photon） 对应的 MCP e index
    mu_t: light curve 的全空间积分，即其归一化系数
    U[:, :, -1]: residue waveform (raw - predicted)
    istar, erandom: shape 为 (TRIALS, l_e) 的 [0, 1) 随机数，用于点中 PE，同时可用于创生位置的选取
    MCPs = + MCP e 还是 + PE 的抉择 threshold
    flip: +1 创生一个 PE， -1 消灭一个 PE， +2 向左或向右移动
    wanders: 左右移动的多少
    accepts: metropolis 接收的 log 概率阈值
    取 (1 - ) 是为了方便 +1 PE 总概率的计算
    home_s 是在指标意义上的连续变量
    0   1   2   3   4   5   6   7   8   9   l_t = 10
    +---+---+---+---+---+---+---+---+---+
    0  0.1 0.2 0.2 0.5 0.4 0.3 0.2  0   0   q_s
    0  0.1 0.3 0.5 1.0 1.4 1.7 1.9 1.9 1.9  1.9 * cq
     _h \in [0, 9) 范围的测度是 9
    int(_h) 是插值的下指标 占比为 1 - decimal(_h)
    int(_h) + 1 是插值的上指标 占比为 decimal(_h)
    
    e: index in gain
    """
    mus = spe_data["mus"]
    sig2s = spe_data["sig2s"]
    sig2s_g = cp.asarray(sig2s, np.float64)
    sigs = cp.sqrt(sig2s_g)
    sig2w_g = cp.asarray(index["sig2w"], np.float64)
    mus_g = cp.asarray(mus, np.float64)
    # U 和 Σ 的比例
    ascoef = mus_g / sigs
    NPE = np.copy(index["s0"])
    e_overflow = np.zeros(len(NPE), dtype=np.bool_)

    l_e = len(index)
    w_all_g = cp.arange(l_e, dtype=np.uint32)  # index of all the waveforms
    w_all = np.arange(l_e)

    # 富余的 PE 活动空间
    l_s = s.shape[1]
    s_null = -np.ones((l_e, (l_s + 1) * 5), np.float64)
    e_null = -np.ones_like(s_null, np.int8)

    s = np.append(s, s_null, axis=1)
    elist = np.append(elist, e_null, axis=1)

    l_t = tq.shape[1]

    # 根据 p_cha 采样得到的 PE 序列。供以后的产生过程使用。这两行是使用了 InverseCDF 算法进行的MC采样。
    fp = np.arange(l_t)
    home_s = np.empty_like(istar)
    for i, (_xp, _lt) in enumerate(zip(tq["cq"], index["l_t"])):
        home_s[:, i] = np.interp(
            istar[:, i], xp=_xp[:_lt], fp=fp[:_lt]
        )
    home_s = periodic(home_s, index["l_t"] - 1)
    # gain["ratio"] 是 (l_e, max(nmean)) 后面补 0
    csg = np.cumsum(gain["ratio"], axis=1)
    assert np.allclose(csg[:, -1], 1.0)
    # erandom 是 (TRIALS, l_e) 的在 (0, 1) 区间的随机数
    # estar 是 (TRIALS, l_e) 的数组
    estar = (erandom[:, :, None] > csg[None, :, :]).argmin(axis=-1)

    Δν_g = cp.zeros(l_e, dtype=np.float64) # GPU 上的 Δν
    Δν = np.zeros(l_e, dtype=np.float64)
    ind_history = []
    Δν_history = [] # list of Δν's
    nu_lc_history = []
    op_history = []
    creations = []
    creations_e = []
    step_history = []
    s0_history = []  # 0-norm of s

    log_mu = np.log(np.sum(tq["q_s"], axis=1))
    loc = np.zeros((l_e, 2), np.float64)
    loc_e_mean = np.zeros((l_e, 2), np.float64)
    loc_e_sig2 = np.zeros((l_e, 2, 2), np.float64)

    # ν_lc 初始值
    nu_lc = np.zeros(l_e, dtype=np.float64)
    # 时间（light curve）项
    for iPE in range(s.shape[1]) :
        e_add_PE = iPE < NPE
        nu_lc[e_add_PE] += v_lc(tq["q_s"], e_add_PE, s[e_add_PE, iPE])
    # 泊松总强度项
    nu_lc += - mu_t
    nu_lc0 = np.copy(nu_lc)

    l_A = cp.asarray(index["l_t"], dtype=cp.int32)

    e_mu = np.append(gain["mean"], np.zeros((l_e, 1)), axis=1).astype(np.float64)
    e_sig2 = np.append(gain["sig2"], np.zeros((l_e, 1)), axis=1).astype(np.float64)

    for i, (t, new_e, step, home, wander, accept) in enumerate(
        zip(istar, estar, flip, home_s, wanders, accepts)
    ):
        if i % RECAL_STEP == 0:
            U = prepare_params(index["l_wave"], z, spe_data["ser"], tq["t_s"], index["l_t"])
            w_bar = cp.zeros((U.shape[0], U.shape[2], 1), cp.float64)
            w_bar[:, -1, :] = 1
            U = prepare_immediate(index["l_wave"], mus, spe_data["ser"], s, elist, NPE, tq["t_s"], gain, U)
            F = prepare_quadratic(U, sigs, ascoef, l_s, l_A, w_all, s, elist, NPE, sig2w_g, e_sig2)

        # 计算要操作的PE位置
        op = np.array(t * NPE, dtype=np.uint16) # in index

        # PE 时间 index，对于增加 PE 无意义
        op_s = s[w_all, op]
        # MCP 的 e index，对于增加 PE 无意义
        op_e = elist[w_all, op]
        
        e_bounce = NPE == 0
        e_normal = ~e_bounce

        # 对于loc里面设置，第一列为湮灭，第二列为产生
        if step == 1:
            loc[:, 0] = l_t # 0 A_vec
            loc[:, 1] = home[:] # 在插值时已经保证不会越界
            loc_e_mean[:, 0] = -1
            loc_e_sig2[:, 0, 0] = -1
        elif step == -1:
            loc[:, 1] = l_t # 0 A_vec
            loc_e_mean[:, 1] = 1
            loc_e_sig2[:, 1, 1] = 1
        if step in (-1, 2):
            loc[e_normal, 0] = op_s[e_normal]  # annihilate + move in ns
            loc_e_mean[e_normal, 0] = -e_mu[e_normal, op_e[e_normal]]
            loc_e_sig2[e_normal, 0, 0] = -1 / e_sig2[e_normal, op_e[e_normal]]
        if step == 2:
            loc[:, 1] = periodic(loc[:, 0] + wander,
                                  np.float64(index["l_t"][:] - 1)) # (减 1 因为 q_s 最后一位必为0)
            loc_e_mean[e_normal, 1] = e_mu[e_normal, new_e[e_normal]]
            loc_e_sig2[e_normal, 1, 1] = 1 / e_sig2[e_normal, new_e[e_normal]]
        if step == 1:
            loc_e_mean[:, 1] = e_mu[w_all, new_e]
            loc_e_sig2[:, 1, 1] = 1 / e_sig2[w_all, new_e]

        ### Δν 先验部分的计算（包括 +/- PE 的 PE 时间 proposal 计算）
        # move cases，仅泊松强度先验项中的 light curve 贡献
        Δν[:] = 0
        if step == 2:
            Δν[e_normal] += ( v_lc(tq["q_s"], e_normal, loc[e_normal, 1]) - 
                              v_lc(tq["q_s"], e_normal, loc[e_normal, 0]) )
            Δν[e_bounce] -= np.inf
        Δν_lc = Δν.copy()
        # 泊松强度先验项的 light curve 项
        if step == 1:
            Δν_lc[:] += v_lc(tq["q_s"], w_all, loc[:, 1])
        elif step == -1:
            Δν_lc[e_normal] -= v_lc(tq["q_s"], e_normal, loc[e_normal, 0])

        # non-move cases, step == 1, -1，加上泊松强度先验项以及 +/- PE 的时间 proposal 项，
        # 当 PE 时间 proposal 的形式是 light curve 限制在 tlist 定义域上时，泊松强度先验项与+/- PE 的时间 proposal 项是一个比值 mu0，取对数为 log_mu
        # 因为 mu0 是由 light curve 制作 PE 的时间 proposal 的归一化因子。见 MCP.tex 关于 h(t) 与 mu0 的定义
        if step == 1:
            Δν[:] -= np.log(NPE + 1) # 消灭的 proposal 分布项
            Δν[:] += log_mu
        if step == -1:
            Δν[e_normal] += np.log(NPE[e_normal]) # 消灭的 proposal 分布项
            Δν[e_bounce] -= np.inf
            Δν[:] -= log_mu

        ### Δν 的 Likelihood 部分计算（矩阵运算）
        ti, frac = update_coeff(l_t, cp.asarray(loc, np.float64))
        fmu_bundled_move = cp.asarray(loc_e_mean * mus[:, None], np.float64)
        fsig2s_inv = cp.asarray(loc_e_sig2 / sig2s[:, None, None], np.float64)
        Δν_g, beta = vmove1(w_bar, F, mus_g, sig2s_g, 
                            fmu_bundled_move,
                            fsig2s_inv, ti, frac, l_A)
        
        ########
        Δν += cp.asnumpy(Δν_g)
        #######

        ### 计算是否接受
        e_accept = Δν >= accept
        e_accept_indices_np = np.where(e_accept)[0]
        e_accept_g = cp.asarray(e_accept)
        e_accept_indices = cp.where(e_accept_g)[0].astype(cp.uint32)
        
        ## 更新 w_bar, F
        Δw_bar, ΔF = vmove2(cp.asarray(loc_e_mean),
                            F,
                            beta,
                            sigs,
                            ti, 
                            frac,
                            l_A,
                            e_accept_indices)
        w_bar += Δw_bar
        sel_add(ΔF, e_accept_indices, F)
        ########
        # 增加 PE
        ind = e_accept_indices_np
        ind_history.append(ind)
        if step in (1, 2):
            creations.append(loc[ind, 1])
            creations_e.append(new_e[ind])
        else:
            creations.append(np.zeros_like(ind, dtype=loc.dtype))
            creations_e.append(np.zeros_like(ind, dtype=new_e.dtype))

        if step == 1:
            ea_overflow = (NPE >= s.shape[1]) & e_accept
            e_new_overflow = ea_overflow & (~e_overflow)
            if np.any(e_new_overflow):
                warnings.warn(
                    f"""
                    NPE overflow at step {i}:
                    {index[e_new_overflow]}
                    {index.dtype}
                    """,
                    RuntimeWarning,
                )
                e_overflow |= e_new_overflow
            # 注意这里的更改是为了不影响剩下的波形，但是这些 overflow 的波形后面的链都是不正常的
            # 相关的说明参见 fsmp_replay 中为此调整的 try-except
            NPE[ea_overflow] -= 1
            s[e_accept, NPE[e_accept]] = loc[e_accept, 1]
            elist[e_accept, NPE[e_accept]] = new_e[e_accept]
            NPE[e_accept] += 1

        # op 更新
        if step in (-1, 2):
            op_history.append(op[ind])
        else:
            op_history.append(np.zeros_like(ind, dtype=op.dtype))

        # 减少 PE
        if step == -1:
            NPE[e_accept] -= 1
            s[e_accept, op[e_accept]] = s[e_accept, NPE[e_accept]]
            s[e_accept, NPE[e_accept]] = -1
            elist[e_accept, op[e_accept]] = elist[e_accept, NPE[e_accept]]
            elist[e_accept, NPE[e_accept]] = -1

        # 移动
        if step == 2:
            s[e_accept, op[e_accept]] = loc[e_accept, 1]
            elist[e_accept, op[e_accept]] = new_e[e_accept]

        Δν[~e_accept] = 0
        nu_lc[e_accept] += Δν_lc[e_accept]
        Δν_history.append(Δν[ind])
        nu_lc_history.append(nu_lc[ind])

        step_history.append(step)
        s0_history.append(NPE[ind])
        if full_writer:
            mmse_sel = e_accept
            mmse_sel_indices_np = e_accept_indices_np
            mmse_sel_indices = e_accept_indices
            if full_step is not None:
                if (i >= (TRIALS // 2)) and (((i - (TRIALS // 2)) % full_step) == 0):
                    # `append_async` 是异步的，所以要复制
                    mmse_sel = np.repeat(True, l_e)
                    mmse_sel_indices_np = w_all
                    mmse_sel_indices = w_all_g
                else:
                    continue
            s_accept = s[mmse_sel, :]
            if enable_mmse:
                ti, frac = update_coeff(l_t, cp.asarray(s_accept))
                q_accept = mmse(
                    sigs,
                    mus_g,
                    ti,
                    frac,
                    F,
                    w_bar,
                    l_A,
                    mmse_sel_indices,
                    cp.asarray(e_mu[mmse_sel_indices_np[:, None], elist[mmse_sel, :]]),
                    cp.asarray(e_sig2[mmse_sel_indices_np[:, None], elist[mmse_sel, :]]),
                ).get()
            else:
                q_accept = None
            full_writer.append_async(
                index[mmse_sel],
                tq[mmse_sel],
                i,
                NPE[mmse_sel],
                s_accept,
                elist[mmse_sel, :],
                q_accept,
                nu_lc[mmse_sel],
            )
    return (nu_lc0, ind_history, step_history, s0_history, Δν_history, nu_lc_history, op_history, creations, creations_e)
