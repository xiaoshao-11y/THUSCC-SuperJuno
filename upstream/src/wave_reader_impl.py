import duckdb
import h5py as h5
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.testing import assert_equal

from wave_reader import WaveformRawReader, WaveformReader


class WaveformRawReaderImpl(WaveformRawReader):
    """
    Simulation 的输出 pmt_id 为 identifier，
    因此借助于 J25.3.0 的 junoid 表找到对应的 copy_number
    """

    def __init__(self, wave_file):
        super().__init__()

        with h5.File(wave_file, "r") as h:
            waves = h["CdWaveform/Waveform"][:]
            tracks = h["Sim/Track"][:]

        # 只保留 truth 中存在 e+ 的物理事例，后续所有数组共用这一事件筛选。
        eplus_eids = np.unique(tracks["id"][tracks["pdg_id"] == -11])
        waves = waves[np.isin(waves["id"], eplus_eids)]

        junoid0 = pq.read_table(
            "data/junoid_J25.3.0.pq",
            columns=["copy_number", "id"],
            filters=[[("channel", ">=", 0), ("channel", "<=", 2)]],
        ).to_pandas()
        self.junoid = junoid0.set_index("id")

        pmt_ids = waves["pmt_id"]
        copy_numbers = pd.Series(pmt_ids).map(self.junoid["copy_number"])

        # 有个别 identifier 找不到对应 copy_number，扔掉不管
        valid_mask = ~copy_numbers.isna()

        self.waves = waves[valid_mask.to_numpy()]
        self.copy_number = copy_numbers[valid_mask].astype("uint64").to_numpy()
        self.l_e = len(self.waves)

    def __len__(self) -> int:
        return self.l_e

    @property
    def eid(self) -> np.ndarray:
        return self.waves["id"]

    @property
    def ch(self) -> np.ndarray:
        return self.copy_number

    @property
    def offset(self) -> np.ndarray:
        return np.zeros(self.l_e, dtype=np.uint16)

    @property
    def l_wave(self) -> np.ndarray:
        return np.repeat(self.raw_waveform.shape[1], self.l_e)

    @property
    def raw_waveform(self) -> np.ndarray:
        return self.waves["adc"][:, :1000].astype(np.int16)


class WaveformReaderImpl(WaveformRawReaderImpl, WaveformReader):
    # J26.4.0 ElecSim with enableNewFadcParameters and negative pulses:
    # int((2**14 - 1) * (1 - 0.292)) == 11599.
    ADC_BASELINE = 11599

    def __init__(self, wave_file, baseline_file, cut=False):
        super().__init__(wave_file)

    @property
    def sig2w(self) -> np.ndarray:
        return np.repeat(4, self.l_e)

    @property
    def baseline(self) -> np.ndarray:
        return np.repeat(self.ADC_BASELINE, self.l_e)

    @property
    def waveform(self) -> np.ndarray:
        # 如果有 dynode 调用了 MCP 电荷模型，会有 Gamma 标准差为 0 从而产出 nan 的 amp
        # 对应的波形会在脉冲开始位置突然掉到 0 且一直保持到波形结束
        waveforms = self.ADC_BASELINE - self.raw_waveform
        assert self.ADC_BASELINE not in waveforms, "Invalid waveforms"
        return waveforms.astype(np.float64)
