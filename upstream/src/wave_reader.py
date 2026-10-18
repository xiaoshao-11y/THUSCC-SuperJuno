from abc import ABCMeta, abstractmethod
from numpy import ndarray
import numpy as np


class WaveformRawReader(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def eid(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def ch(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def offset(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def l_wave(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def raw_waveform(self) -> ndarray:
        pass


class WaveformReader(WaveformRawReader):
    @property
    @abstractmethod
    def baseline(self) -> ndarray:
        pass

    @property
    def waveform(self) -> ndarray:
        waves = self.raw_waveform.astype(np.float64)
        return self.baseline[:, None] - waves
