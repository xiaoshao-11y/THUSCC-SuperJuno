import numpy as np
import cupy as cp
from itertools import chain
from cupyx.scipy.signal import fftconvolve
from ._private import maybe_tqdm


def fftconvolve_force2d(in1, in2):
    res = fftconvolve(in1, in2, mode="full", axes=1)
    if len(res.shape) == 1:
        return res[None, :]
    else:
        return res


def lucyddm(
    waveform: cp.ndarray, ser: cp.ndarray, use_tqdm=False, n=2000
) -> cp.ndarray:
    """Batched LucyDDM method

    Parameters
    ----------
    waveform: cp.ndarray (N, W)
        Multiple waveforms to deconvolution.
        The integration of waveform should be PE count instead of charge.
        In other words, the waveform should be ADC value divided by gain.
    ser: cp.ndarray (N, R)
        Multiple corresponding SER.
    use_tqdm: bool
        Whether to use tqdm.
    n: int
        The itertion counts. The default is 2000.

    Returns
    -------
    cp.ndarray (N, W + R - 1)
        The deconvolution result.
    """
    l_ser = ser.shape[1]
    (l_e, l_wave) = waveform.shape
    l_psi = l_wave + l_ser - 1
    waveform = cp.clip(waveform, 1e-6, cp.inf)
    spe = cp.clip(ser, 1e-6, cp.inf)
    wave_deconv = cp.full((l_e, l_psi), fill_value=1e-6, dtype=waveform.dtype)
    wave_deconv[:, l_ser - 1 :] = waveform
    spe_mirror = spe[:, ::-1]
    for i in maybe_tqdm(use_tqdm, range(n)):
        relative_blur = waveform / fftconvolve_force2d(wave_deconv, spe)[:, l_ser - 1 : l_psi]
        wave_deconv *= fftconvolve_force2d(relative_blur, spe_mirror)
    return wave_deconv


def multi_deconv(
    waveform: np.ndarray,
    mus: np.ndarray,
    ser: np.ndarray,
    use_tqdm=False,
    niter=2000,
) -> np.ndarray:
    """Batched deconvolution method, could be limited by ram.

    Parameters
    ----------
    waveform: (N, W)
        Multiple waveforms to deconvolution.
    mus: (N,)
        Corresponding gain values. Used to scale the waveforms.
    ser: (N, R)
        Corresponding SER.
    use_tqdm: bool
        Whether to use tqdm.
    niter: int
        The iteration counts of LucyDDM. The default is 2000.

    Returns
    -------
    np.ndarray (N, W)
        The deconvolution result.
    """
    n_wave = waveform.shape[0]
    if n_wave == 0:
        return waveform
    l_wave = waveform.shape[1]
    l_ser = ser.shape[1]
    with cp.cuda.Device(0) as d:
        vram = d.mem_info[1] # in Byte
    loops = ((l_wave + l_ser * 2) * n_wave * 8 * 20) // vram + 1
    span = n_wave // loops
    # rocFFT 的 bug：batch size 不能超过 max grid size
    if cp.cuda.runtime.is_hip and span > 65535:
        loops = n_wave // 65535 + 1
        span = n_wave // loops
    indices = np.arange(loops) * span
    chars = []
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    cache = cp.fft.config.get_plan_cache()
    for start, end in zip(indices, chain(indices[1:], [n_wave])):
        if start == end:
            continue
        char = lucyddm(
            cp.asarray(waveform[start:end] / mus[start:end, None]),
            cp.asarray(ser[start:end, :]),
            use_tqdm=use_tqdm,
            n=niter,
        )
        chars.append(cp.asnumpy(char))
        cache.clear()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    wave_deconv = np.concatenate(chars)
    return wave_deconv


def alpha_scaling(
    hitt: np.ndarray, char: np.ndarray, waveform: np.ndarray, ser: np.ndarray
):
    rec_wave = np.zeros_like(waveform)
    l_wave = len(rec_wave)
    l_ser = len(ser)
    for t, q in zip(hitt, char):
        ser_begin = max(-t, 0)
        ser_end = min(l_wave - t, l_ser)
        # 若 t < 0, -t < l_ser & -t < l_wave - t => ser_begin < ser_end
        rec_wave[max(t, 0) : max(t, 0) - ser_begin + ser_end] += q * ser[ser_begin : ser_end]
    alpha = np.dot(rec_wave, waveform) / np.dot(rec_wave, rec_wave)
    return alpha


def prior(
    char: np.ndarray,
    mus: float,
    waveform: np.ndarray,
    ser: np.ndarray,
    nsp=4,
    thres=0.2,
):
    """Use deconvolution result as prior.

    Parameters
    ----------
    char: (W + R - 1,)
        One piece of deconvolutioned waveform.
    nsp: int
        The extended width.
    thres: float
        The cut threshold.

    Returns
    -------
    t_s: (T,)
        The time window. PEs should be in this range.
    q_s: (T,)
        The prior, scaled by waveform integration.
        The last element is 0.
    cq: (T,)
        The cumulative summation of normalized q_s.
    s: (S,)
        The initial PE sequence.
    mu0: (S,)
        The etimated mu (NPE expectation)
    """

    def clip(pet, cha, thres):
        pet = pet[cha > thres]
        cha = cha[cha > thres]
        return pet, cha

    l_psi = char.shape[0]
    l_ser = len(ser)
    l_wave = l_psi - l_ser + 1
    # Cut when there's only 1 PE and it's too small
    hitt, char = clip(np.arange(1 - l_ser, l_wave), char, thres)

    if len(hitt) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    alpha = alpha_scaling(hitt, char, waveform, mus * ser)
    if alpha <= 0: # 当 LucyDDM 的边界效应过阈 thres 时，应排除。
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    char *= alpha
    char_sum = char.sum()
    # Expand hit times with nsp
    # 概率向右延展，留一空位
    tlist = np.arange(max(np.min(hitt) - nsp, 1 - l_ser), min(np.max(hitt) + 1 + nsp, l_wave))

    npe_init = np.zeros(len(tlist))
    npe_init[np.isin(tlist, hitt)] = char

    s_cha = np.cumsum(npe_init)
    # 累积分布的错位相减，等价于滑动窗口部分和。经过归一化，成为滑动平均。
    npe_init = np.pad(s_cha[3:] - s_cha[:-3], (2, 1), "edge")
    # npe_init 作为 light_curve 和 proposal distribution 输入。需要让
    # 它有一个非零的数值，否则 PE 被移动到 npe_init==0 的地方，概率会
    # 成为 0，随机游走的遍历性被破坏。
    npe_init += 1e-8
    # 右边界管辖的概率不在 tlist 区间中，丢弃
    npe_init[-1] = 0
    p_cha = npe_init / np.sum(npe_init)
    # q_s: pdf of LucyDDM charge (由 charge 引导的 PE 强度流先验)
    cq = np.cumsum(np.roll(p_cha, 1))
    cq[-1] = 1

    l_t = len(tlist)
    NPE = round(char_sum)
    # t 的初始位置，取值为 {0.5, 1.5, 2.5, ..., (NPE-0.5)} / NPE 的 ，由 LucyDDM 给出的 PE 当做 light curve 的 InverseCDF
    # MCMC 链的 PE configuration 初值 s0
    s = np.interp((np.arange(NPE) + 0.5) / NPE, xp=cq, fp=np.arange(l_t))
    return tlist, p_cha * char_sum, cq, s, char_sum
