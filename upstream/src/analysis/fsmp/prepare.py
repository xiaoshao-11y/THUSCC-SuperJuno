from typing import Tuple, Generator
import os
import sys

import numpy as np
import cupy as cp
from cupy.cublas import get_batched_gesv_limit, set_batched_gesv_limit
from . import bar_F_multiply, afa_multiply, beye


def spe(t: np.ndarray, ser: np.ndarray) -> np.ndarray:
    """Interporlate SER with time.

    Parameters
    ----------
    t: ArrayLike
        The 1d time.
    ser: ArrayLike
        The 1d SER.

    Returns
    -------
    The same dimension as t.
    """
    return np.interp(t, np.arange(len(ser)), ser, 0, 0)


def prepare_params(
    l_wave: np.ndarray,
    wave: np.ndarray,
    ser: np.ndarray,
    t_s: np.ndarray,
    l_t: np.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Prepare invariant params for FSMP.

    Parameters
    ----------
    l_wave: (l_e,)
        Lengths of waveforms.
    wave: (l_e, ?)
        Waveforms.
    ser: (l_e, l_ser)
        Corresponding SER.
    t_s: (l_e, max(l_t))
        Corresponding time window.
    l_t: (l_e,)
        Lengths of time windows.

    Returns
    -------
    U: cp.ndarray (l_e, max(l_wave), max(l_t) + 2)
        The combination of A and w.
    """
    l_e = len(l_wave)
    max_l_wave = np.max(l_wave)
    max_l_t = np.max(l_t)

    U = np.zeros((l_e, max_l_wave, max_l_t + 2), np.float64)
    for i in range(l_e):
        U[i, : l_wave[i], : l_t[i]] = spe(
            np.arange(l_wave[i])[:, None] - t_s[i, : l_t[i]], ser[i]
        )
    U[:, :, -1] = wave[:, :max_l_wave]
    U = cp.asarray(U)
    return U


def prepare_immediate(
    l_wave: np.ndarray,
    mus: np.ndarray,
    ser: np.ndarray,
    s: np.ndarray,
    e: np.ndarray,
    s0: np.ndarray,
    t_s: np.ndarray,
    gain: np.ndarray,
    U: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Prepare varient params for FSMP.

    Parameters
    ----------
    l_wave: (l_e,)
        Lengths of waveforms.
    mus: (l_e,)
        Corresponding gain.
    ser: (l_e, l_ser)
        Corresponding SER.
    s: (l_e, ?)
        Corresponding initial PEs.
    s0: (l_e,)
        Corresponding NPE.
    t_s: (l_e, max(l_t))
        Corresponding time window.
    gain: (l_e,)
        Flattened gain array.
    U: cp.ndarray (l_e, max(l_wave), max(l_t) + 2)
        The combination of A and w.

    Returns
    -------
    U: cp.ndarray (l_e, max(l_wave), max(l_t) + 2)
        The combination of A and w.
    """
    l_e = len(l_wave)
    max_l_wave = np.max(l_wave)
    max_s0 = np.max(s0)

    vA = np.zeros((l_e, max_s0, max_l_wave), np.float64)
    for i in range(len(l_wave)):
        vA[i, : s0[i], : l_wave[i]] = spe(
            np.arange(l_wave[i])[None, :] - (s[i, : s0[i], None] + t_s[i, 0:1, None]),
            ser[i],
        )
    vA = cp.asarray(vA)
    fmu = cp.asarray(mus[:, None] * gain["mean"][np.arange(l_e)[:, None], e[:, :max_s0]], np.float64)
    U[:, :, -1] -= (fmu[:, None, :] @ vA).squeeze()
    return U


def prepare_quadratic(U, sigs, ascoef, l_s, l_A, w_all, s, elist, NPE, sig2w_g, e_sig2):
    l_e = U.shape[0]
    l_t = U.shape[2] - 2
    # 生成初始的A_vec
    # A_vec: (l_e, l_s, l_wave)
    frac, ti = cp.modf(cp.asarray(s[:, :l_s]))
    ti = cp.array(ti + 1, cp.int32) % (l_t + 1)

    # U: (l_e, l_wave, l_t + 2)
    U[:, :, :-1] *= sigs[:, None, None]
    U[:, :, -1] /= ascoef[:, None]
    # (l_e, l_t + 2, l_wave)
    F = cp.transpose(U, (0, 2, 1)) @ U
    del U

    # (l_e, l_s, l_t + 2)
    ATUTU = bar_F_multiply(ti, frac, F, l_A, None)
    AUUA = cp.zeros((l_e, l_s, l_s), cp.float64)
    afa_multiply(AUUA, ATUTU, ti, frac)
    # (l_e, l_s)
    sqrte = cp.sqrt(cp.asarray(e_sig2[w_all[:, None], elist[:, :l_s]], cp.float64))
    ATUTU *= sqrte[:, :, None]
    AUUA = sqrte[:, :, None] * AUUA * sqrte[:, None, :]
    beye(AUUA, cp.asarray(NPE, cp.uint32), sig2w_g)
    set_batched_gesv_limit(min(AUUA.shape[-1], get_batched_gesv_limit()))
    F -= cp.transpose(ATUTU, (0, 2, 1)) @ cp.linalg.solve(AUUA.astype(np.float64), ATUTU.astype(np.float64)).astype(np.float64)
    F /= sig2w_g[:, None, None]
    return F


FLIP_STREAM = 0
WAVEFORM_STREAM = 1


def prepare_flip(seed: int, TRIALS: int) -> np.ndarray:
    """Prepare the PCG64 action schedule shared by all waveforms."""
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, FLIP_STREAM]))
    )
    return rng.choice(
        np.array((-1, 1, 2), np.int8),
        TRIALS,
        p=np.array((1, 1, 2)) / 4,
    )


def prepare_random(seed: int, TRIALS: int, requester: np.ndarray):
    """Prepare waveform-specific PCG64 random variables for FSMP.

    Parameters
    ----------
    seed: int
        Global random seed.
    TRIALS: int
        Number of MCMC trials.
    requester: (l_e,)
        Waveform identities with ``eid`` and ``ch`` fields.

    Returns
    -------
    istar: (TRIALS, l_e)
        Normalized values used to select a PE or proposal time.
    erandom: (TRIALS, l_e)
        Uniform values used to select the new MCPe index.
    wanders: (TRIALS, l_e)
        Change of PE time when action is move.
    accepts: (TRIALS, l_e)
        Metropolis acceptance thresholds in log space.
    """
    # [SUPERJUNO-OPT] Explicit opt-in (SUPERJUNO_SJRNG=1): generate the same
    # four PCG64/SeedSequence streams with the bit-exact C++17 backend
    # (src/sjrng.py + src/_sjrng_core.so, verified bit-identical to the numpy
    # loop below by cpp/test_sjrng.py). Streams stay keyed per (EventID,
    # channel) with unchanged draw order, dtypes and logical ownership, so
    # the random mapping is untouched. Any load or generation error falls
    # back to the untouched upstream path below.
    if os.environ.get("SUPERJUNO_SJRNG") == "1":
        try:
            import sjrng

            return sjrng.prepare_random(seed, TRIALS, requester)
        except Exception as error:  # noqa: BLE001 - always keep upstream path
            print(
                f"[SUPERJUNO] sjrng unavailable, using upstream numpy path: "
                f"{error!r}",
                file=sys.stderr,
            )
    l_e = len(requester)
    istar = np.empty((TRIALS, l_e), np.float64)
    erandom = np.empty_like(istar)
    wanders = np.empty_like(istar)
    accepts = np.empty_like(istar)

    for i, (eid, ch) in enumerate(requester):
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [seed, WAVEFORM_STREAM, int(eid), int(ch)]
                )
            )
        )
        istar[:, i] = rng.random(TRIALS)
        erandom[:, i] = rng.random(TRIALS)
        wanders[:, i] = rng.standard_normal(TRIALS, np.float64)
        accepts[:, i] = np.log(rng.random(TRIALS))

    return istar, erandom, wanders, accepts


def occupy_ram(l_e, l_wave, l_t, l_NPE):
    return l_e*(229 + 108*l_NPE**2 + 4*l_t**2 + 15*l_wave + l_NPE*(253 + 18*l_t + l_wave) + l_t*(36 + 5*l_wave))


def multi_data(
    index_full, ram_limit: int
) -> Generator[Tuple[int, int, int, int, int], None, None]:
    l_e_full = len(index_full)

    loops = l_e_full
    indices = np.arange(l_e_full + 1, dtype=int)

    part = 0
    while part < loops:
        start = indices[part]
        for end_part in range(part + 1, loops + 1):
            end = indices[end_part]
            l_e0 = (end - start).item()
            index0 = index_full[start:end]
            lp_t0 = index0["l_t"].max().item()
            lp_wave0 = index0["l_wave"].max().item()
            lp_NPE0 = index0["s0"].max().item()
            # 保证至少有一个 part
            if (
                end_part == part + 1
                or occupy_ram(l_e0, lp_wave0, lp_t0, lp_NPE0) < 1000**3 * ram_limit
            ):
                l_e, lp_t, lp_wave, lp_NPE = l_e0, lp_t0, lp_wave0, lp_NPE0
            else:
                end_part -= 1
                break
        yield start, l_e, lp_t, lp_wave, lp_NPE
        part = end_part
