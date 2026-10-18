import os

import cupy as cp
from . import afa_multiply, delta_w_bar_and_fa, beta_inversion, stable_det, bar_F_multiply


vstep = cp.array((-1, 1), cp.float64)


def vinterp(arr, frac, ti, w_all):
    """Interp the array with time.

    Parameters
    ----------
    arr: (E, l_wave, l_t + 1)
        A or cx
    frac: (l_e, l_s)
        The fraction of time.
    ti: (l_e, l_s)
        The intergal of time.
    w_all: (l_e, 1)
        The selected event indices.

    Returns
    -------
    (l_e, l_s, l_wave)
        vA or vc
    """
    left_arr = (1 - frac)[:, :, None] * arr[w_all, :, ti - 1]
    right_arr = frac[:, :, None] * arr[w_all, :, ti]
    return left_arr + right_arr


def update_coeff(l_t, t):
    """由 t 得到线性插值系数

    Parameters
    ----------
    l_t: int
    t: (l_e, 2)
        The time.
        
    Returns
    -------
    ti: (l_e, 2)
    frac: (l_e, 2)
    """
    l_ex_t = l_t + 1
    frac, ti = cp.modf(t)
    # avoid ti + 1 to overflow
    ti = cp.array(ti + 1, cp.int32) % l_ex_t
    return ti, frac


def vmove1(w_bar, F, mus, sig2s, fmu, fsig2s_inv, ti, frac, l_A):
    """计算 Δν

    Parameters
    ----------
    w_bar: (l_e, l_t + 2, 1)
        波形残差的线性组合系数
    F: (l_e, l_t + 2, l_t + 2)
        二次型矩阵
    mus: (l_e,)
    sig2s: (l_e,)
    fmu: (l_e, 2)
        vstep * mus
    fsig2s_inv: (l_e, 2, 2)
        vstep / sig2s
    ti: (l_e, 2)
        线性插值系数
    frac: (l_e, 2)
        线性插值系数
    l_A: (l_e,)
        波形的有效基底数

    Returns
    -------
    Δν: (l_e,)
    beta: (l_e, 2, 2)
    """
    # proposal: F @ a_bar
    # fa = cp.zeros((F.shape[0], F.shape[1], 2), dtype=cp.float64)
    # fa_multiply(fa, F, ti, frac, l_A)
    af = bar_F_multiply(ti, frac, F, l_A, None)
    # proposal: a_bar.T @ F @ a_bar
    afa = cp.zeros((F.shape[0], 2, 2), dtype=cp.float64)
    afa_multiply(afa, af, ti, frac)
    # 计算B的逆
    beta_inv = fsig2s_inv + afa / sig2s[:, None, None]
    # 修复对称性
    beta_inv = (beta_inv + cp.transpose(beta_inv, axes=(0, 2, 1))) / 2
    # 计算B
    # [SUPERJUNO-OPT] Explicit opt-in (SUPERJUNO_BETA_INV=1): replace the
    # cuSOLVER batched-LU cp.linalg.inv on (l_e, 2, 2) with the explicit
    # adjugate/determinant kernel upstream already ships but never calls
    # (seladd.cu beta_inversion_kern, wrapped as beta_inversion in seladd.py).
    # Same 2x2 inversion formula in float64, one kernel launch instead of a
    # batched solve. NOT bit-identical to LU rounding; it must pass the
    # official `scripts/run.py <id> --compare` before being trusted. Default
    # (env unset) keeps the exact upstream call.
    if os.environ.get("SUPERJUNO_BETA_INV") == "1":
        beta = cp.empty_like(beta_inv)
        beta_inversion(beta, beta_inv)
    else:
        beta = cp.linalg.inv(beta_inv)
    Y = (af @ w_bar).squeeze(2) * (mus / sig2s)[:, None]
    Y += (fmu[:, None, :] @ fsig2s_inv).squeeze(1)
    Δν = 0.5 * ((Y[:, None, :] @ beta @ Y[:, :, None]).squeeze((1, 2)) - (fmu[:, None, :] @ fsig2s_inv @ fmu[:, :, None]).squeeze((1, 2)))
    Δν += 0.5 * cp.log(stable_det(beta) * stable_det(fsig2s_inv))
    return Δν, beta


def vmove2(bundle_move, F, beta, sigs, ti, frac, l_A, e_accept_indices):
    """更新 w_bar, F

    Parameters
    ----------
    bundle_move: (l_e, 2)
        MCP 系数
    F: (l_e, l_t + 2, l_t + 2)
        二次型矩阵
    beta: (l_e, 2, 2)
    sigs: (l_e,)
    ti: (l_e, 2)
    frac: (l_e, 2)
    l_A: (l_e,)
        波形的有效基底数
    e_accept_indices: (l_e,) bool
        接受的波形

    Returns
    -------
    Δw_bar: (l_e, l_wave)
    ΔF: (l_e, l_t + 2, l_t + 2)
    """
    Δw_bar = cp.zeros((F.shape[0], F.shape[1], 1), dtype=cp.float64)
    fa = cp.zeros((F.shape[0], F.shape[1], 2), dtype=cp.float64)
    # proposal: Δw_bar = -(a_bar @ vstep) * bundle_move
    #           fa_bar = F @ a_bar / sigs
    delta_w_bar_and_fa(Δw_bar, ti, frac, bundle_move, fa, F, l_A, sigs, e_accept_indices)
    ΔF = fa[e_accept_indices] @ (-beta[e_accept_indices]) @ cp.transpose(fa[e_accept_indices], axes=(0, 2, 1))
    return Δw_bar, ΔF


def mmse(sigs, mus, ti, frac, F, w_bar, l_A, e_accept_indices, e_mu, e_sig2):
    """Calculate charges with MMSE.

    Reference: An Introduction to Signal Detection and Estimation, eBook ISBN
    978-1-4757-2341-0, Page 155, eqaution IV.B.49a

    Original Formula:
    E[q|w] = μ_q + Σ_{qw} Σ_w^-1 (w - μ_w) = μ_q + R A^T Σ^-1 (w - U(z))

    Parameters
    ----------
    sigs: (l_e,)
        Square root of variance of gains.
    mus: (l_e,)
        Gains.
    ti: (l_e_accept, l_s)
    frac: (l_e_accept, l_s)
    F: (l_e, l_t + 2, l_t + 2)
        二次型矩阵
    w_bar: (l_e, l_t + 2, 1)
        波形残差的线性组合系数
    l_A: (l_e,)
        波形的有效基底数
    e_accept_indices: (l_e_accept,) bool
        接受的波形
    e_mu: (l_e,)
        gain["mean"]
    e_sig2: (l_e,)
        gain["sig2"]

    Returns
    -------
    (l_e, l_s)
        Charges.
    """
    AF = bar_F_multiply(ti, frac, F, l_A, e_accept_indices)
    mmse_charge = mus[e_accept_indices, None] * (e_mu + (AF @ w_bar[e_accept_indices]).squeeze() * e_sig2)
    return mmse_charge
