"""Formula validation for the (unwired) fused vmove1 tail candidate.

Compares the host mirror cpp/vmove1_fused_ref.cpp — statement-for-statement
identical to gpu/vmove1_fused.cu — against a NumPy transcription of upstream
src/analysis/fsmp/move.py::vmove1 lines 87-97 (commit 82f25ea).

This validates the FORMULA only. It is not bit-exact by design (cuBLAS vs
sequential reduction order) and says nothing about CUDA wiring; both remain
CLUSTER_PENDING. LOCAL_PASS here means: same math as upstream within
rtol 1e-12 / atol 1e-15 on random and boundary inputs, including a
near-singular beta_inv and a 1-element l_ex_t.
"""

import ctypes
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load():
    lib = ctypes.CDLL(os.path.join(HERE, "_vmove1_fused_ref.so"))
    lib.vmove1_tail_fused_ref.restype = ctypes.c_int
    lib.vmove1_tail_fused_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # af (n, l, 2)
        ctypes.POINTER(ctypes.c_double),  # w_bar (n, l, 1)
        ctypes.POINTER(ctypes.c_double),  # afa (n, 2, 2)
        ctypes.POINTER(ctypes.c_double),  # fmu (n, 2)
        ctypes.POINTER(ctypes.c_double),  # fsig2s_inv (n, 2, 2)
        ctypes.POINTER(ctypes.c_double),  # mus (n,)
        ctypes.POINTER(ctypes.c_double),  # sig2s (n,)
        ctypes.POINTER(ctypes.c_double),  # delta_nu out
        ctypes.POINTER(ctypes.c_double),  # beta out (n, 2, 2)
        ctypes.c_int64,  # n
        ctypes.c_int64,  # l_ex_t
    ]
    return lib


def numpy_reference(af, w_bar, afa, fmu, fsig2s_inv, mus, sig2s):
    """Transcription of move.py:87-97 (the tail after afa_multiply)."""
    beta_inv = fsig2s_inv + afa / sig2s[:, None, None]
    beta_inv = (beta_inv + np.transpose(beta_inv, axes=(0, 2, 1))) / 2
    beta = np.linalg.inv(beta_inv)
    Y = (af @ w_bar).squeeze(2) * (mus / sig2s)[:, None]
    Y += (fmu[:, None, :] @ fsig2s_inv).squeeze(1)
    Δν = 0.5 * ((Y[:, None, :] @ beta @ Y[:, :, None]).squeeze((1, 2)) -
                (fmu[:, None, :] @ fsig2s_inv @ fmu[:, :, None]).squeeze((1, 2)))
    # stable_det per seladd.cu:133-142
    def stable_det(A):
        return (A[:, 0, 0] - A[:, 0, 1]) * A[:, 1, 1] - \
               (A[:, 1, 0] - A[:, 1, 1]) * A[:, 0, 1]
    Δν = Δν + 0.5 * np.log(stable_det(beta) * stable_det(fsig2s_inv))
    return Δν, beta


def run_mirror(lib, af, w_bar, afa, fmu, fsig2s_inv, mus, sig2s):
    n, _, l = af.shape
    l_ex_t = l - 1
    out_nu = np.empty(n, dtype=np.float64)
    out_beta = np.empty((n, 2, 2), dtype=np.float64)
    lib.vmove1_tail_fused_ref(
        af.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        w_bar.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        afa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fmu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fsig2s_inv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sig2s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_nu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n, l_ex_t,
    )
    return out_nu, out_beta


class TestFusedFormula(unittest.TestCase):
    RTOL, ATOL = 1e-12, 1e-15

    def make_inputs(self, n, l_ex_t, rng, scale=1.0):
        l = l_ex_t + 1
        # af mirrors bar_F_multiply output: (n, 2, l) — endpoint axis outer,
        # basis index contiguous; w_bar is (n, l, 1).
        af = rng.standard_normal((n, 2, l)) * scale
        w_bar = rng.standard_normal((n, l, 1)) * scale
        afa = rng.standard_normal((n, 2, 2)) * scale
        afa = afa + np.transpose(afa, (0, 2, 1))  # typically near-symmetric
        fmu = rng.standard_normal((n, 2)) * scale
        si = rng.standard_normal((n, 2, 2)) * scale
        si = si + np.transpose(si, (0, 2, 1))
        si += np.eye(2)[None] * 4.0  # positive-definite-ish
        mus = rng.random(n) * 40 + 1.0
        sig2s = rng.random(n) * 2 + 0.5
        return af, w_bar, afa, fmu, si, mus, sig2s

    def compare(self, inputs):
        ref_nu, ref_beta = numpy_reference(*inputs)
        got_nu, got_beta = run_mirror(_load(), *inputs)
        np.testing.assert_allclose(
            got_nu, ref_nu, rtol=self.RTOL, atol=self.ATOL)
        np.testing.assert_allclose(
            got_beta, ref_beta, rtol=self.RTOL, atol=self.ATOL)

    def test_random_typical(self):
        rng = np.random.default_rng(0)
        self.compare(self.make_inputs(256, 128, rng))

    def test_minimum_width(self):
        # l_ex_t = 1: the reduction loop has exactly two terms.
        rng = np.random.default_rng(1)
        self.compare(self.make_inputs(8, 1, rng))

    def test_wide(self):
        # l_ex_t on the order of a real time window (l_t ~ 1000).
        rng = np.random.default_rng(2)
        self.compare(self.make_inputs(64, 1000, rng))

    def test_near_singular_beta_inv(self):
        # afa ~ 0 makes beta_inv ~ fsig2s_inv; well-conditioned but exercises
        # the det=1 path contrast. Then one row with afa ~ -fsig2s_inv*sig2s
        # (near-zero beta_inv determinant) — both implementations must blow
        # up the same way, so compare against linalg with same tolerance
        # only where finite.
        rng = np.random.default_rng(3)
        af, w_bar, afa, fmu, si, mus, sig2s = self.make_inputs(16, 32, rng)
        afa[:] = 0.0
        self.compare((af, w_bar, afa, fmu, si, mus, sig2s))
        # Near-singular: scale afa so beta_inv determinant approaches zero.
        afa2 = afa.copy()
        target = -si * sig2s[:, None, None]
        afa2[:] = target * (1 - 1e-9)
        ref_nu, ref_beta = numpy_reference(af, w_bar, afa2, fmu, si, mus, sig2s)
        got_nu, got_beta = run_mirror(_load(), af, w_bar, afa2, fmu, si, mus,
                                      sig2s)
        self.assertTrue(np.all(np.isfinite(ref_nu)))
        np.testing.assert_allclose(got_nu, ref_nu, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
