// vmove1_fused.cu — REVIEW-ONLY CUDA C++ optimization candidate.
//
// NOT WIRED INTO THE RUNTIME. This file is a design artifact for fusing the
// tail of upstream src/analysis/fsmp/move.py::vmove1 (upstream commit
// 82f25ea, lines 87-97) into one kernel per MCMC step. It is not compiled
// by the pipeline and not enabled by any environment variable in this
// repository; it must first pass the official
//     python scripts/run.py <id> --compare
// on the cluster exactly as documented in IMPLEMENTATION.md before any
// wiring is added.
//
// What it would replace (upstream, per MCMC step, for l_e waveforms):
//   move.py:88   beta_inv = fsig2s_inv + afa / sig2s[:, None, None]   (CuPy elementwise)
//   move.py:90   beta_inv = (beta_inv + beta_inv.T) / 2               (CuPy elementwise)
//   move.py:92   beta = cp.linalg.inv(beta_inv)                       (cuSOLVER batched LU on (l_e,2,2))
//   move.py:93   Y = (af @ w_bar).squeeze(2) * (mus / sig2s)[:, None] (cuBLAS batched GEMV)
//   move.py:94   Y += (fmu[:, None, :] @ fsig2s_inv).squeeze(1)       (cuBLAS batched GEMM)
//   move.py:95   Δν = 0.5 * (Yᵀ β Y − fμᵀ Σ⁻¹ fμ)                     (two batched GEMMs)
//   move.py:96   Δν += 0.5 * log(stable_det(β) * stable_det(Σ⁻¹))     (two kernels + elementwise)
// i.e. roughly eight launches and several (l_e, 2, 2)/(l_e, 2) temporaries
// per step — at 4000 steps per FSMP part this launch and temporary traffic
// is the motivation. afa and af themselves already come from upstream
// kernels (afa_multiply_kern, bar_F_multiply_kern) and stay untouched.
//
// Numerical caveat (why this stays opt-out until cluster-validated): the
// j-reduction of af·w_bar is performed by this kernel in a fixed ascending
// order, which will NOT match cuBLAS's internal reduction order, so Δν can
// differ from upstream in the last ulps. As documented in docs/recon.tex
// ("唯一参考与结果比较"), ulp-level differences may flip borderline
// Metropolis acceptances and diverge the chain; acceptance must therefore
// be decided by the official PE/tvE comparators, never assumed.
//
// Layouts (all contiguous, float64, matching upstream):
//   af            (l_e, l_ex_t + 1, 2)   bar_F_multiply output
//   w_bar         (l_e, l_ex_t + 1, 1)   flat to (l_e, l_ex_t + 1)
//   afa           (l_e, 2, 2)            afa_multiply output (pre-symmetry)
//   fmu           (l_e, 2)               loc_e_mean * mus
//   fsig2s_inv    (l_e, 2, 2)            loc_e_sig2 / sig2s
//   mus, sig2s    (l_e,)
// Out:
//   delta_nu      (l_e,)
//   beta          (l_e, 2, 2)            needed later by vmove2
//
// The scalar algebra below is mirrored statement-for-statement in
// cpp/vmove1_fused_ref.cpp (host C++), which cpp/test_fused_formula.py
// checks against a NumPy transcription of vmove1 (rtol 1e-12, plus a
// near-singular boundary case). That validates the formula only — CUDA
// indexing and wiring remain cluster-pending.

extern "C" __global__ void vmove1_tail_fused_kern(
    const double *__restrict__ af, const double *__restrict__ w_bar,
    const double *__restrict__ afa, const double *__restrict__ fmu,
    const double *__restrict__ fsig2s_inv, const double *__restrict__ mus,
    const double *__restrict__ sig2s, double *__restrict__ delta_nu,
    double *__restrict__ beta, int n, int l_ex_t) {
  // One thread per waveform; grid-stride so any launch geometry works.
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    const int l = l_ex_t + 1; // rows of af / w_bar per waveform

    // Y first term: (af @ w_bar) in ascending j order (see caveat above).
    // af is (l_e, 2, l_ex_t + 1) — the move-endpoint axis is outer and the
    // basis index j is contiguous, matching bar_F_multiply_kern's output.
    double s0 = 0.0, s1 = 0.0;
    const double *af_i = af + (size_t)i * l * 2;
    const double *w_i = w_bar + (size_t)i * l;
    for (int j = 0; j < l; ++j) {
      s0 += af_i[0 * l + j] * w_i[j];
      s1 += af_i[1 * l + j] * w_i[j];
    }
    const double mus_over_sig2s = mus[i] / sig2s[i];

    // beta_inv = fsig2s_inv + afa / sig2s, symmetrized (move.py:88-90).
    const double inv_sig2s = 1.0 / sig2s[i];
    const double *A = afa + (size_t)i * 4;         // row-major 2x2
    const double *Si = fsig2s_inv + (size_t)i * 4; // row-major 2x2
    double b00 = Si[0] + A[0] * inv_sig2s;
    double b01 = Si[1] + A[1] * inv_sig2s;
    double b10 = Si[2] + A[2] * inv_sig2s;
    double b11 = Si[3] + A[3] * inv_sig2s;
    b00 = (b00 + b00) * 0.5; // symmetrize: (M + Mᵀ)/2 keeps 00/11 in place
    b01 = (b01 + b10) * 0.5;
    b10 = b01;
    b11 = (b11 + b11) * 0.5;

    // beta = adj(B)/det(B) — the same explicit formula as upstream's own
    // unused beta_inversion_kern (seladd.cu:99-112).
    const double det = b00 * b11 - b01 * b10;
    const double inv_det = 1.0 / det;
    double *beta_i = beta + (size_t)i * 4;
    beta_i[0] = b11 * inv_det;
    beta_i[1] = -b01 * inv_det;
    beta_i[2] = -b10 * inv_det;
    beta_i[3] = b00 * inv_det;

    // Y = (af @ w_bar) * (mus/sig2s) + fmu @ fsig2s_inv (move.py:93-94).
    const double f0 = fmu[i * 2], f1 = fmu[i * 2 + 1];
    const double y0 = s0 * mus_over_sig2s +
                      (f0 * Si[0] + f1 * Si[2]);
    const double y1 = s1 * mus_over_sig2s +
                      (f0 * Si[1] + f1 * Si[3]);

    // Δν = 0.5 * (YᵀβY − fμᵀΣ⁻¹fμ) (move.py:95), then the log-det term
    // (move.py:96) using upstream's stable_det formula (seladd.cu:133-142):
    //   det_st(A) = (A00 − A01)·A11 − (A10 − A11)·A01
    const double yby = y0 * y0 * beta_i[0] +
                       y0 * y1 * (beta_i[1] + beta_i[2]) +
                       y1 * y1 * beta_i[3];
    const double fsf = f0 * f0 * Si[0] + f0 * f1 * (Si[1] + Si[2]) +
                       f1 * f1 * Si[3];
    double nu = 0.5 * (yby - fsf);
    const double det_beta_st =
        (beta_i[0] - beta_i[1]) * beta_i[3] -
        (beta_i[2] - beta_i[3]) * beta_i[1];
    const double det_si_st =
        (Si[0] - Si[1]) * Si[3] - (Si[2] - Si[3]) * Si[1];
    nu += 0.5 * log(det_beta_st * det_si_st);
    delta_nu[i] = nu;
  }
}

// Suggested host wiring (NOT used anywhere yet): a CuPy RawModule compiled
// from this file at first use inside a new opt-in branch
// (e.g. SUPERJUNO_VMOVE1_FUSED=1) in move.py::vmove1, replacing lines
// 88-97 but leaving bar_F_multiply/afa_multiply inputs and the returned
// (Δν, beta) contract identical, with a hard fallback to the current
// implementation on any compile or launch failure. Grid: 1024 threads per
// block, ceil(l_e / 1024) blocks, matching seladd.py's launch style.
