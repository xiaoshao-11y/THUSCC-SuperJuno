// vmove1_fused_ref.cpp — host C++ mirror of gpu/vmove1_fused.cu.
//
// Purpose: let cpp/test_fused_formula.py validate the fused kernel's scalar
// algebra on the CPU against a NumPy transcription of upstream vmove1
// (src/analysis/fsmp/move.py:87-97 at commit 82f25ea) without any CUDA
// toolchain. Every statement below appears in the same order as the kernel
// thread body; if either file changes, change both.
//
// Build: g++ -std=c++17 -O2 -fPIC -shared -o _vmove1_fused_ref.so
//        vmove1_fused_ref.cpp   (done by the top-level Makefile)

#include <cmath>
#include <cstdint>

extern "C" {

// See gpu/vmove1_fused.cu for the layout contract. Same signatures.
int vmove1_tail_fused_ref(const double *af, const double *w_bar,
                          const double *afa, const double *fmu,
                          const double *fsig2s_inv, const double *mus,
                          const double *sig2s, double *delta_nu, double *beta,
                          std::int64_t n, std::int64_t l_ex_t) {
  const std::int64_t l = l_ex_t + 1;
  for (std::int64_t i = 0; i < n; ++i) {
    double s0 = 0.0, s1 = 0.0;
    const double *af_i = af + static_cast<std::size_t>(i) * l * 2;
    const double *w_i = w_bar + static_cast<std::size_t>(i) * l;
    for (std::int64_t j = 0; j < l; ++j) {
      s0 += af_i[0 * l + j] * w_i[j];
      s1 += af_i[1 * l + j] * w_i[j];
    }
    const double mus_over_sig2s = mus[i] / sig2s[i];

    const double inv_sig2s = 1.0 / sig2s[i];
    const double *A = afa + static_cast<std::size_t>(i) * 4;
    const double *Si = fsig2s_inv + static_cast<std::size_t>(i) * 4;
    double b00 = Si[0] + A[0] * inv_sig2s;
    double b01 = Si[1] + A[1] * inv_sig2s;
    double b10 = Si[2] + A[2] * inv_sig2s;
    double b11 = Si[3] + A[3] * inv_sig2s;
    b00 = (b00 + b00) * 0.5;
    b01 = (b01 + b10) * 0.5;
    b10 = b01;
    b11 = (b11 + b11) * 0.5;

    const double det = b00 * b11 - b01 * b10;
    const double inv_det = 1.0 / det;
    double *beta_i = beta + static_cast<std::size_t>(i) * 4;
    beta_i[0] = b11 * inv_det;
    beta_i[1] = -b01 * inv_det;
    beta_i[2] = -b10 * inv_det;
    beta_i[3] = b00 * inv_det;

    const double f0 = fmu[i * 2], f1 = fmu[i * 2 + 1];
    const double y0 = s0 * mus_over_sig2s + (f0 * Si[0] + f1 * Si[2]);
    const double y1 = s1 * mus_over_sig2s + (f0 * Si[1] + f1 * Si[3]);

    const double yby = y0 * y0 * beta_i[0] +
                       y0 * y1 * (beta_i[1] + beta_i[2]) +
                       y1 * y1 * beta_i[3];
    const double fsf = f0 * f0 * Si[0] + f0 * f1 * (Si[1] + Si[2]) +
                       f1 * f1 * Si[3];
    double nu = 0.5 * (yby - fsf);
    const double det_beta_st =
        (beta_i[0] - beta_i[1]) * beta_i[3] - (beta_i[2] - beta_i[3]) * beta_i[1];
    const double det_si_st =
        (Si[0] - Si[1]) * Si[3] - (Si[2] - Si[3]) * Si[1];
    nu += 0.5 * std::log(det_beta_st * det_si_st);
    delta_nu[i] = nu;
  }
  return 0;
}

} // extern "C"
