extern "C" __global__ void
sel_add_kern(const double *__restrict__ delta,
             const unsigned int *__restrict__ indexes, double *__restrict__ dest,
             unsigned int N, unsigned int total_add_operation) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < total_add_operation; i += blockDim.x * gridDim.x) {
    unsigned int accepted_waveform_index = i / N;
    unsigned int location_in_this = i % N;
    dest[indexes[accepted_waveform_index] * N + location_in_this] +=
        delta[accepted_waveform_index * N + location_in_this];
  }
}

extern "C" __global__ void afa_multiply_kern(double *__restrict__ afa,
                                             const int *__restrict__ ti,
                                             const double *__restrict__ frac,
                                             const double *__restrict__ af,
                                             int N, int l_s, int l_ex_t) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int row = i * (l_ex_t + 1) * l_s;
    int trow = i * l_s;
    int afarow = i * l_s * l_s;

    for (int j = 0; j < l_s; j++) {
      for (int k = 0; k < l_s; k++) {
        afa[afarow + j * l_s + k] =
          (1 - frac[trow + k]) * af[row + ((ti[trow + k] - 1 + l_ex_t) % l_ex_t) + j * (l_ex_t + 1)] +
          frac[trow + k] * af[row + ti[trow + k] + j * (l_ex_t + 1)];
      }
    }
  }
}

extern "C" __global__ void delta_w_bar_and_fa_kern(
    double *__restrict__ delta_w_bar, const int *__restrict__ ti,
    const double *__restrict__ frac, const double *__restrict__ bundle_move,
    double *__restrict__ fa, const double *__restrict__ F,
    const int *__restrict__ l_A, const double *__restrict__ sigs,
    const unsigned int *__restrict__ e_accept_indices, unsigned int N,
    unsigned int l_ex_t) {
  for (unsigned int k = blockDim.x * blockIdx.x + threadIdx.x; k < N;
       k += blockDim.x * gridDim.x) {
    unsigned int i = e_accept_indices[k];

    unsigned int row = i * (l_ex_t + 1) * 2;
    unsigned int trow = i * 2;
    unsigned int zrow = i * (l_ex_t + 1);
    unsigned int Frow = i * (l_ex_t + 1) * (l_ex_t + 1);

    int ti_del = ti[trow];
    double frac_del = frac[trow];
    int ti_add = ti[trow + 1];
    double frac_add = frac[trow + 1];

    // bundle_move < 0 for del.
    delta_w_bar[zrow + ((ti_del - 1 + l_ex_t) % l_ex_t)] -=
        (1 - frac_del) * (bundle_move[trow]);
    delta_w_bar[zrow + ti_del] -= frac_del * (bundle_move[trow]);
    delta_w_bar[zrow + ((ti_add - 1 + l_ex_t) % l_ex_t)] +=
        (1 - frac_add) * (-bundle_move[trow + 1]);
    delta_w_bar[zrow + ti_add] += frac_add * (-bundle_move[trow + 1]);

    for (int j = 0; j < l_A[i]; j++) {
      fa[row + j * 2] =
          ((1 - frac_del) *
               F[Frow + j * (l_ex_t + 1) + ((ti_del - 1 + l_ex_t) % l_ex_t)] +
           frac_del * F[Frow + j * (l_ex_t + 1) + ti_del]) /
          sigs[i];
      fa[row + j * 2 + 1] =
          ((1 - frac_add) *
               F[Frow + j * (l_ex_t + 1) + ((ti_add - 1 + l_ex_t) % l_ex_t)] +
           frac_add * F[Frow + j * (l_ex_t + 1) + ti_add]) /
          sigs[i];
    }
    fa[row + (l_ex_t - 1) * 2] =
        ((1 - frac_del) * F[Frow + (l_ex_t - 1) * (l_ex_t + 1) +
                            ((ti_del - 1 + l_ex_t) % l_ex_t)] +
         frac_del * F[Frow + (l_ex_t - 1) * (l_ex_t + 1) + ti_del]) /
        sigs[i];
    fa[row + (l_ex_t - 1) * 2 + 1] =
        ((1 - frac_add) * F[Frow + (l_ex_t - 1) * (l_ex_t + 1) +
                            ((ti_add - 1 + l_ex_t) % l_ex_t)] +
         frac_add * F[Frow + (l_ex_t - 1) * (l_ex_t + 1) + ti_add]) /
        sigs[i];
    fa[row + l_ex_t * 2] =
        ((1 - frac_del) * F[Frow + l_ex_t * (l_ex_t + 1) +
                            ((ti_del - 1 + l_ex_t) % l_ex_t)] +
         frac_del * F[Frow + l_ex_t * (l_ex_t + 1) + ti_del]) /
        sigs[i];
    fa[row + l_ex_t * 2 + 1] =
        ((1 - frac_add) * F[Frow + l_ex_t * (l_ex_t + 1) +
                            ((ti_add - 1 + l_ex_t) % l_ex_t)] +
         frac_add * F[Frow + l_ex_t * (l_ex_t + 1) + ti_add]) /
        sigs[i];
  }
}

extern "C" __global__ void beta_inversion_kern(double *__restrict__ B_inv,
                                               const double *__restrict__ B,
                                               int N) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int Brow = i * 4;
    double denominator = B[Brow] * B[Brow + 3] - B[Brow + 1] * B[Brow + 2];

    B_inv[Brow] = B[Brow + 3] / denominator;
    B_inv[Brow + 1] = -B[Brow + 1] / denominator;
    B_inv[Brow + 2] = -B[Brow + 2] / denominator;
    B_inv[Brow + 3] = B[Brow] / denominator;
  }
}

extern "C" __global__ void beye_kern(double *__restrict__ dest,
                                     const unsigned int *__restrict__ len,
                                     const double *__restrict__ coef,
                                     unsigned int batch, unsigned int width) {
  unsigned int N = batch * width;
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    unsigned int row = i / width;
    unsigned int j = i % width;

    unsigned int index = row * width * width + j * width + j;
    if (j < len[row]) {
      dest[index] += coef[row];
    } else {
      dest[index] += 1.0;
    }
  }
}

extern "C" __global__ void stable_det_kern(double *__restrict__ det_A,
                                           const double *__restrict__ A,
                                           int N) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int Arow = i * 4;
    
    det_A[i] = (A[Arow] - A[Arow + 1]) * A[Arow + 3] - (A[Arow + 2] - A[Arow + 3]) * A[Arow + 1];
  }
}

extern "C" __global__ void
bar_F_multiply_kern(double *__restrict__ bar_F, const int *__restrict__ ti,
                 const double *__restrict__ frac, const double *__restrict__ F,
                 const int *__restrict__ l_A, const unsigned int *__restrict__ e_accept_indices,
                 int N, int l_s, int l_ex_t) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    unsigned int k;
    if (!e_accept_indices) {
        k = (unsigned int)(i / l_s);
    } else {
        k = e_accept_indices[i / l_s];
    }
    unsigned int row = i * (l_ex_t + 1);
    unsigned int Frow = k * (l_ex_t + 1) * (l_ex_t + 1);

    for (int j = 0; j < l_A[k]; j++) {
      bar_F[row + j] =
          (1 - frac[i]) *
              F[Frow + ((ti[i] - 1 + l_ex_t) % l_ex_t) * (l_ex_t + 1) + j] +
          frac[i] * F[Frow + ti[i] * (l_ex_t + 1) + j];
    }
    bar_F[row + (l_ex_t - 1)] =
          (1 - frac[i]) *
              F[Frow + ((ti[i] - 1 + l_ex_t) % l_ex_t) * (l_ex_t + 1) + (l_ex_t - 1)] +
          frac[i] * F[Frow + ti[i] * (l_ex_t + 1) + (l_ex_t - 1)];
    bar_F[row + l_ex_t] =
          (1 - frac[i]) *
              F[Frow + ((ti[i] - 1 + l_ex_t) % l_ex_t) * (l_ex_t + 1) + l_ex_t] +
          frac[i] * F[Frow + ti[i] * (l_ex_t + 1) + l_ex_t];
  }
}

