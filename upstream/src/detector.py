"""Detector response interpolation, vertex initialization, and proposal bounds."""

import hashlib
import os
from pathlib import Path

import cupy as cp
import h5py
import numpy as np

# Detector and proposal constants.
npe = 8500  # PDE-corrected PE yield per MeV; calibration pending.
E0 = 1.1  # Energy normalization for a 1 MeV electron.
chnums = 17612  # Total number of PMT channels.
shell = 17700  # Detector radius [mm].
wavel = 1000  # Stored waveform width [ns].
t_up = 950
t_down = -200
E_down = 0.025  # Minimum active-vertex energy [MeV].
sigmas = (2, 0.005)  # Proposal widths for time and position.
rate = 2E5  # Point-process intensity [Hz].
Entries = np.log(rate * (t_up - t_down) * 1e-9)
response_integral_stride = 8  # Spacing of finite-window integral anchors [ns].


def Boundary(vertex, clip=False):
    """Return the mask of vertices inside the normalized space-time bounds.

    With ``clip=True``, also return the radii of rejected vertices for
    initialization rescaling.
    """
    radii = cp.linalg.norm(vertex[:, :3], axis=1)
    inside = (radii <= 1) & (vertex[:, -1] <= t_up) & (vertex[:, -1] >= t_down)
    if clip:
        return inside, radii[~inside]
    return inside


def Elimit(vertices, vertex_counts):
    """Return events whose active vertices all exceed the energy floor."""
    return cp.all(
        (vertices[:, :, 3] > E_down)
        | (cp.arange(vertices.shape[0])[:, None] >= vertex_counts),
        axis=0,
    )


visible_response_sum_kernel_code = r"""
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif
extern "C" {
    __global__ void visible_response_sum_kernel(
        const float* ratio1, const float* ratio2,
        const float* probe,
        const float* prefix,
        const int* index1, const int* index2,
        const int* tts_group,
        const float* vertex_time,
        float* res,
        int dim1, int dim2, int dim3, int dim4,
        int prefix_dim3, int stride, int pmt_count,
        size_t n
    ) {
        size_t linear_index = blockDim.x * blockIdx.x + threadIdx.x;
        if (linear_index >= n) return;

        int x_upper = index1[linear_index];
        int y_upper = index2[linear_index];

        int pmt_index = linear_index % pmt_count;
        int tts_group_index = tts_group[pmt_index];

        float time = vertex_time[linear_index / pmt_count];
        bool use_suffix = time < 0.0f;
        float cutoff = use_suffix ? -time : ((float) dim3 - time);
        cutoff = fminf(fmaxf(cutoff, 0.0f), (float) dim3);

        int whole_bins = (int) floorf(cutoff);
        float fraction = cutoff - (float) whole_bins;
        int prefix_block = whole_bins / stride;
        int first_residual_bin = prefix_block * stride;

        float x_weights[2] = {
            1.0f - ratio1[linear_index], ratio1[linear_index]
        };
        float y_weights[2] = {
            1.0f - ratio2[linear_index], ratio2[linear_index]
        };
        float response = 0.0f;

        for (int x_corner = 0; x_corner < 2; ++x_corner) {
            int x_bin = x_upper - 1 + x_corner;
            for (int y_corner = 0; y_corner < 2; ++y_corner) {
                int y_bin = y_upper - 1 + y_corner;
                size_t probe_base =
                    ((size_t) x_bin * dim2 + y_bin) * dim3 * dim4;
                size_t prefix_base =
                    ((size_t) x_bin * dim2 + y_bin) * prefix_dim3 * dim4;

                float integral = prefix[
                    prefix_base + (size_t) prefix_block * dim4
                    + tts_group_index
                ];
                for (int bin = first_residual_bin; bin < whole_bins; ++bin) {
                    integral += probe[
                        probe_base + (size_t) bin * dim4 + tts_group_index
                    ];
                }
                if (fraction > 0.0f) {
                    integral += fraction * probe[
                        probe_base + (size_t) whole_bins * dim4
                        + tts_group_index
                    ];
                }
                if (use_suffix) {
                    float full = prefix[
                        prefix_base + (size_t) (prefix_dim3 - 1) * dim4
                        + tts_group_index
                    ];
                    integral = full - integral;
                }
                response += x_weights[x_corner] * y_weights[y_corner] * integral;
            }
        }
        res[linear_index] = response;
    }
}
"""

visible_response_sum_kernel = cp.RawKernel(
    visible_response_sum_kernel_code, "visible_response_sum_kernel"
)

padded_hit_response_kernel_code = r"""
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif
extern "C" {
    __global__ void padded_hit_response_kernel(
        const float* pe_times,
        const short* pe_counts,
        const int* channels,
        const float* vertex_time,
        const int* x_index,
        const float* ratio_x,
        const int* y_index,
        const float* ratio_y,
        const int* tts_group,
        const float* values,
        float support_low,
        float support_high,
        float* result,
        int channel_count,
        int max_pe,
        int pmt_count,
        int dim1,
        int dim2,
        int dim3,
        int dim4,
        size_t n
    ) {
        size_t linear_index = blockDim.x * blockIdx.x + threadIdx.x;
        if (linear_index >= n) return;

        int pe_slot = linear_index % max_pe;
        size_t channel_index = linear_index / max_pe;
        if (pe_slot >= (int) pe_counts[channel_index]) {
            result[linear_index] = 0.0f;
            return;
        }

        int event_index = channel_index / channel_count;
        int pmt_index = channels[channel_index];
        size_t spatial_index =
            (size_t) event_index * pmt_count + pmt_index;

        float x_fraction = ratio_x[spatial_index];
        float y_fraction = ratio_y[spatial_index];
        int x_upper = x_index[spatial_index];
        int y_upper = y_index[spatial_index];
        int tts_group_index = tts_group[pmt_index];

        float shifted_time = pe_times[linear_index] - vertex_time[event_index];
        bool time_in_support =
            shifted_time >= support_low && shifted_time < support_high;
        float clipped_time = floorf(shifted_time);
        clipped_time = fminf(fmaxf(clipped_time, 0.0f), (float) (dim3 - 1));
        int time_bin = (int) clipped_time;
        bool last_time_bin = time_bin == dim3 - 1;
        int time_upper = last_time_bin ? time_bin : time_bin + 1;
        float time_fraction = last_time_bin ? 1.0f : 0.0f;

        float x_weights[2] = {1.0f - x_fraction, x_fraction};
        float y_weights[2] = {1.0f - y_fraction, y_fraction};
        float time_weights[2] = {1.0f - time_fraction, time_fraction};

        int x_lower = x_upper - 1;
        int y_lower = y_upper - 1;
        int time_lower = time_upper - 1;

        float v000 = values[x_lower * dim2 * dim3 * dim4 + y_lower * dim3 * dim4 + time_lower * dim4 + tts_group_index];
        float v001 = values[x_lower * dim2 * dim3 * dim4 + y_lower * dim3 * dim4 + time_upper * dim4 + tts_group_index];
        float v010 = values[x_lower * dim2 * dim3 * dim4 + y_upper * dim3 * dim4 + time_lower * dim4 + tts_group_index];
        float v011 = values[x_lower * dim2 * dim3 * dim4 + y_upper * dim3 * dim4 + time_upper * dim4 + tts_group_index];
        float v100 = values[x_upper * dim2 * dim3 * dim4 + y_lower * dim3 * dim4 + time_lower * dim4 + tts_group_index];
        float v101 = values[x_upper * dim2 * dim3 * dim4 + y_lower * dim3 * dim4 + time_upper * dim4 + tts_group_index];
        float v110 = values[x_upper * dim2 * dim3 * dim4 + y_upper * dim3 * dim4 + time_lower * dim4 + tts_group_index];
        float v111 = values[x_upper * dim2 * dim3 * dim4 + y_upper * dim3 * dim4 + time_upper * dim4 + tts_group_index];

        float response =
            x_weights[0]*y_weights[0]*time_weights[0] * v000 + x_weights[0]*y_weights[0]*time_weights[1] * v001 +
            x_weights[0]*y_weights[1]*time_weights[0] * v010 + x_weights[0]*y_weights[1]*time_weights[1] * v011 +
            x_weights[1]*y_weights[0]*time_weights[0] * v100 + x_weights[1]*y_weights[0]*time_weights[1] * v101 +
            x_weights[1]*y_weights[1]*time_weights[0] * v110 + x_weights[1]*y_weights[1]*time_weights[1] * v111;

        result[linear_index] = response * (time_in_support ? 1.0f : 0.0f);
    }
}
"""

padded_hit_response_kernel = cp.RawKernel(
    padded_hit_response_kernel_code, "padded_hit_response_kernel"
)


def VisibleResponseSum_hip(
    x_fraction,
    y_fraction,
    probe,
    prefix,
    x_index,
    y_index,
    tts_group,
    vertex_time,
    pmt_count,
    stride,
):
    """Spatially interpolate the response integral visible in the waveform."""
    element_count = x_fraction.size
    response = cp.empty_like(x_fraction, dtype=cp.float32)
    x_bins, y_bins, time_bins, tts_groups = probe.shape
    threads_per_block = 256
    blocks_per_grid = (element_count + threads_per_block - 1) // threads_per_block

    x_fraction = cp.ascontiguousarray(x_fraction, dtype=cp.float32)
    y_fraction = cp.ascontiguousarray(y_fraction, dtype=cp.float32)
    x_index = cp.ascontiguousarray(x_index, dtype=cp.int32)
    y_index = cp.ascontiguousarray(y_index, dtype=cp.int32)
    tts_group = cp.ascontiguousarray(tts_group, dtype=cp.int32)
    vertex_time = cp.ascontiguousarray(vertex_time, dtype=cp.float32)

    visible_response_sum_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            x_fraction,
            y_fraction,
            probe,
            prefix,
            x_index,
            y_index,
            tts_group,
            vertex_time,
            response,
            x_bins,
            y_bins,
            time_bins,
            tts_groups,
            prefix.shape[2],
            stride,
            pmt_count,
            element_count,
        ),
    )
    return response


def PaddedHitResponse_hip(
    pe_times,
    pe_counts,
    channel_ids,
    vertex_time,
    x_index,
    ratio_x,
    y_index,
    ratio_y,
    tts_group,
    values,
    support_low,
    support_high,
):
    """Interpolate hit densities directly into the padded PE-time tensor."""
    pe_times = cp.ascontiguousarray(cp.asarray(pe_times, dtype=cp.float32))
    pe_counts = cp.ascontiguousarray(cp.asarray(pe_counts, dtype=cp.int16))
    channel_ids = cp.ascontiguousarray(cp.asarray(channel_ids, dtype=cp.int32))
    vertex_time = cp.ascontiguousarray(cp.asarray(vertex_time, dtype=cp.float32))
    x_index = cp.ascontiguousarray(cp.asarray(x_index, dtype=cp.int32))
    ratio_x = cp.ascontiguousarray(cp.asarray(ratio_x, dtype=cp.float32))
    y_index = cp.ascontiguousarray(cp.asarray(y_index, dtype=cp.int32))
    ratio_y = cp.ascontiguousarray(cp.asarray(ratio_y, dtype=cp.float32))
    tts_group = cp.ascontiguousarray(cp.asarray(tts_group, dtype=cp.int32))
    values = cp.ascontiguousarray(cp.asarray(values, dtype=cp.float32))

    result = cp.empty_like(pe_times, dtype=cp.float32)
    element_count = result.size
    threads_per_block = 256
    blocks_per_grid = (element_count + threads_per_block - 1) // threads_per_block
    x_bins, y_bins, time_bins, tts_groups = values.shape
    padded_hit_response_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            pe_times,
            pe_counts,
            channel_ids,
            vertex_time,
            x_index,
            ratio_x,
            y_index,
            ratio_y,
            tts_group,
            values,
            np.float32(support_low),
            np.float32(support_high),
            result,
            pe_times.shape[1],
            pe_times.shape[2],
            x_index.shape[1],
            x_bins,
            y_bins,
            time_bins,
            tts_groups,
            element_count,
        ),
    )
    return result


class Probe:
    """Load and interpolate the histogram detector-response model on the GPU.

    Parameters
    ----------
    filename : str
        HDF5 file containing ``probe_t`` and its bin centers. ``probe_t`` has
        axes ``(x_bin, y_bin, time_bin, tts_group)``.
    pde : cupy.ndarray
        Per-PMT photon-detection efficiencies as fractions, shape
        ``(chnums,)``.

    PMT positions are loaded from the fixed detector-geometry table and
    normalized by ``shell``.
    """

    def __init__(self, filename, pde):
        # Convert detector positions to coordinates normalized by the radius.
        self.pmt_pos = cp.asarray(
            np.loadtxt("data/J25.3.0.PMTPos_CD_LPMT.csv")[:, 1:4] / shell,
            dtype=cp.float32,
        )
        self.pmt_r = cp.linalg.norm(self.pmt_pos, axis=1)
        self.pde = pde
        with h5py.File(filename, "r") as probe_file:
            self.probe_t = cp.asarray(probe_file["probe_t"][:])
            self.x = cp.asarray(probe_file["x_centers"][:])
            self.y = cp.asarray(probe_file["y_centers"][:])
            time_centers = probe_file["t_centers"][:]
            self.tts = probe_file["tts_group"][:]
        time_step = float(time_centers[1] - time_centers[0])
        self.t_support_low = float(time_centers[0] - time_step / 2)
        self.t_support_high = float(time_centers[-1] + time_step / 2)

        # Store one cumulative anchor per short time block.  A query reads the
        # nearest anchor and at most ``stride - 1`` original bins, so the
        # finite-window integral remains exact for the piecewise-constant 1 ns
        # histogram without materialising a copy for every vertex time.
        self.response_integral_stride = response_integral_stride
        self.tts_gpu = cp.asarray(self.tts, dtype=cp.int32)
        time_bins = self.probe_t.shape[2]
        prefix_block_count = (
            time_bins + self.response_integral_stride - 1
        ) // self.response_integral_stride
        self.probe_time_prefix = cp.empty(
            (
                self.probe_t.shape[0],
                self.probe_t.shape[1],
                prefix_block_count + 1,
                self.probe_t.shape[3],
            ),
            dtype=cp.float32,
        )
        self.probe_time_prefix[:, :, 0, :] = 0
        for block_index in range(prefix_block_count):
            bin_start = block_index * self.response_integral_stride
            bin_stop = bin_start + self.response_integral_stride
            self.probe_time_prefix[:, :, block_index + 1, :] = self.probe_time_prefix[
                :, :, block_index, :
            ] + cp.sum(
                self.probe_t[:, :, bin_start:bin_stop, :],
                axis=2,
                dtype=cp.float32,
            )
        cp.get_default_memory_pool().free_all_blocks()

    def genPos(self, vertex):
        """Map vertices to bilinear spatial-bin indices and PMT weights."""
        # Compute PMT-relative spatial coordinates.
        axial_coordinate = cp.dot(vertex[:, :3], self.pmt_pos.T) / self.pmt_r[None, :]
        transverse_coordinate = (
            cp.linalg.norm(cp.cross(vertex[:, None, :3], self.pmt_pos), axis=-1)
            / self.pmt_r[None, :]
        )

        # Locate interpolation bins and clamp weights.
        x_index = cp.clip(cp.digitize(axial_coordinate, self.x), 1, self.x.shape[0] - 1)
        ratio_x = cp.clip(
            (axial_coordinate - self.x[x_index - 1]) / (self.x[1] - self.x[0]), 0, 1
        )
        y_index = cp.clip(
            cp.digitize(transverse_coordinate, self.y), 1, self.y.shape[0] - 1
        )
        ratio_y = cp.clip(
            (transverse_coordinate - self.y[y_index - 1]) / (self.y[1] - self.y[0]),
            0,
            1,
        )

        return x_index, ratio_x, y_index, ratio_y

    def _visible_response_sum(self, vertex, x_index, ratio_x, y_index, ratio_y):
        """Expected response inside the stored waveform [0, 1000) ns."""
        event_count = len(vertex)
        pmt_count = len(self.pmt_pos)
        response = VisibleResponseSum_hip(
            ratio_x.reshape(-1),
            ratio_y.reshape(-1),
            self.probe_t,
            self.probe_time_prefix,
            x_index.reshape(-1),
            y_index.reshape(-1),
            self.tts_gpu,
            vertex[:, 4],
            pmt_count,
            self.response_integral_stride,
        )
        return cp.maximum(response.reshape(event_count, pmt_count), 0)

    def genR(self, vertex, pe_times, pe_counts, channel_ids):
        """Evaluate per-hit densities and finite-window response integrals.

        ``vertex`` has shape ``(event, 5)``, ``pe_times`` has shape
        ``(event, local_channel, PE_slot)``, ``pe_counts`` stores the valid PE
        count for each local channel, and ``channel_ids`` maps local channels
        to PMTs.
        """
        # Interpolate per-hit and integrated responses.
        x_index, ratio_x, y_index, ratio_y = self.genPos(vertex)
        visible_response_sum = self._visible_response_sum(
            vertex, x_index, ratio_x, y_index, ratio_y
        )

        channels_gpu = cp.asarray(channel_ids, dtype=cp.int32)
        channels_gpu = cp.ascontiguousarray(channels_gpu)
        hit_response = PaddedHitResponse_hip(
            pe_times,
            pe_counts,
            channels_gpu,
            vertex[:, 4],
            x_index,
            ratio_x,
            y_index,
            ratio_y,
            self.tts_gpu,
            self.probe_t,
            self.t_support_low,
            self.t_support_high,
        )

        return (
            hit_response * self.pde[channels_gpu][:, :, None],
            visible_response_sum * self.pde[None, :],
        )

    def Init(self, pe_counts, channel_ids):
        """Build one-vertex energy and charge-centroid initial states."""
        vertex = cp.zeros((pe_counts.shape[0], 5), dtype=cp.float32)
        # Initialize energy from PDE-corrected PE counts.
        channel_index = np.s_[np.arange(pe_counts.shape[0])[:, None], channel_ids]
        pde_corrected_counts = pe_counts / self.pde[None, :][channel_index]
        total_corrected_pe = cp.sum(pde_corrected_counts, axis=1)
        vertex[:, 3] = total_corrected_pe / npe
        # Initialize position from the charge centroid.
        vertex[:, :3] = (
            1.5
            * (pde_corrected_counts[:, None, :] @ self.pmt_pos[channel_ids]).squeeze(
                axis=1
            )
            / total_corrected_pe[:, None]
        )
        inside, outside_radii = Boundary(vertex, True)
        # Rescale out-of-bound centroids to radius 0.95.
        vertex[~inside, :3] = vertex[~inside, :3] / outside_radii[:, None] * 0.95
        return vertex
