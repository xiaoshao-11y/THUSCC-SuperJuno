#!/usr/bin/env python3
"""Reconstruct event vertex positions, energies, and times."""
from queue import Queue
from threading import Thread
from typing import NamedTuple

import cupy as cp
import numpy as np
from cupyx.scipy.special import logsumexp

import detector
from mcmc import Gibbs
from recon_random import ReconstructionRandom

# Structured dtype for the saved MCMC chain.
RECON_DTYPE = np.dtype(
    [
        ("EventID", "<i8"),
        ("step", "<i4"),
        ("move", "<i2"),
        ("count", "<i2"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("E", "<f4"),
        ("t", "<f4"),
        ("NPE", "<f4"),
        ("Likelihood", "<f4"),
        ("epsilon", "<f4"),
        ("acceptv", np.bool_),
    ]
)


def valid_pe_mask(pe_times, pe_counts):
    """Return the mask selecting the sampled PE slots of each local channel."""
    return cp.arange(pe_times.shape[2])[None, None, :] < pe_counts[:, :, None]


def channel_time_spread(pe_times, pe_counts, valid_mask):
    """Return the per-channel 5--95% sampled-PE time spread."""
    sorted_times = cp.sort(cp.where(valid_mask, pe_times, cp.inf), axis=2)
    event_index = cp.arange(pe_counts.shape[0])[:, None]
    channel_index = cp.arange(pe_counts.shape[1])[None, :]
    index_95 = cp.round(0.95 * (pe_counts - 1)).astype(cp.int32)
    index_5 = cp.round(0.05 * (pe_counts - 1)).astype(cp.int32)
    percentile_95 = sorted_times[event_index, channel_index, index_95]
    percentile_5 = sorted_times[event_index, channel_index, index_5]
    return cp.where(
        pe_counts > 2,
        cp.maximum(percentile_95 - percentile_5, 5),
        detector.wavel,
    )


class ChannelLayout(NamedTuple):
    """Map rectangular local-channel tensors to unique physical PMTs."""

    all_global: tuple
    real_local: tuple
    real_global: tuple
    real_mask: cp.ndarray


def channel_layout(channel_ids, real_channel_counts):
    """Build channel indices while keeping reader padding out of PMT scatter."""
    event_count, local_count = channel_ids.shape
    real_mask_host = np.arange(local_count)[None, :] < real_channel_counts[:, None]
    event_index, local_index = np.nonzero(real_mask_host)
    real_channels = channel_ids[event_index, local_index]

    event_grid = cp.arange(event_count, dtype=cp.int32)[:, None]
    all_global = (event_grid, cp.asarray(channel_ids, dtype=cp.int32))
    event_index_gpu = cp.asarray(event_index, dtype=cp.int32)
    real_local = (
        event_index_gpu,
        cp.asarray(local_index, dtype=cp.int32),
    )
    real_global = (
        event_index_gpu,
        cp.asarray(real_channels, dtype=cp.int32),
    )
    return ChannelLayout(
        all_global,
        real_local,
        real_global,
        cp.asarray(real_mask_host),
    )


class Likelihood:
    """Likelihood evaluator."""

    def __init__(self, probe, dark_rate, em_steps, lambda_high=15):
        self.probe = probe
        self.channel_layout = channel_layout
        self.dark_rate = dark_rate
        self.em_steps = em_steps
        self.lambda_high = lambda_high
        self.dark_rate_waveform = self.dark_rate * detector.wavel
        self.lambda_high_per_channel = self.lambda_high * cp.ones(
            len(self.dark_rate), dtype=cp.int16
        )

    def _observation_terms(self, pe_times, pe_counts, time_spread=None):
        """Build the response-independent terms for one FSMP realization."""
        valid_mask = valid_pe_mask(pe_times, pe_counts)
        if time_spread is None:
            time_spread = channel_time_spread(
                pe_times,
                pe_counts,
                valid_mask,
            )
        else:
            # Keep the private observation cache independent of the public
            # sampler diagnostic state.  The latter is intentionally left
            # unchanged by accepted latent-PE updates until that pre-existing
            # state-transition issue is fixed separately.
            time_spread = time_spread.copy()
        return valid_mask, time_spread

    def LH(
        self,
        vertices,
        vertex_counts,
        pe_times,
        pe_counts,
        channel_ids,
        channel_dark_rate,
        update_energy=True,
        epsilon_input=0,
        *,
        layout,
        observation_terms=None,
    ):
        """Evaluate the likelihood, optionally updating energy with EM.

        ``vertex_counts`` gives the active-vertex count for each event.
        ``channel_ids`` maps local-channel slots to physical PMT IDs. The
        returned ``L1`` contains the per-channel log likelihood used by
        latent-PE proposals, while ``L2`` contains the total log likelihood
        for each event. ``update_energy`` is false for latent-PE proposals
        because those proposals must keep the current energy fixed. ``layout``
        maps rectangular local-channel tensors to physical PMTs.
        """
        channel_index = layout.all_global
        real_local = layout.real_local
        real_global = layout.real_global

        # Allocate response tensors.
        R = cp.zeros(
            (
                vertices.shape[0],
                vertices.shape[1],
                pe_times.shape[1],
                pe_times.shape[2],
            ),
            dtype=cp.float32,
        )
        Rsum = cp.zeros(
            (vertices.shape[0], vertices.shape[1], detector.chnums), dtype=cp.float32
        )

        # Evaluate each active vertex response.
        for vertex_index, vertex in enumerate(vertices):
            active_events = vertex_counts > vertex_index
            active_events_host = active_events.get()
            if not np.any(active_events_host):
                continue
            R[vertex_index, active_events], Rsum[vertex_index, active_events] = (
                self.probe.genR(
                    vertex[active_events],
                    pe_times[active_events],
                    pe_counts[active_events],
                    channel_ids[active_events_host],
                )
            )
        lambda_high = self.lambda_high

        # Mask padded PE slots. Vertex proposals reuse the observation mask
        # and channel time spread because they do not change the observation.
        if observation_terms is None:
            valid_mask, time_spread = self._observation_terms(pe_times, pe_counts)
        else:
            valid_mask, time_spread = observation_terms
        R[:, ~valid_mask] = 0

        if update_energy:
            # Initialize the EM state.
            self.pn = cp.full(Rsum.shape[1:], 1 - 1e-4, dtype=np.float32)  # (eid, Nch)
            E = cp.ones(R.shape[:2])
            ER = E[:, :, None, None] * R
            totalER = (
                cp.sum(ER, axis=0) + channel_dark_rate
            )  # Denominator of Eq. (4.20), shape (eid, nch, s0).
            L1 = cp.sum(
                cp.log(totalER, dtype=cp.float32) * valid_mask, axis=2
            )  # Hit contribution, shape (eid, nch).
            epsilon = 1 - cp.mean(self.pn, axis=1)  # (eid,)
            lambda_total = (
                cp.sum(Rsum * E[:, :, None], axis=0) + self.dark_rate_waveform
            )
            # Compute the abnormal-hit likelihood from each channel time spread.
            abnormal_hit = cp.log(lambda_high / time_spread) * pe_counts

            # Iterate normal-channel probabilities and energy.
            for _ in range(self.em_steps):
                # Update normal-channel probabilities from energy and epsilon.
                ratio = epsilon / (1 - epsilon)  # (eid,)
                self.pn[:] = 1 / (
                    1 + (ratio[:, None] * cp.exp(-lambda_high + lambda_total))
                )
                # Update probabilities for channels with hits.
                self.pn_hit = 1 / (
                    1
                    + (
                        ratio[:, None]
                        * cp.exp(
                            (abnormal_hit - lambda_high)
                            - (L1 - lambda_total[channel_index])
                        )
                    )
                )
                self.pn[real_global] = self.pn_hit[real_local]

                # Update epsilon from the normal-channel probabilities.
                epsilon = 1 - cp.mean(self.pn, axis=1)
                epsilon = cp.clip(epsilon, 1e-7, 1 - 1e-7)

                # Update energy from the normal-channel probabilities.
                # Q assigns each hit to a vertex; normal_response_sum is the
                # corresponding integrated signal response for that vertex.
                Q = ER / totalER[None, :, :, :] * self.pn_hit[None, :, :, None]
                normal_response_sum = cp.sum(Rsum * self.pn, axis=2)
                E[:] = cp.where(
                    normal_response_sum > 0,
                    cp.sum(Q, axis=(2, 3)) / normal_response_sum,
                    0,
                )  # The integrated response can be zero in multi-vertex fits.

                # Recompute likelihood with the updated energy.
                ER = E[:, :, None, None] * R
                totalER = cp.sum(ER, axis=0) + channel_dark_rate
                L1 = cp.sum(
                    cp.log(totalER, dtype=cp.float32) * valid_mask, axis=2
                )  # Hit contribution, shape (eid, nch).
                lambda_total = (
                    cp.sum(Rsum * E[:, :, None], axis=0) + self.dark_rate_waveform
                )
        else:
            E = vertices[:, :, 3] / detector.E0
            ER = E[:, :, None, None] * R
            totalER = cp.sum(ER, axis=0) + channel_dark_rate
            L1 = cp.sum(cp.log(totalER, dtype=cp.float32) * valid_mask, axis=2)
            lambda_total = (
                cp.sum(Rsum * E[:, :, None], axis=0) + self.dark_rate_waveform
            )
            # Compute the abnormal-hit likelihood from each channel time spread.
            abnormal_hit = cp.log(lambda_high / time_spread) * pe_counts
            Q = ER / totalER[None, :, :, :]
            epsilon = epsilon_input.copy()

        # Combine the normal and abnormal channel hypotheses.
        L = cp.empty(
            (3, vertices.shape[1], len(self.dark_rate)), dtype=cp.float32
        )  # normal, abnormal, total
        L[0] = -lambda_total
        L[0][real_global] += L1[real_local]
        L[1] = -self.lambda_high_per_channel
        L[1][real_global] += abnormal_hit[real_local]
        L[2] = logsumexp(L[:2], b=cp.stack((1 - epsilon, epsilon))[:, :, None], axis=0)

        # Return total likelihood and per-channel abnormal probability.
        abnormal_probability = 1 / (
            1
            + ((1 - epsilon[:, None]) / epsilon[:, None])
            * cp.exp(L[0][channel_index] - L[1][channel_index])
        )

        return (
            L[2][channel_index],
            cp.sum(L[2], axis=1),
            E * detector.E0,
            epsilon,
            Q,
            abnormal_probability,
            time_spread,
        )


def Reconstruction(
    event_reader,
    probe,
    likelihood,
    pmt_time_correction,
    mc_steps,
    max_vertices,
    recon_dataset,
    random_seed=0,
):
    """Run reconstruction for all event blocks."""
    event_offset = 0

    def reconstruct_block(block):
        """Reconstruct one event block."""
        (
            event_ids,
            channel_ids,
            real_channel_counts,
            pe_time_samples,
            sample_metadata,
        ) = block
        sample_count = len(pe_time_samples)

        random_inputs = ReconstructionRandom(
            random_seed,
            event_ids,
            channel_ids,
            real_channel_counts,
            mc_steps,
            sample_count,
        )

        layout = likelihood.channel_layout(channel_ids, real_channel_counts)

        # Move FSMP sample tensors to the GPU.
        pe_time_samples = cp.asarray(pe_time_samples, dtype=cp.float32)
        pe_count_samples = cp.asarray(sample_metadata["s0"], dtype=cp.int16)
        channel_likelihood_samples = cp.asarray(
            sample_metadata["nu_lc"],
            dtype=cp.float32,
        )

        # Use the first FSMP realization as the initial observation.
        pe_times_without_offset = pe_time_samples[0].copy()
        pe_counts = pe_count_samples[0].copy()
        channel_likelihood = channel_likelihood_samples[0].copy()

        # Gather per-PMT calibration for each event's local channel layout.
        global_channel_ids = layout.all_global[1]
        channel_dark_rate = likelihood.dark_rate[global_channel_ids][:, :, None]
        channel_time_correction = pmt_time_correction[global_channel_ids][
            :, :, None
        ]

        # Initialize vertices.
        pe_times = pe_times_without_offset + channel_time_correction
        vertices = cp.zeros(
            (max_vertices, pe_counts.shape[0], 5),
            dtype=cp.float32,
        )
        vertices[0] = probe.Init(pe_counts, channel_ids)
        active_vertex_counts = cp.ones(len(event_ids), dtype=cp.int16)

        # Initialize the Gibbs sampler.
        sampler = Gibbs(
            mc_steps,
            len(event_ids),
            max_vertices,
            vertices,
            active_vertex_counts,
            detector.sigmas,
            likelihood,
            pe_times,
            pe_counts,
            channel_ids,
            channel_dark_rate,
            real_nch=real_channel_counts,
            channel_layout=layout,
            random_inputs=random_inputs,
        )

        # Buffer one event block in host memory and write /Recon only once.
        recon_block = np.zeros(
            (max_vertices, len(event_ids), mc_steps),
            dtype=RECON_DTYPE,
        )

        # Run Gibbs sampling.
        event_indices = cp.arange(len(event_ids), dtype=cp.int32)
        for step in range(mc_steps):
            # Propose a latent-PE realization.
            sample_indices = cp.asarray(
                random_inputs.proposal_sample_indices[step], dtype=cp.int64
            )
            log_uniform = cp.asarray(
                random_inputs.channel_log_uniforms(step), dtype=cp.float32
            )
            proposed_pe_times = (
                pe_time_samples[sample_indices, event_indices]
                + channel_time_correction
            )
            proposed_pe_counts = pe_count_samples[sample_indices, event_indices]
            proposed_channel_likelihood_samples = channel_likelihood_samples[
                sample_indices, event_indices
            ]
            proposed_L1, _, _, _, _, _, proposed_time_spread = likelihood.LH(
                sampler.vertices,
                sampler.vertex_counts,
                proposed_pe_times,
                proposed_pe_counts,
                channel_ids,
                channel_dark_rate,
                False,
                epsilon_input=sampler.epsilon,
                layout=layout,
            )
            proposed_observation = likelihood._observation_terms(
                proposed_pe_times,
                proposed_pe_counts,
                proposed_time_spread,
            )
            accepted_channels = layout.real_mask & (
                (
                    channel_likelihood
                    - proposed_channel_likelihood_samples
                    + proposed_L1
                    - sampler.L1
                )
                > log_uniform
            )
            # Apply accepted latent-PE proposals.
            pe_times[accepted_channels] = proposed_pe_times[accepted_channels]
            pe_counts[accepted_channels] = proposed_pe_counts[accepted_channels]
            channel_likelihood[accepted_channels] = (
                proposed_channel_likelihood_samples[accepted_channels]
            )
            sampler._update_observation(accepted_channels, proposed_observation)

            # Rerun EM and refresh energy and likelihood for changed events.
            accepted_events = cp.any(accepted_channels, axis=1)
            accepted_events_host = accepted_events.get()
            if np.any(accepted_events_host):
                if np.all(accepted_events_host):
                    accepted_layout = layout
                else:
                    accepted_layout = likelihood.channel_layout(
                        channel_ids[accepted_events_host],
                        real_channel_counts[accepted_events_host],
                    )
                (
                    sampler.L1[accepted_events],
                    sampler.L2[accepted_events],
                    sampler.vertices[:, accepted_events, 3],
                    _,
                    _,
                    _,
                    _,
                ) = likelihood.LH(
                    sampler.vertices[:, accepted_events],
                    sampler.vertex_counts[accepted_events],
                    pe_times[accepted_events],
                    pe_counts[accepted_events],
                    channel_ids[accepted_events_host],
                    channel_dark_rate[accepted_events],
                    observation_terms=sampler._observation_subset(accepted_events),
                    layout=accepted_layout,
                )

            # Refresh the observation before sampling vertex state.
            sampler.pe_times, sampler.pe_counts = pe_times, pe_counts
            accepted_vertex_moves, move_type = sampler.sample()

            # Copy the saved state to the host once per field and buffer it.
            count_step = sampler.vertex_counts.get()
            vertices_step = sampler.vertices.get()
            npe_step = np.sum(pe_counts.get(), axis=1, dtype=np.int32)
            likelihood_step = sampler.L2.get()
            epsilon_step = sampler.epsilon.get()
            accept_step = accepted_vertex_moves.get()

            recon_block["EventID"][:, :, step] = event_ids
            recon_block["step"][:, :, step] = step
            recon_block["count"][:, :, step] = count_step
            recon_block["x"][:, :, step] = vertices_step[:, :, 0]
            recon_block["y"][:, :, step] = vertices_step[:, :, 1]
            recon_block["z"][:, :, step] = vertices_step[:, :, 2]
            recon_block["E"][:, :, step] = vertices_step[:, :, 3]
            recon_block["t"][:, :, step] = vertices_step[:, :, 4]
            recon_block["NPE"][:, :, step] = npe_step
            recon_block["Likelihood"][:, :, step] = likelihood_step
            recon_block["epsilon"][:, :, step] = epsilon_step
            recon_block["acceptv"][:, :, step] = accept_step
            recon_block["move"][:, :, step] = move_type

        recon_dataset[
            :,
            event_offset : event_offset + len(event_ids),
            :,
        ] = recon_block
        return len(event_ids)

    # Reconstruct blocks while prefetching the next block.
    read_queue = Queue()
    sync_queue = Queue()

    def prefetch_blocks(reader):
        for block in reader:
            read_queue.put(block)
            sync_queue.get()
        read_queue.put(None)

    reader_thread = Thread(target=prefetch_blocks, args=(event_reader,))
    reader_thread.start()

    while True:
        block = read_queue.get()
        if block is None:
            break
        # Release the reader to prefetch the next block.
        sync_queue.put(None)
        event_offset += reconstruct_block(block)
