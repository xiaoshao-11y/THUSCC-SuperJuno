"""Gibbs/RJMCMC sampler specialized for one- and two-vertex events."""

import cupy as cp
import numpy as np

import detector

FLOAT32_KWARGS = {"dtype": cp.float32}
BIRTH_DEATH_PROBABILITY = 0.4


def make_move_schedule(
    mc_steps,
    max_vertices,
    dimension_uniforms,
    scale_low=5,
    scale_mid=20,
    scale_high=40,
):
    """Build the dimension-move schedule from fixed CPU random inputs."""
    proposal_scale = cp.ones(mc_steps, dtype=cp.float32)
    step_indices = cp.arange(mc_steps, dtype=cp.int32)
    warmup_mask = step_indices < (mc_steps / 10)
    proposal_scale[warmup_mask] = cp.cos(
        10 * step_indices[warmup_mask] / mc_steps * cp.pi
    ) * (scale_mid / 2 - 0.5) + (scale_mid / 2 + 0.5)

    dimension_moves = np.zeros(mc_steps, dtype=np.int32)
    if max_vertices == 2:
        dimension_uniforms = np.asarray(dimension_uniforms, dtype=np.float32)
        if dimension_uniforms.shape != (mc_steps,):
            raise ValueError("dimension_uniforms has an unexpected shape")
        dimension_moves[dimension_uniforms < BIRTH_DEATH_PROBABILITY] = 1
        dimension_moves[
            (dimension_uniforms >= BIRTH_DEATH_PROBABILITY)
            & (dimension_uniforms < 2 * BIRTH_DEATH_PROBABILITY)
        ] = -1
        # Front-load dimension changes during the part of the chain discarded
        # as burn-in so two-vertex states are reached more quickly.
        dimension_moves[: (mc_steps // 10)] = 0
        dimension_moves[(mc_steps // 10) : (mc_steps // 4)] = 1
        proposal_scale[(mc_steps // 10) : (mc_steps // 5)] = scale_high
        proposal_scale[(mc_steps // 5) : (mc_steps // 4)] = scale_mid
        proposal_scale[(mc_steps // 4) : (mc_steps // 3)] = scale_low

    return dimension_moves, proposal_scale


class Gibbs:
    """Sampler state for vertex dimension, position, time, and energy."""

    def __init__(
        self,
        mc_steps,
        event_count,
        max_vertices,
        vertices,
        vertex_counts,
        proposal_scales,
        likelihood,
        pe_times,
        pe_counts,
        channels,
        local_dark_rates,
        real_nch,
        channel_layout,
        random_inputs,
    ):
        """Initialize proposal arrays and the current likelihood state."""
        self.log_acceptance_uniforms = cp.asarray(
            random_inputs.log_acceptance_uniforms, dtype=cp.float32
        )
        self.proposal_normals = cp.asarray(
            random_inputs.proposal_normals, dtype=cp.float32
        )
        self.vertex_selection_uniforms = cp.asarray(
            random_inputs.vertex_selection_uniforms, dtype=cp.float32
        )
        self.birth_vertices = cp.asarray(
            random_inputs.birth_vertices, dtype=cp.float32
        )
        dimension_moves, self.proposal_scale = make_move_schedule(
            mc_steps, max_vertices, random_inputs.dimension_uniforms
        )
        self.vertices = vertices
        self.vertex_counts = vertex_counts
        self.likelihood = likelihood
        self.pe_times = pe_times
        self.pe_counts = pe_counts
        self.channels = channels
        self.real_channel_counts = real_nch
        self.channel_layout = channel_layout
        self.local_dark_rates = local_dark_rates
        (
            self.L1,
            self.L2,
            self.vertices[:, :, 3],
            self.epsilon,
            self.Q,
            self.abn,
            self.wavel_ch,
        ) = self.likelihood.LH(
            vertices,
            vertex_counts,
            pe_times,
            pe_counts,
            channels,
            local_dark_rates,
            layout=channel_layout,
        )
        self._observation_cache = self.likelihood._observation_terms(
            pe_times, pe_counts, self.wavel_ch
        )
        self.dimension_moves = dimension_moves
        self.time_sigma, self.position_sigma = proposal_scales
        self.step = 0

    def _observation_subset(self, event_mask):
        """Return the cached observation terms for an event subset."""
        return tuple(value[event_mask] for value in self._observation_cache)

    def _update_observation(self, channel_mask, proposed_terms):
        """Apply accepted channel-wise latent-PE updates to the private cache."""
        for current, proposed in zip(self._observation_cache, proposed_terms):
            current[channel_mask] = proposed[channel_mask]

    def _channel_subset(self, event_mask):
        """Return channel rows and their layout for an event subset."""
        event_mask_host = event_mask.get()
        if np.all(event_mask_host):
            return self.channels, self.channel_layout
        subset_channels = self.channels[event_mask_host]
        subset_layout = self.likelihood.channel_layout(
            subset_channels, self.real_channel_counts[event_mask_host]
        )
        return subset_channels, subset_layout

    def _propose_vertex(self, current_vertices, move_kind):
        """Apply a position or time proposal to selected vertices."""
        proposed_vertices = current_vertices.copy()
        event_count = len(current_vertices)
        if move_kind == "P":
            proposed_vertices[:, :3] = (
                current_vertices[:, :3]
                + self.position_sigma
                * self.proposal_normals[self.step, :event_count, :3]
                * self.proposal_scale[self.step]
            )
        elif move_kind == "T":
            proposed_vertices[:, -1] = (
                current_vertices[:, -1]
                + self.time_sigma
                * self.proposal_normals[self.step, :event_count, -1]
                * self.proposal_scale[self.step]
            )
        return proposed_vertices

    def _move_vertices(self, vertex_indices, event_mask, move_kind):
        """Propose and Metropolis-test one vertex move per selected event."""
        proposed_vertices = self.vertices[:, event_mask].copy()
        selected_vertices = cp.s_[
            vertex_indices[event_mask], cp.arange(proposed_vertices.shape[1])
        ]
        proposed_vertices[selected_vertices] = self._propose_vertex(
            proposed_vertices[selected_vertices], move_kind
        )
        within_boundary = detector.Boundary(proposed_vertices[selected_vertices])
        subset_channels, subset_layout = self._channel_subset(event_mask)
        (
            proposed_channel_likelihood,
            proposed_event_likelihood,
            proposed_vertices[:, :, 3],
            proposed_epsilon,
            proposed_assignment_probabilities,
            proposed_abn,
            proposed_wavel_ch,
        ) = self.likelihood.LH(
            proposed_vertices,
            self.vertex_counts[event_mask],
            self.pe_times[event_mask],
            self.pe_counts[event_mask],
            subset_channels,
            self.local_dark_rates[event_mask],
            observation_terms=self._observation_subset(event_mask),
            layout=subset_layout,
        )
        acceptance_uniform_slot = {"P": 1, "T": 2}[move_kind]
        likelihood_accepted = (
            proposed_event_likelihood - self.L2[event_mask]
        ) > self.log_acceptance_uniforms[self.step, event_mask, acceptance_uniform_slot]
        above_energy_floor = detector.Elimit(
            proposed_vertices, self.vertex_counts[event_mask]
        )
        accepted_subset = within_boundary & likelihood_accepted & above_energy_floor
        accepted_event_mask = event_mask.copy()
        accepted_event_mask[event_mask] = accepted_subset

        self.vertices[:, accepted_event_mask] = proposed_vertices[:, accepted_subset]
        self.L2[accepted_event_mask] = proposed_event_likelihood[accepted_subset]
        self.L1[accepted_event_mask] = proposed_channel_likelihood[accepted_subset]
        self.epsilon[accepted_event_mask] = proposed_epsilon[accepted_subset]
        self.Q[:, accepted_event_mask] = proposed_assignment_probabilities[
            :, accepted_subset
        ]
        self.abn[accepted_event_mask] = proposed_abn[accepted_subset]
        self.wavel_ch[accepted_event_mask] = proposed_wavel_ch[accepted_subset]
        return accepted_subset

    def sample(self):
        """Run one dimension, position, and time sweep for every event."""
        vertex_indices = (
            self.vertex_counts
            * self.vertex_selection_uniforms[self.step, :, 0]
        )
        vertex_indices = vertex_indices.astype(cp.int16)
        dimension_move = int(self.dimension_moves[self.step])

        if dimension_move == 1:
            # Propose births only for events below the vertex-count limit.
            eligible_events = self.vertex_counts < len(self.vertices)
            proposed_vertices = self.vertices[:, eligible_events].copy()
            if proposed_vertices.shape[1] > 0:
                new_vertex_indices = cp.s_[
                    self.vertex_counts[eligible_events],
                    cp.arange(proposed_vertices.shape[1]),
                ]
                proposed_vertices[new_vertex_indices] = self.birth_vertices[
                    self.step, eligible_events
                ]
                within_boundary = detector.Boundary(
                    proposed_vertices[new_vertex_indices]
                )
                subset_channels, subset_layout = self._channel_subset(eligible_events)
                (
                    proposed_channel_likelihood,
                    proposed_event_likelihood,
                    proposed_vertices[:, :, 3],
                    proposed_epsilon,
                    proposed_assignment_probabilities,
                    proposed_abn,
                    proposed_wavel_ch,
                ) = self.likelihood.LH(
                    proposed_vertices,
                    self.vertex_counts[eligible_events] + 1,
                    self.pe_times[eligible_events],
                    self.pe_counts[eligible_events],
                    subset_channels,
                    self.local_dark_rates[eligible_events],
                    observation_terms=self._observation_subset(eligible_events),
                    layout=subset_layout,
                )
                likelihood_accepted = (
                    proposed_event_likelihood
                    - self.L2[eligible_events]
                    + detector.Entries
                    - cp.log(
                        self.vertex_counts[eligible_events] + 1,
                        **FLOAT32_KWARGS,
                    )
                ) > self.log_acceptance_uniforms[self.step, eligible_events, 0]
                above_energy_floor = detector.Elimit(
                    proposed_vertices, self.vertex_counts[eligible_events] + 1
                )
                accepted_subset = (
                    within_boundary & likelihood_accepted & above_energy_floor
                )
                eligible_events[eligible_events] = accepted_subset

                self.vertices[:, eligible_events] = proposed_vertices[
                    :, accepted_subset
                ]
                self.L2[eligible_events] = proposed_event_likelihood[accepted_subset]
                self.L1[eligible_events] = proposed_channel_likelihood[accepted_subset]
                self.epsilon[eligible_events] = proposed_epsilon[accepted_subset]
                self.Q[:, eligible_events] = proposed_assignment_probabilities[
                    :, accepted_subset
                ]
                self.abn[eligible_events] = proposed_abn[accepted_subset]
                self.wavel_ch[eligible_events] = proposed_wavel_ch[accepted_subset]
                self.vertex_counts[eligible_events] += 1
            accepted_event_mask = eligible_events

        elif dimension_move == -1:
            # Propose deaths only for events with more than one active vertex.
            eligible_events = self.vertex_counts > 1
            proposed_vertices = self.vertices[:, eligible_events].copy()
            if proposed_vertices.shape[1] > 0:
                # Active vertices must occupy the contiguous prefix ``[:count]``.
                # Removing an arbitrary slot by only zeroing it breaks that
                # invariant: when slot 0 is removed from a two-vertex state,
                # slot 1 would be ignored after count becomes 1.  Move the last
                # active vertex into the removed slot before clearing the tail.
                event_indices = cp.arange(proposed_vertices.shape[1])
                removed_vertex_indices = vertex_indices[eligible_events]
                last_vertex_indices = self.vertex_counts[eligible_events] - 1
                proposed_vertices[removed_vertex_indices, event_indices] = (
                    proposed_vertices[last_vertex_indices, event_indices].copy()
                )
                proposed_vertices[last_vertex_indices, event_indices] = 0
                subset_channels, subset_layout = self._channel_subset(eligible_events)
                (
                    proposed_channel_likelihood,
                    proposed_event_likelihood,
                    proposed_vertices[:, :, 3],
                    proposed_epsilon,
                    proposed_assignment_probabilities,
                    proposed_abn,
                    proposed_wavel_ch,
                ) = self.likelihood.LH(
                    proposed_vertices,
                    self.vertex_counts[eligible_events] - 1,
                    self.pe_times[eligible_events],
                    self.pe_counts[eligible_events],
                    subset_channels,
                    self.local_dark_rates[eligible_events],
                    observation_terms=self._observation_subset(eligible_events),
                    layout=subset_layout,
                )
                likelihood_accepted = (
                    proposed_event_likelihood
                    - self.L2[eligible_events]
                    - detector.Entries
                    + cp.log(self.vertex_counts[eligible_events], **FLOAT32_KWARGS)
                ) > self.log_acceptance_uniforms[self.step, eligible_events, 0]
                above_energy_floor = detector.Elimit(
                    proposed_vertices, self.vertex_counts[eligible_events] - 1
                )
                accepted_subset = likelihood_accepted & above_energy_floor
                eligible_events[eligible_events] = accepted_subset

                self.vertices[:, eligible_events] = proposed_vertices[
                    :, accepted_subset
                ]
                self.L2[eligible_events] = proposed_event_likelihood[accepted_subset]
                self.L1[eligible_events] = proposed_channel_likelihood[accepted_subset]
                self.epsilon[eligible_events] = proposed_epsilon[accepted_subset]
                self.Q[:, eligible_events] = proposed_assignment_probabilities[
                    :, accepted_subset
                ]
                self.abn[eligible_events] = proposed_abn[accepted_subset]
                self.wavel_ch[eligible_events] = proposed_wavel_ch[accepted_subset]
                self.vertex_counts[eligible_events] -= 1
            accepted_event_mask = eligible_events

        # A rejected birth/death is the self-loop of that MH sub-step.  Follow
        # every dimension sub-step (accepted, rejected or not proposed) with
        # independent position and time MH sub-steps.  This is a composition of
        # invariant kernels, unlike selecting the second move conditional on a
        # rejection.
        all_events = cp.ones(len(self.vertex_counts), dtype=bool)
        position_vertex_indices = (
            self.vertex_counts
            * self.vertex_selection_uniforms[self.step, :, 1]
        ).astype(cp.int16)
        position_accepted = self._move_vertices(
            position_vertex_indices, all_events, "P"
        )
        time_vertex_indices = (
            self.vertex_counts
            * self.vertex_selection_uniforms[self.step, :, 2]
        ).astype(cp.int16)
        self._move_vertices(time_vertex_indices, all_events, "T")
        if dimension_move == 0:
            # Preserve the historical meaning of acceptv for move == 0.
            accepted_event_mask = position_accepted

        self.step += 1
        return accepted_event_mask, dimension_move
