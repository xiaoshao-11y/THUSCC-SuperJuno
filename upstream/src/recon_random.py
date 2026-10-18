"""Backend-independent random inputs for event reconstruction.

Random values are attached to logical event/channel coordinates rather than
to a CuPy backend, an event block, or the order in which proposals happen to
execute.  All stateful streams use NumPy PCG64, matching the FSMP random-stream
contract.  The much larger per-channel acceptance tensor is generated
statelessly from its full logical coordinate to avoid materialising
``mc_steps * events * channels`` values.
"""

import numpy as np


DIMENSION_STREAM = 0
PE_SAMPLE_STREAM = 1
CHANNEL_ACCEPT_STREAM = 2
GIBBS_ACCEPT_STREAM = 3
PROPOSAL_NORMAL_STREAM = 4
VERTEX_SELECTION_STREAM = 5
BIRTH_STREAM = 6

_MASK64 = (1 << 64) - 1
_MIX_EVENT = np.uint64(0xD2B74407B1CE6E93)
_MIX_CHANNEL = np.uint64(0xCA5A826395121157)
_MIX_STEP = np.uint64(0x9E3779B97F4A7C15)
_MIX_SEED = np.uint64(0x94D049BB133111EB)
_MIX_PURPOSE = np.uint64(0xBF58476D1CE4E5B9)


def _uint64_words(value):
    """Split an arbitrary Python integer into two SeedSequence words."""
    unsigned = int(value) & _MASK64
    return [unsigned & 0xFFFFFFFF, unsigned >> 32]


def _generator(seed, stream, *coordinates):
    """Return a PCG64 stream identified only by logical coordinates."""
    entropy = []
    for value in (seed, stream, *coordinates):
        entropy.extend(_uint64_words(value))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _event_array(seed, stream, event_ids, mc_steps, tail_shape, method):
    """Generate one independent PCG64 tensor per EventID."""
    result = np.empty((mc_steps, len(event_ids), *tail_shape), dtype=np.float32)
    for event_index, event_id in enumerate(event_ids):
        rng = _generator(seed, stream, int(event_id))
        result[:, event_index] = method(rng, (mc_steps, *tail_shape))
    return result


def _splitmix64(values):
    """Mix uint64 logical counters into deterministic pseudorandom words."""
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values ^ (values >> np.uint64(31))


def _coordinate_uniform(seed, purpose, event_ids, channel_ids, step):
    """Return float32 uniforms keyed by seed, EventID, channel, and step."""
    event_words = np.asarray(event_ids, dtype=np.int64).view(np.uint64)[:, None]
    channel_words = np.asarray(channel_ids, dtype=np.int64).view(np.uint64)
    step_word = np.asarray(step, dtype=np.uint64)
    seed_word = np.asarray(int(seed) & _MASK64, dtype=np.uint64)
    purpose_word = np.asarray(purpose, dtype=np.uint64)
    counters = (
        event_words * _MIX_EVENT
        ^ channel_words * _MIX_CHANNEL
        ^ step_word * _MIX_STEP
        ^ seed_word * _MIX_SEED
        ^ purpose_word * _MIX_PURPOSE
    )
    words = _splitmix64(counters)
    # Use 23 random bits and bin midpoints.  The result is strictly inside
    # (0, 1), exactly representable as float32, and therefore safe for log().
    mantissas = (words >> np.uint64(41)).astype(np.uint32)
    return ((mantissas.astype(np.float32) + np.float32(0.5)) * np.float32(2**-23))


def _birth_vertices(rng, shape):
    """Generate already-transformed float32 birth proposals on the CPU."""
    mc_steps = shape[0]
    uniforms = rng.random((mc_steps, 4), dtype=np.float32)
    vertices = np.zeros((mc_steps, 5), dtype=np.float32)
    cos_theta = np.float32(2) * uniforms[:, 0] - np.float32(1)
    sin_theta = np.sqrt(np.float32(1) - cos_theta**2).astype(np.float32)
    phi = np.float32(2 * np.pi) * uniforms[:, 1]
    radius = np.cbrt(uniforms[:, 2]).astype(np.float32)
    vertices[:, 0] = radius * sin_theta * np.cos(phi).astype(np.float32)
    vertices[:, 1] = radius * sin_theta * np.sin(phi).astype(np.float32)
    vertices[:, 2] = radius * cos_theta
    vertices[:, 4] = np.float32(-200) + np.float32(1150) * uniforms[:, 3]
    return vertices


class ReconstructionRandom:
    """Random tensors for one event block, keyed independently per EventID."""

    def __init__(
        self,
        seed,
        event_ids,
        channel_ids,
        real_channel_counts,
        mc_steps,
        sample_count,
    ):
        self.seed = int(seed)
        self.event_ids = np.asarray(event_ids, dtype=np.int64)
        self.channel_ids = np.asarray(channel_ids, dtype=np.int64)
        self.real_channel_counts = np.asarray(real_channel_counts, dtype=np.int32)
        self.mc_steps = int(mc_steps)

        if self.channel_ids.shape[0] != len(self.event_ids):
            raise ValueError("channel_ids and event_ids have incompatible shapes")
        if self.real_channel_counts.shape != (len(self.event_ids),):
            raise ValueError("real_channel_counts and event_ids have incompatible shapes")

        self.dimension_uniforms = _generator(
            self.seed, DIMENSION_STREAM
        ).random(self.mc_steps, dtype=np.float32)

        self.proposal_sample_indices = np.empty(
            (self.mc_steps, len(self.event_ids)), dtype=np.int64
        )
        for event_index, event_id in enumerate(self.event_ids):
            self.proposal_sample_indices[:, event_index] = _generator(
                self.seed, PE_SAMPLE_STREAM, int(event_id)
            ).integers(0, sample_count, self.mc_steps, dtype=np.int64)

        acceptance_uniforms = _event_array(
            self.seed,
            GIBBS_ACCEPT_STREAM,
            self.event_ids,
            self.mc_steps,
            (4,),
            lambda rng, shape: rng.random(shape, dtype=np.float32),
        )
        self.log_acceptance_uniforms = np.log(acceptance_uniforms).astype(np.float32)
        self.proposal_normals = _event_array(
            self.seed,
            PROPOSAL_NORMAL_STREAM,
            self.event_ids,
            self.mc_steps,
            (5,),
            lambda rng, shape: rng.standard_normal(shape, dtype=np.float32),
        )
        self.vertex_selection_uniforms = _event_array(
            self.seed,
            VERTEX_SELECTION_STREAM,
            self.event_ids,
            self.mc_steps,
            (3,),
            lambda rng, shape: rng.random(shape, dtype=np.float32),
        )
        self.birth_vertices = _event_array(
            self.seed,
            BIRTH_STREAM,
            self.event_ids,
            self.mc_steps,
            (5,),
            _birth_vertices,
        )

    def channel_log_uniforms(self, step):
        """Return channel thresholds for one step in the block's padded layout."""
        if not 0 <= step < self.mc_steps:
            raise IndexError(f"random step {step} is outside [0, {self.mc_steps})")
        uniforms = _coordinate_uniform(
            self.seed,
            CHANNEL_ACCEPT_STREAM,
            self.event_ids,
            self.channel_ids,
            step,
        )
        result = np.log(uniforms).astype(np.float32)
        padding = (
            np.arange(self.channel_ids.shape[1])[None, :]
            >= self.real_channel_counts[:, None]
        )
        result[padding] = 0
        return result
