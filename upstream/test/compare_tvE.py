#!/usr/bin/env python3
"""Compare saved reconstruction states."""

import h5py
import numpy as np

from comparison import differences, run, tolerance

VERTEX_FIELDS = ("x", "y", "z", "E", "t")
EXACT_FIELDS = ("move", "count", "NPE", "acceptv")
EVENT_FLOATS = ("Likelihood", "epsilon")
EVENT_FIELDS = ("EventID", "step", *EXACT_FIELDS, *EVENT_FLOATS)


def load_recon(filename):
    with h5py.File(filename, "r") as handle:
        if "Recon" not in handle or not isinstance(handle["Recon"], h5py.Dataset):
            raise ValueError(f"{filename}: missing /Recon dataset")
        recon = handle["Recon"][:]
    if recon.ndim != 3 or any(size == 0 for size in recon.shape):
        raise ValueError(f"{filename}: /Recon must be a non-empty 3D dataset")
    missing = set((*VERTEX_FIELDS, *EVENT_FIELDS)) - set(recon.dtype.names or ())
    if missing:
        raise ValueError(f"{filename}: missing /Recon fields {sorted(missing)}")
    for field in ("EventID", "step", "move", "count"):
        if recon.dtype[field].kind not in "iu":
            raise ValueError(f"{filename}: {field} must have integer dtype")
    if recon.dtype["acceptv"].kind != "b":
        raise ValueError(f"{filename}: acceptv must have boolean dtype")
    for field in (*VERTEX_FIELDS, "NPE", *EVENT_FLOATS):
        if recon.dtype[field].kind not in "fiu":
            raise ValueError(f"{filename}: {field} must have numeric dtype")
    for field in ("EventID", "step", "move", "count", "acceptv"):
        values = recon[field]
        first = np.broadcast_to(values[0:1], values.shape)
        if not np.array_equal(values, first, equal_nan=True):
            raise ValueError(f"{filename}: {field} differs across vertex slots")
    event_ids = recon["EventID"][0, :, 0]
    if not np.all(recon["EventID"] == event_ids[None, :, None]):
        raise ValueError(f"{filename}: EventID changes within an event")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError(f"{filename}: duplicate EventID values")
    if any(len(np.unique(steps)) != recon.shape[2] for steps in recon["step"][0]):
        raise ValueError(f"{filename}: duplicate steps within an event")
    if np.any((recon["count"] < 1) | (recon["count"] > recon.shape[0])):
        raise ValueError(f"{filename}: count outside the vertex-slot range")
    return recon, {int(event_id): index for index, event_id in enumerate(event_ids)}


def compare(reference_path, candidate_path):
    reference, reference_events = load_recon(reference_path)
    candidate, candidate_events = load_recon(candidate_path)
    if reference.shape[0] != candidate.shape[0]:
        yield (
            f"Vertex-slot count differs: reference={reference.shape[0]}, "
            f"candidate={candidate.shape[0]}"
        )
        return
    for event_id in sorted(reference_events.keys() | candidate_events.keys()):
        if event_id not in candidate_events:
            yield f"EventID={event_id}: missing event in candidate"
            continue
        if event_id not in reference_events:
            yield f"EventID={event_id}: extra event in candidate"
            continue
        ref_event = reference[:, reference_events[event_id]]
        cand_event = candidate[:, candidate_events[event_id]]
        ref_steps, cand_steps = ref_event["step"][0], cand_event["step"][0]
        for step in np.setdiff1d(ref_steps, cand_steps):
            yield f"EventID={event_id}, step={step}: missing step in candidate"
        for step in np.setdiff1d(cand_steps, ref_steps):
            yield f"EventID={event_id}, step={step}: extra step in candidate"
        steps, ref_indices, cand_indices = np.intersect1d(
            ref_steps, cand_steps, return_indices=True
        )
        ref, cand = ref_event[:, ref_indices].T, cand_event[:, cand_indices].T
        location = lambda i: f"EventID={event_id}, step={steps[i]}"
        for field in ("move", "count", "acceptv"):
            yield from differences(
                ref[field][:, 0], cand[field][:, 0], field, location, exact=True
            )
        # Check every copy; report the largest excess over tolerance per step.
        rows = np.arange(len(steps))
        for field in ("NPE", *EVENT_FLOATS):
            finite = np.isfinite(ref[field]) & np.isfinite(cand[field])
            if field == "NPE":
                errors = (~finite | (ref[field] != cand[field])).astype(float)
            else:
                with np.errstate(invalid="ignore", over="ignore"):
                    errors = np.abs(cand[field].astype(float) - ref[field].astype(float))
                    errors -= tolerance(ref[field])
            errors[~finite] = np.inf
            vertices = np.argmax(errors, axis=1)
            location = lambda i: (
                f"EventID={event_id}, step={steps[i]}, vertex={vertices[i]}"
            )
            yield from differences(
                ref[field][rows, vertices], cand[field][rows, vertices],
                field, location, exact=field == "NPE"
            )
        # Only active vertices with matching counts can be paired by saved slot.
        active = np.arange(reference.shape[0])[None, :] < ref["count"][:, :1]
        active &= ref["count"][:, :1] == cand["count"][:, :1]
        step_indices, vertex_indices = np.nonzero(active)
        location = lambda i: (
            f"EventID={event_id}, step={steps[step_indices[i]]}, "
            f"vertex={vertex_indices[i]}"
        )
        for field in VERTEX_FIELDS:
            yield from differences(ref[field][active], cand[field][active], field, location)


if __name__ == "__main__":
    run(compare, __doc__, "tvE")
