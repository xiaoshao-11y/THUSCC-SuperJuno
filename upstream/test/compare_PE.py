#!/usr/bin/env python3
"""Check saved PE states."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from comparison import differences, run

KEYS = ["eid", "ch", "step"]
COLUMNS = KEYS + ["offset", "PEt", "e", "q", "nu_lc"]


def load_states(path):
    # Arrow's parallel reader can abort at interpreter shutdown.
    with pq.ParquetFile(path) as parquet:
        frame = parquet.read(use_threads=False).to_pandas(use_threads=False)
    missing = sorted(set(COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: no saved PE states")
    for field in KEYS + ["offset"]:
        if not pd.api.types.is_integer_dtype(frame[field]) or frame[field].isna().any():
            raise ValueError(f"{path}: {field} must contain integers")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate (eid,ch,step) keys")

    states = {}
    for eid, ch, step, offset, pet, e, q, nu_lc in frame[COLUMNS].itertuples(index=False, name=None):
        location = f"{path}: eid={eid},ch={ch},step={step}"
        pet, e, q = np.asarray(pet, dtype=float), np.asarray(e), np.asarray(q, dtype=float)
        if any(values.ndim != 1 for values in (pet, e, q)):
            raise ValueError(f"{location}: PEt, e and q must be arrays")
        if len(pet) != len(e) or len(pet) != len(q):
            raise ValueError(f"{location}: PEt, e and q lengths differ")
        if e.size and e.dtype.kind not in "iu":
            raise ValueError(f"{location}: e must contain integers")
        states[eid, ch, step] = (offset, pet, e, q, float(nu_lc))
    return states


def compare(reference_path, candidate_path):
    reference, candidate = load_states(reference_path), load_states(candidate_path)
    for key in sorted(reference.keys() | candidate.keys()):
        location = f"eid={key[0]},ch={key[1]},step={key[2]}"
        if key not in candidate:
            yield f"Missing candidate record; {location}"
            continue
        if key not in reference:
            yield f"Unexpected candidate record; {location}"
            continue
        ref_offset, ref_pet, ref_e, ref_q, ref_nu = reference[key]
        cand_offset, cand_pet, cand_e, cand_q, cand_nu = candidate[key]
        yield from differences(ref_offset, cand_offset, "offset", location, exact=True)
        yield from differences(len(ref_pet), len(cand_pet), "NPE", location, exact=True)
        if len(ref_pet) == len(cand_pet):
            yield from differences(ref_e, cand_e, "e", location, exact=True)
            yield from differences(ref_pet, cand_pet, "PEt", location)
            # Without --mmse, the writer fills q with null placeholders.
            yield from differences(ref_q, cand_q, "q", location, allow_nan=True)
        yield from differences(ref_nu, cand_nu, "nu_lc", location)


if __name__ == "__main__":
    run(compare, __doc__, "PE")
