"""Shared tolerance checks and command-line output for the two comparators."""

import argparse
from itertools import islice
from pathlib import Path

import numpy as np

ATOL = 1e-6
RTOL = 1e-6
MAX_ERRORS = 10


def tolerance(reference):
    return ATOL + RTOL * np.abs(np.asarray(reference, dtype=float))


def differences(reference, candidate, field, location, *, exact=False, allow_nan=False):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        yield (f"{field} shape mismatch; {location}; "
               f"ref={reference.shape}, cand={candidate.shape}")
        return
    if not reference.size:
        return

    finite = np.isfinite(reference) & np.isfinite(candidate)
    if exact:
        matches = finite & (reference == candidate)
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
            allowed = tolerance(reference)
        matches = finite & (absolute <= allowed)
        if allow_nan:
            matches |= np.isnan(reference) & np.isnan(candidate)

    for i in np.flatnonzero(~matches):
        if callable(location):
            position = location(i)
        else:
            position = location + (f",i={i}" if reference.ndim else "")
        values = (f"ref={reference.flat[i].item()!r}, "
                  f"cand={candidate.flat[i].item()!r}")
        if not finite.flat[i]:
            yield f"{field} non-finite value; {position}; {values}"
        elif exact:
            yield f"{field} mismatch; {position}; {values}"
        else:
            yield (f"{field} out of tolerance; {position}; {values}; "
                   f"abs_diff={absolute.flat[i].item()!r}, tol={allowed.flat[i]:g}")


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"2: {message}")
        self.exit(2)


def run(compare, description, kind):
    parser = ArgumentParser(
        description=f"{description} Tolerance: {ATOL:g} + {RTOL:g} * abs(reference)."
    )
    parser.add_argument("reference", help="reference output file")
    parser.add_argument("candidate", help="candidate output file")
    args = parser.parse_args()
    try:
        messages = list(islice(compare(args.reference, args.candidate), MAX_ERRORS))
        code = 1 if messages else 0
        if not messages:
            messages = [f"PASS (atol={ATOL:g}, rtol={RTOL:g})"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        code, messages = 2, [str(error)]

    result = "".join(f"{code}: {' '.join(message.splitlines())}\n" for message in messages)
    report = Path("output") / kind / f"{Path(args.candidate).name}.compare.txt"
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(result, encoding="utf-8")
    except OSError as error:
        code = 2
        result = f"2: Could not write comparison report: {' '.join(str(error).splitlines())}\n"
    print(result, end="")
    raise SystemExit(code)
