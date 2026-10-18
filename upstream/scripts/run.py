#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMING_COLUMNS = (
    "stage",
    "start_utc",
    "end_utc",
    "real_s",
    "exit_code",
    "output_bytes",
    "target",
)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_pipeline(input_id):
    python = sys.executable
    raw = f"data/raw/{input_id}.h5"
    ser = "data/ser_truth_J25.7.4+.h5"
    probe = "data/sim.h5"
    pmt = "data/PMT_J26.4.0.h5"
    sparsify = f"output/sparsify/{input_id}.h5"
    pe = f"output/PE/{input_id}.pq"
    reconstruction = f"output/tvE/{input_id}.h5"
    required_files = [raw, ser, probe, pmt,
                      "data/junoid_J25.3.0.pq", "data/J25.3.0.PMTPos_CD_LPMT.csv"]
    stages = [
        ("01_fsmp_prior", sparsify,
         [python, "src/fsmp_prior.py", raw, "-b", raw, "--ref", ser, "-o", sparsify]),
        ("02_fsmp", pe,
         [python, "src/fsmp.py", sparsify, "--ref", ser, "--full", pe,
          "--full-step", "50", "--trials", "4000"]),
        ("03_reconstruct", reconstruction,
         [python, "src/main.py", "-f", pe, "--probe", probe, "--PMT", pmt,
          "--EMstep", "20", "-n", "10", "-m", "2000", "--bappen", "2",
          "-o", reconstruction]),
    ]
    return required_files, stages, reconstruction


def prepare_environment():
    for directory in ("sparsify", "PE", "tvE", "logs"):
        (PROJECT_ROOT / "output" / directory).mkdir(parents=True, exist_ok=True)

    cache_root = PROJECT_ROOT / "output" / "tmp"
    runtime_root = cache_root / "runtime"
    for directory in ("runtime", "cupy", "cuda", "xdg", "pycache"):
        (cache_root / directory).mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": str(runtime_root),
            "TMP": str(runtime_root),
            "TEMP": str(runtime_root),
            "SUPERJUNO_CACHE_DIR": str(cache_root),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "CUPY_CACHE_DIR": str(cache_root / "cupy"),
            "CUDA_CACHE_PATH": str(cache_root / "cuda"),
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
        }
    )
    return environment


def output_size(relative_path):
    path = PROJECT_ROOT / relative_path
    return path.stat().st_size if path.is_file() else 0


def run_stage(stage, environment, timing_writer):
    name, target, command = stage
    stdout_path = PROJECT_ROOT / f"{target}.log"
    start_utc = utc_now()
    started = time.perf_counter()

    print(f"==> [{name}] {target}", flush=True)
    with stdout_path.open("wb") as stdout:
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stdout,
                check=False,
            )
            returncode = result.returncode
            if returncode < 0:
                returncode = 128 - returncode
        except OSError as error:
            print(error, file=sys.stderr)
            returncode = 127

    timing_writer.writerow(
        (
            name,
            start_utc,
            utc_now(),
            f"{time.perf_counter() - started:.6f}",
            returncode,
            output_size(target),
            target,
        )
    )
    if returncode:
        print(f"Stage failed: {name} (exit {returncode})", file=sys.stderr)
    return returncode


def run_comparisons(input_id, environment):
    status = 0
    for kind, extension in (("PE", "pq"), ("tvE", "h5")):
        reference = f"data/ana_{kind}/{input_id}.{extension}"
        candidate = f"output/{kind}/{input_id}.{extension}"
        print(f"==> [compare_{kind}] {candidate}", flush=True)
        print(f"\n[{kind}]\nReference: {reference}\nCandidate: {candidate}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, f"test/compare_{kind}.py", reference, candidate],
                cwd=PROJECT_ROOT,
                env=environment,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = result.returncode
            if returncode < 0:
                returncode = 128 - returncode
        except OSError as error:
            print(f"Comparison failed: {error}", flush=True)
            returncode = 127
        except KeyboardInterrupt:
            print("Comparison interrupted.", flush=True)
            print("Comparison interrupted", file=sys.stderr)
            return 130
        print(f"Comparison exit code: {returncode}", flush=True)
        if returncode:
            print(f"Comparison failed: {kind} (exit {returncode})", file=sys.stderr)
        status = max(status, returncode)
    return status


def run_pipeline(input_id, compare=False):
    required_files, stages, final_target = build_pipeline(input_id)
    environment = prepare_environment()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    timing_path = PROJECT_ROOT / "output/logs" / f"{input_id}-{run_id}.time.csv"
    with timing_path.open("w", encoding="utf-8", newline="") as timing_file:
        timing_writer = csv.writer(timing_file, lineterminator="\n")
        timing_writer.writerow(TIMING_COLUMNS)
        start_utc = utc_now()
        started = time.perf_counter()
        status = 1

        try:
            for relative_path in required_files:
                path = PROJECT_ROOT / relative_path
                if not path.is_file() or not os.access(path, os.R_OK):
                    message = f"Missing input: {relative_path}"
                    print(message, file=sys.stderr)
                    print(message, flush=True)
                    break
            else:
                for stage in stages:
                    status = run_stage(stage, environment, timing_writer)
                    if status:
                        print(f"Stage failed: {stage[0]} (exit {status})", flush=True)
                        break
                if status == 0:
                    print(f"Completed: {final_target}")
        except OSError as error:
            status = 1
            print(error, file=sys.stderr)
            print(f"Pipeline failed: {error}", flush=True)
        except KeyboardInterrupt:
            status = 130
            print("Interrupted", file=sys.stderr)
            print("Pipeline interrupted.", flush=True)
        finally:
            total_s = f"{time.perf_counter() - started:.6f}"
            timing_writer.writerow(
                (
                    "TOTAL",
                    start_utc,
                    utc_now(),
                    total_s,
                    status,
                    output_size(final_target),
                    final_target,
                )
            )
            timing_file.flush()
            print(f"TOTAL: {total_s} s", flush=True)
            print(f"Pipeline exit code: {status}", flush=True)
            print(f"Timing: {timing_path.relative_to(PROJECT_ROOT)}", flush=True)

        if status:
            print("Comparison skipped: pipeline did not complete.", flush=True)
        elif compare:
            status = run_comparisons(input_id, environment)
        else:
            print("Comparison skipped: --compare was not specified.", flush=True)
        print(f"\nFinal exit code: {status}", flush=True)
        return status


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the three-stage PMT reconstruction pipeline."
    )
    parser.add_argument("input_id", metavar="RAW_FILE_STEM")
    parser.add_argument(
        "--compare", action="store_true",
        help="compare PE and tvE outputs against references after a successful run",
    )
    args = parser.parse_args()
    if (
        not args.input_id
        or Path(args.input_id).name != args.input_id
        or args.input_id in {".", ".."}
    ):
        parser.error("RAW_FILE_STEM must be a file stem, not a path")
    return args


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_pipeline(args.input_id, compare=args.compare))
