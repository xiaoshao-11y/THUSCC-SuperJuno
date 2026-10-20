"""Reconstruct event vertex positions, energies, and times."""

import argparse
import os
import subprocess
import sys

import cupy as cp
import h5py

import recon
from detector import Probe, chnums
from fsmp_reader import FullReader


def _shard_output_path(path, index):
    """Return the per-shard suffix of an output path."""
    return f"{path}.shard{index}"


def _merge_recon_shards(final_path, shard_paths):
    """Concatenate shard /Recon files along the event axis into one file.

    The tvE comparator aligns by (EventID, step) keys, so shard order only
    needs to be deterministic (shards are appended in index order).
    """
    shapes = []
    with h5py.File(shard_paths[0], "r") as first:
        attrs = dict(first.attrs)
    for path in shard_paths:
        with h5py.File(path, "r") as shard:
            shapes.append(shard["Recon"].shape)
    vertex_slots, _, steps = shapes[0]
    total_events = sum(shape[1] for shape in shapes)
    with h5py.File(final_path, "w") as output_file:
        for key, value in attrs.items():
            output_file.attrs[key] = value
        merged = output_file.create_dataset(
            "Recon",
            (vertex_slots, total_events, steps),
            dtype=recon.RECON_DTYPE,
            compression="gzip",
            shuffle=True,
        )
        offset = 0
        for path, shape in zip(shard_paths, shapes):
            with h5py.File(path, "r") as shard:
                merged[:, offset : offset + shape[1], :] = shard["Recon"][:]
            offset += shape[1]
    for path in shard_paths:
        os.remove(path)


def _run_sharded(argv, shard_count):
    """Spawn one reconstruction worker per shard (one GPU each) and merge.

    Workers are re-invocations of this script with SJ_SHARD_INDEX set; each
    reconstructs the event blocks with ``number % shard_count == index``
    and writes a suffixed output. Blocks are independent and the
    reconstruction random streams are keyed by EventID, so the merged
    result matches the single-process run.
    """
    gpu_ids = os.environ.get("SJ_SHARD_GPUS", "")
    gpu_ids = gpu_ids.split(",") if gpu_ids else [str(i) for i in range(shard_count)]
    if len(gpu_ids) < shard_count:
        raise SystemExit(
            f"SJ_SHARD_GPUS lists {len(gpu_ids)} devices < SJ_SHARDS={shard_count}"
        )
    workers = []
    for index in range(shard_count):
        env = os.environ.copy()
        env["SJ_SHARD_INDEX"] = str(index)
        env["SJ_SHARD_SPAWNED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index].strip()
        print(
            f"[SJ_SHARDS] recon worker {index} -> CUDA_VISIBLE_DEVICES="
            f"{env['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        workers.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), *argv], env=env
            )
        )
    failed = [i for i, worker in enumerate(workers) if worker.wait() != 0]
    if failed:
        raise SystemExit(f"[SJ_SHARDS] recon shard workers failed: {failed}")
    args = build_parser().parse_args(argv)
    shard_paths = [
        _shard_output_path(args.output_file, index) for index in range(shard_count)
    ]
    missing = [p for p in shard_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"[SJ_SHARDS] missing recon shard outputs: {missing}")
    _merge_recon_shards(args.output_file, shard_paths)
    print(
        f"[SJ_SHARDS] merged {shard_count} recon shards -> {args.output_file}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct event vertices from FSMP PE-time samples."
    )
    parser.add_argument(
        "-f", "--file", dest="input_file", type=str, help="FSMP PE-time sample file"
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        metavar="output[*.h5]",
        help="HDF5 reconstruction output file",
    )
    parser.add_argument(
        "--probe",
        dest="probe_file",
        type=str,
        default="probe_tts.h5",
        help="detector response file",
    )
    parser.add_argument(
        "--PMT",
        dest="pmt_file",
        type=str,
        metavar="PMT[*.h5]",
        help="PMT calibration file",
    )
    parser.add_argument(
        "-n",
        "--num",
        dest="block_count",
        type=int,
        default=10,
        help="number of event blocks",
    )
    parser.add_argument(
        "-m",
        "--MCstep",
        dest="mc_steps",
        type=int,
        default=10000,
        help="number of MCMC steps",
    )
    parser.add_argument(
        "-e",
        "--EMstep",
        dest="em_steps",
        type=int,
        default=30,
        help="number of EM steps for energy estimation",
    )
    parser.add_argument(
        "--bappen",
        dest="max_vertices",
        type=int,
        default=2,
        help="maximum number of vertices",
    )
    parser.add_argument(
        "--lhigh",
        dest="lambda_high",
        type=float,
        default=50,
        help="lambda_high for the glass-emission likelihood",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="backend-independent reconstruction random seed",
    )
    return parser


def write_output_attributes(handle, args) -> None:
    """Record reconstruction input paths and run parameters."""
    handle.attrs["input_file"] = args.input_file
    handle.attrs["probe_file"] = args.probe_file
    handle.attrs["pmt_file"] = args.pmt_file
    handle.attrs["mc_steps"] = args.mc_steps
    handle.attrs["em_steps"] = args.em_steps
    handle.attrs["max_vertices"] = args.max_vertices
    handle.attrs["random_seed"] = args.seed


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # [SUPERJUNO-OPT] Stage-3 block sharding is deliberately SEPARATE from
    # stage 2's SJ_SHARDS. Empirically (2026-10-19, scored case 0): sharding
    # the reconstruction changes cuBLAS float32 kernel selection (pointer
    # alignment / allocation history), giving ulp-level differences in the
    # initial vertices that 2000-step chains amplify beyond the 1e-6
    # comparator tolerance (2/100 events crossed -> tvE FAIL), while the
    # sharded execution itself is bit-stable across nodes/GPUs. Stage-2
    # sharding is bit-exact at full scale (PE outputs bitwise equal) and
    # stays on SJ_SHARDS; stage 3 defaults to the upstream single process.
    shard_count = max(1, int(os.environ.get("SJ_RECON_SHARDS", "1")))
    shard_index = int(os.environ.get("SJ_SHARD_INDEX", "0"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit("SJ_SHARD_INDEX must be within [0, SJ_SHARDS)")
    if shard_count > 1 and "SJ_SHARD_SPAWNED" not in os.environ:
        raise SystemExit(_run_sharded(sys.argv[1:], shard_count))
    output_path = (
        _shard_output_path(args.output_file, shard_index)
        if shard_count > 1
        else args.output_file
    )

    # Load PMT calibration and detector response.
    with h5py.File(args.pmt_file, "r") as pmt_file:
        pmt_data = pmt_file["PMT"][:]
        dark_rate = cp.asarray(pmt_data["darkrate"], dtype=cp.float32) / 1e9
        # Calibration convention: observed time = physical time + PMT offset.
        pmt_time_correction = -cp.asarray(pmt_data["offset"], dtype=cp.float32)
        pde = cp.asarray(pmt_data["PDE"] / 100, dtype=cp.float32)
    print("Finished Reading PMT")

    response = Probe(args.probe_file, pde)
    print("Finished Loading Probe")

    # Initialize the likelihood evaluator.
    likelihood = recon.Likelihood(
        response, dark_rate, args.em_steps, args.lambda_high
    )

    # Load FSMP event samples.
    event_reader = FullReader(args.input_file, chnums, args.block_count)
    print("Finished Loading Observation")

    # Create the output file.
    dataset_options = {"compression": "gzip", "shuffle": True}
    if shard_count > 1:
        # Sharded workers trim their dataset to the events they actually
        # reconstructed, which requires an extensible event axis.
        dataset_options["maxshape"] = (args.max_vertices, None, args.mc_steps)
    event_count = len(event_reader)
    with h5py.File(output_path, "w") as output_file:
        write_output_attributes(output_file, args)
        recon_dataset = output_file.create_dataset(
            "Recon",
            (args.max_vertices, event_count, args.mc_steps),
            dtype=recon.RECON_DTYPE,
            **dataset_options,
        )

        # Run reconstruction.
        recon.Reconstruction(
            event_reader,
            response,
            likelihood,
            pmt_time_correction,
            args.mc_steps,
            args.max_vertices,
            recon_dataset,
            args.seed,
            shard_count=shard_count,
            shard_index=shard_index,
        )


if __name__ == "__main__":
    main()
