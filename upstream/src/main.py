"""Reconstruct event vertex positions, energies, and times."""

import argparse

import cupy as cp
import h5py

import recon
from detector import Probe, chnums
from fsmp_reader import FullReader


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
    event_count = len(event_reader)
    with h5py.File(args.output_file, "w") as output_file:
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
        )


if __name__ == "__main__":
    main()
