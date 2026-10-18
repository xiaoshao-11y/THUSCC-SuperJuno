"""Run waveform-level FSMP sampling."""

import argparse

import cupy as cp
import h5py
import hdf5plugin  # noqa: F401  Register HDF5 filter 32015.
import numpy as np

from analysis.fsmp import batch, multi_data, prepare_flip, prepare_random
from analysis.fsmp.writer import BasicWriter, FullWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="opt", type=str, default=None, help="output file")
    parser.add_argument("ipt", type=str, help="input file")
    parser.add_argument("--ref", type=str, help="SER file")
    parser.add_argument(
        "--PMT", dest="PMT", type=str, help="PMT info file for sig2w input"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed for random number"
    )
    parser.add_argument(
        "--full", type=str, default=None, help="The full chain output"
    )
    parser.add_argument(
        "--full-step",
        type=int,
        default=None,
        help="The step to record full chain, None for all",
    )
    parser.add_argument(
        "--trials", type=int, default=5000, help="Trial steps of MCMC"
    )
    parser.add_argument("--mmse", action="store_true", help="Output MMSE")
    parser.add_argument(
        "--recal", type=int, default=None, help="Recalculate U and F every N steps"
    )
    parser.add_argument(
        "--ch", type=int, nargs="+", default=None, help="Channels to process"
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    np.seterr(all="raise")

    with cp.cuda.Device(0) as device:
        vram = device.mem_info[1] / (2**10) ** 3 / 5  # in GiB
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    trials = args.trials
    if args.recal is None:
        recalculation_step = trials + 1
    else:
        recalculation_step = args.recal

    with h5py.File(args.ipt, "r", libver="latest", swmr=True) as input_file:
        index_full = input_file["index"][:]

        reorder_index = np.argsort(index_full["l_t"])
        # Avoid saturated waveforms if the historical guard is re-enabled.
        # reorder_index = reorder_index[
        #     (index_full["s0"] <= np.fmin(300, index_full["l_wave"]))[
        #         reorder_index
        #     ]
        # ]
        if args.ch is not None:
            reorder_index = reorder_index[
                np.isin(index_full["ch"][reorder_index], args.ch)
            ]

        index_full = index_full[reorder_index]
        mu0_full = input_file["mu0"][:][reorder_index]
        s_full = input_file["s"][:][reorder_index]
        e_full = input_file["e"][:][reorder_index]
        tq_full = input_file["tq"][:][reorder_index]
        z_full = input_file["z"][:][reorder_index]

    index_dtypes = {
        "eid_dtype": index_full["eid"].dtype,
        "ch_dtype": index_full["ch"].dtype,
        "offset_dtype": index_full["offset"].dtype,
    }

    full_writer = FullWriter(args.full, **index_dtypes) if args.full else None

    with h5py.File(args.ref, "r", libver="latest", swmr=True) as reference_file:
        spe_data_full = reference_file["ser"][:]
        gains_full = reference_file["gain"][:]

    # Replace sig2w with the value from the PMT info file if provided.
    if args.PMT:
        with h5py.File(args.PMT, "r") as pmt_file:
            assert (pmt_file["PMT"]["ch"][:] == spe_data_full["ch"]).all()
            sig2w_pmt = pmt_file["PMT"]["sig2w"][:]
        spe_data_full["sig2w"] = sig2w_pmt

    waveform_count = len(index_full)
    flip_schedule = prepare_flip(args.seed, trials)
    basic_writer = BasicWriter(args.opt, **index_dtypes) if args.opt else None

    print(f"Analyze waveforms, {waveform_count} in total.")

    for part, (start, part_size, max_t, max_wave, max_npe) in enumerate(
        multi_data(index_full, vram)
    ):
        print(
            f"Part {part} | {start}:{start + part_size} with lengths of PE t "
            f"{max_t}, waveform {max_wave} and NPE {max_npe}"
        )
        part_slice = np.s_[start : start + part_size]
        mu0 = mu0_full[part_slice]
        part_index = index_full[part_slice]
        initial_pe = s_full[part_slice, :max_npe]
        gain_components = e_full[part_slice, :max_npe]
        time_charge = tq_full[part_slice, :max_t]
        waveforms = z_full[part_slice, :max_wave]
        spe_data = spe_data_full[
            np.searchsorted(spe_data_full["ch"], part_index["ch"])
        ]
        gain_data = gains_full[
            np.searchsorted(gains_full["ch"], part_index["ch"])
        ]
        requester = part_index[["eid", "ch"]]
        istar, erandom, wanders, accepts = prepare_random(
            args.seed, trials, requester
        )

        print("Starting batch loop...")
        (
            nu_lc0,
            ind_history,
            step_history,
            s0_history,
            delta_nu_history,
            nu_lc_history,
            op_history,
            creations,
            creations_e,
        ) = batch(
            trials,
            recalculation_step,
            waveforms,
            part_index,
            spe_data,
            time_charge,
            initial_pe,
            gain_components,
            mu0,
            gain_data,
            istar,
            erandom,
            flip_schedule,
            wanders,
            accepts,
            full_writer,
            args.full_step,
            args.mmse,
        )
        if basic_writer:
            print("Summarising...")
            basic_writer.append_async(
                trials,
                part_index["eid"],
                part_index["ch"],
                part_index["offset"],
                part_index["s0"],
                nu_lc0,
                ind_history,
                step_history,
                s0_history,
                op_history,
                creations,
                creations_e,
                delta_nu_history,
                nu_lc_history,
            )
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if basic_writer:
        basic_writer.close()
    if full_writer:
        full_writer.close()


if __name__ == "__main__":
    main()
