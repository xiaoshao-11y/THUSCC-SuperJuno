"""Generate the first-round FSMP proposal from raw PMT waveforms."""

import argparse

import h5py as h5
import hdf5plugin  # noqa: F401  Register HDF5 filter 32015.
import numpy as np

from analysis.lucyddm import multi_deconv, prior
from wave_reader import WaveformReader
from wave_reader_impl import WaveformReaderImpl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="opt", type=str, help="output file")
    parser.add_argument("ipt", type=str, help="input file")
    parser.add_argument("-b", type=str, help="baseline file")
    parser.add_argument(
        "--PMT", dest="PMT", type=str, help="PMT info file for sig2w input"
    )
    parser.add_argument("--ref", type=str, help="reference file")
    parser.add_argument(
        "--seed", type=int, default=0, help="the seed for random number"
    )
    parser.add_argument(
        "--thres", type=float, default=0.1, help="q cut threshold"
    )
    parser.add_argument(
        "--cut", action="store_true", help="cut the right window of waveform"
    )
    parser.add_argument(
        "--ch", type=int, nargs="+", default=None, help="restrict to these channels"
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    fopt = args.opt
    reference = args.ref

    with h5.File(reference, "r", libver="latest", swmr=True) as ref:
        spe_data = ref["ser"][:]
        gain_data = ref["gain"][:]

    # Replace sig2w with the value from the PMT info file if provided.
    if args.PMT:
        with h5.File(args.PMT, "r") as pmt_file:
            assert (pmt_file["PMT"]["ch"][:] == spe_data["ch"]).all()
            sig2w_pmt = pmt_file["PMT"]["sig2w"][:]
        spe_data["sig2w"] = sig2w_pmt

    reader: WaveformReader = WaveformReaderImpl(args.ipt, args.b, args.cut)
    eids = reader.eid
    chs = reader.ch
    offsets = reader.offset
    ls_wave = reader.l_wave

    waves = reader.waveform
    sds = spe_data[np.searchsorted(spe_data["ch"], chs)]
    gains = gain_data[np.searchsorted(gain_data["ch"], chs)]

    selected = sds["mus"] != 0
    if args.ch is not None:
        selected &= np.isin(chs, args.ch)
    eids = eids[selected]
    chs = chs[selected]
    offsets = offsets[selected]
    ls_wave = ls_wave[selected]
    waves = waves[selected]
    sds = sds[selected]
    gains = gains[selected]

    wave_deconv = multi_deconv(waves, sds["Gm"], sds["ser"])

    # Number of waveforms and variance of white noise.
    n_wave = len(eids)
    sig2w = sds["sig2w"]
    shapes = np.zeros((n_wave, 2), np.int_)
    rng = np.random.default_rng(args.seed)

    t0_values = []
    time_values = []
    charge_values = []
    cumulative_values = []
    npe_values = []
    initial_pe_values = []
    gain_component_values = []
    mu0_values = []

    for row, (channel, wave_length, char, wave, single_pe, gain) in enumerate(
        zip(chs, ls_wave, wave_deconv, waves, sds, gains)
    ):
        ser_length = len(single_pe["ser"])
        times, charges, cumulative, initial_pe, mu0 = prior(
            char[: wave_length + ser_length - 1],
            single_pe["Gm"],
            wave[:wave_length],
            single_pe["ser"],
            thres=args.thres,
        )
        npe = len(initial_pe)
        time_length = len(times)

        t0 = 0 if time_length == 0 else max(0, times[0])
        times -= t0

        wave_length = min(wave_length - t0, time_length + ser_length)
        shapes[row, :] = [wave_length, time_length]

        if np.sum(gain["ratio"][: gain["nmean"]]) == 0:
            # This PMT is not calibrated; ignore it by setting l_t to zero.
            gain_components = []
            shapes[row, 1] = 0
        else:
            gain_components = rng.choice(
                np.arange(gain["nmean"]),
                size=npe,
                p=gain["ratio"][: gain["nmean"]],
            )

        t0_values.append(t0)
        time_values.append(times)
        charge_values.append(charges)
        cumulative_values.append(cumulative)
        npe_values.append(npe)
        gain_component_values.append(gain_components)
        initial_pe_values.append(initial_pe)
        mu0_values.append(mu0)

    mu0_values = np.array(mu0_values, dtype=np.float32)
    wave_lengths, time_lengths = shapes.T
    zero_pe = time_lengths == 0
    (signal_rows,) = np.nonzero(~zero_pe)
    signal_count = len(signal_rows)

    if n_wave == 0:
        max_wave_length, max_time_length = 0, 0
        max_npe = 0
    else:
        max_wave_length, max_time_length = np.max(shapes, axis=0)
        max_npe = np.max(npe_values)
    t0_values = np.array(t0_values)

    signal_sort_indices = np.argsort(wave_lengths[signal_rows])
    dataset_options = {
        "compression": 32015,
        "compression_opts": (1, 1),
        "shuffle": True,
    }
    with h5.File(fopt, "w") as output:
        time_charge_dtype = [
            ("t_s", np.float32),
            ("q_s", np.float32),
            ("cq", np.float32),
        ]
        time_charge = np.zeros(
            (signal_count, max_time_length), dtype=time_charge_dtype
        )
        initial_pe = -np.ones((signal_count, max_npe), np.float32)
        gain_components = -np.ones((signal_count, max_npe), np.int8)
        waveforms = np.zeros((signal_count, max_wave_length), np.float32)

        for location, row in enumerate(signal_rows):
            time_charge[location, : time_lengths[row]]["t_s"] = time_values[row]
            time_charge[location, : time_lengths[row]]["q_s"] = charge_values[row]
            time_charge[location, : time_lengths[row]]["cq"] = cumulative_values[
                row
            ]
            initial_pe[location, : npe_values[row]] = initial_pe_values[row]
            gain_components[location, : npe_values[row]] = gain_component_values[row]
            waveforms[location, : wave_lengths[row]] = waves[
                row, t0_values[row] : t0_values[row] + wave_lengths[row]
            ]

        output.create_dataset(
            "tq", data=time_charge[signal_sort_indices], **dataset_options
        )
        output.create_dataset(
            "s", data=initial_pe[signal_sort_indices], **dataset_options
        )
        output.create_dataset(
            "e", data=gain_components[signal_sort_indices], **dataset_options
        )
        output.create_dataset(
            "z", data=waveforms[signal_sort_indices], **dataset_options
        )

        index_dtype = [
            ("eid", eids.dtype),
            ("ch", chs.dtype),
            ("offset", offsets.dtype),
            ("s0", np.uint16),
            ("l_t", np.uint32),
            ("l_wave", np.uint32),
            ("sig2w", np.float32),
        ]
        index = np.zeros((n_wave,), dtype=index_dtype)
        index["eid"] = eids
        index["ch"] = chs
        index["offset"] = offsets + t0_values
        index["s0"] = npe_values
        index["sig2w"] = sig2w
        index["l_wave"] = wave_lengths
        index["l_t"] = time_lengths
        output.create_dataset(
            "mu0",
            data=mu0_values[signal_rows][signal_sort_indices],
            **dataset_options,
        )
        output.create_dataset(
            "index", data=index[signal_rows][signal_sort_indices], **dataset_options
        )
        output.create_dataset("index_0PE", data=index[zero_pe], **dataset_options)
        output.attrs["n_eid"] = len(np.unique(eids))


if __name__ == "__main__":
    main()
