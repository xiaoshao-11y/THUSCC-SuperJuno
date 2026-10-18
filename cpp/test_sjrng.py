"""Bit-exactness tests for the C++ sjrng backend against local NumPy.

Two independent references are used:

1. ``prepare_random_reference`` — a verbatim transcription of upstream
   SuperJUNO/src/analysis/fsmp/prepare.py::prepare_random, lines 187-207 at
   upstream commit 82f25ea (the numpy body only; the cupy imports of that
   module are irrelevant here). If upstream changes, update the reference
   and re-run.
2. ``upstream_prepare_random`` (TestAgainstExtractedUpstream) — the actual
   function extracted from the immutable git object
   ``82f25ea:src/analysis/fsmp/prepare.py`` via ``git show`` + ast, so the
   comparison never trusts a hand-copied body and needs no cupy import.

Run from the SuperJUNO directory:  python3 cpp/test_sjrng.py

Status: LOCAL_PASS means every generated float64 bit matches the local NumPy
2.x Generator/PCG64/SeedSequence implementation. This is a unit test of the
random streams only — it is NOT an official PE/tvE comparison and NOT a
pipeline speedup measurement. It is version-scoped: it proves equality with
the NumPy installed beside the test, not a cross-version guarantee.
"""

import ast
import os
import subprocess
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sjrng  # noqa: E402


WAVEFORM_STREAM = 1  # upstream prepare.py
UPSTREAM_COMMIT = "82f25ea67f10ede8881f4dd4656ad78cc40cfd5f"


def prepare_random_reference(seed: int, trials: int, requester):
    """Upstream prepare_random body, transcribed verbatim (numpy part)."""
    l_e = len(requester)
    istar = np.empty((trials, l_e), np.float64)
    erandom = np.empty_like(istar)
    wanders = np.empty_like(istar)
    accepts = np.empty_like(istar)

    for i, (eid, ch) in enumerate(requester):
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [seed, WAVEFORM_STREAM, int(eid), int(ch)]
                )
            )
        )
        istar[:, i] = rng.random(trials)
        erandom[:, i] = rng.random(trials)
        wanders[:, i] = rng.standard_normal(trials, np.float64)
        accepts[:, i] = np.log(rng.random(trials))

    return istar, erandom, wanders, accepts


def make_requester(eids, chs):
    dt = np.dtype([("eid", np.int64), ("ch", np.int64)])
    arr = np.zeros(len(eids), dtype=dt)
    arr["eid"] = eids
    arr["ch"] = chs
    return arr


_upstream_fn = None


def upstream_prepare_random(seed, trials, requester):
    """The real upstream function, exec'd from its immutable git object.

    Extracted exactly like the supervisor's root verify_juno_supervisor.py
    does (git show <commit>:src/analysis/fsmp/prepare.py + ast), so the body
    cannot drift from upstream and no cupy import is needed.
    """
    global _upstream_fn
    if _upstream_fn is None:
        source = subprocess.check_output(
            ["git", "-C", os.path.join(ROOT, "upstream"), "show",
             UPSTREAM_COMMIT + ":src/analysis/fsmp/prepare.py"], text=True)
        definition = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_random")
        namespace = {"np": np, "WAVEFORM_STREAM": WAVEFORM_STREAM}
        exec(compile(ast.Module(body=[definition], type_ignores=[]),
                     "immutable_upstream_prepare_random", "exec"), namespace)
        _upstream_fn = namespace["prepare_random"]
    return _upstream_fn(seed, trials, requester)


def assert_bit_equal(test, produced, reference, label):
    test.assertEqual(produced.dtype, np.float64, f"{label}: dtype")
    test.assertEqual(produced.shape, reference.shape, f"{label}: shape")
    # Bitwise equality including -0.0 vs 0.0 and NaN distinctions via tobytes.
    test.assertEqual(produced.tobytes(), reference.tobytes(),
                     f"{label}: not bit-identical to numpy reference")
    test.assertTrue(np.array_equal(produced, reference), label)


class TestSjrngBitExact(unittest.TestCase):
    def check(self, seed, trials, eids, chs, threads=0):
        requester = make_requester(eids, chs)
        ref = prepare_random_reference(seed, trials, requester)
        if threads:
            os.environ["SJRNG_THREADS"] = str(threads)
            try:
                got = sjrng.prepare_random(seed, trials, requester)
            finally:
                os.environ.pop("SJRNG_THREADS", None)
        else:
            got = sjrng.prepare_random(seed, trials, requester)
        names = ("istar", "erandom", "wanders", "accepts")
        for name, g, r in zip(names, got, ref):
            assert_bit_equal(self, g, r, f"seed={seed} trials={trials} "
                                          f"{name}")

    def test_official_seed_small(self):
        # The actual upstream invocation shape: seed 0, stream 1.
        self.check(0, 7, [0, 1, 2], [0, 1, 17611])

    def test_official_seed_trials_4000(self):
        # Full official FSMP chain length.
        eids = np.arange(24, dtype=np.int64)
        chs = (eids * 733) % 17612
        self.check(0, 4000, eids, chs)

    def test_trials_one_and_empty(self):
        self.check(0, 1, [5], [9])
        # Empty requester: upstream returns empty arrays.
        got = sjrng.prepare_random(0, 4000, make_requester([], []))
        for g in got:
            self.assertEqual(g.shape, (4000, 0))

    def test_alternate_seeds(self):
        for seed in (1, 42, 123456789, 2**31 - 1, 2**32):
            self.check(seed, 33, [0, 7, 999], [3, 1, 2**32 - 1])

    def test_boundary_ids(self):
        # eid/ch extremes incl. values that need two SeedSequence words.
        self.check(
            0, 64,
            [0, 1, 2**31 - 1, 2**32 - 1, 2**32, 2**53 + 3],
            [0, 2**32 - 1, 17611, 2**32, 2**53 + 7, 1],
        )

    def test_thread_count_invariance(self):
        eids = np.arange(40, dtype=np.int64)
        chs = (eids * 31337) % 17612
        requester = make_requester(eids, chs)
        single = sjrng.prepare_random(0, 250, requester)
        for threads in (1, 2, 4, 8):
            os.environ["SJRNG_THREADS"] = str(threads)
            try:
                got = sjrng.prepare_random(0, 250, requester)
            finally:
                os.environ.pop("SJRNG_THREADS", None)
            for g, s in zip(got, single):
                self.assertEqual(g.tobytes(), s.tobytes(),
                                 f"threads={threads} changed the streams")

    def test_ziggurat_branches_exercised(self):
        # The tail (|x| > ziggurat_nor_r = 3.65415...) and wedge branches of
        # numpy's ziggurat must be hit somewhere in a long run; if the port's
        # branch structure were wrong these values would mismatch anyway.
        eids = np.arange(64, dtype=np.int64)
        chs = np.arange(64, dtype=np.int64)
        requester = make_requester(eids, chs)
        _, _, wanders, _ = sjrng.prepare_random(0, 4000, requester)
        tail_hits = int(np.sum(np.abs(wanders) > 3.6541528853610088))
        self.assertGreater(tail_hits, 0, "ziggurat tail branch never taken")
        self.assertLess(
            tail_hits, 0.01 * wanders.size, "ziggurat tail branch too hot"
        )

    def test_accepts_are_log_of_uniform(self):
        requester = make_requester([11, 12], [13, 14])
        got = sjrng.prepare_random(0, 500, requester)
        ref = prepare_random_reference(0, 500, requester)
        assert_bit_equal(self, got[3], ref[3], "accepts")


class TestAgainstExtractedUpstream(unittest.TestCase):
    """C++ backend vs the function exec'd from the immutable git object.

    The supervisor reproduced the original wrapper bug with strided
    structured-array fields (ZCODE_REVIEW_1 item 1); these checks keep that
    failure mode covered against the true upstream body, across field
    widths, views, orderings, thread counts and boundary identities.
    """

    def check(self, label, seed, trials, requester, threads):
        reference = upstream_prepare_random(seed, trials, requester)
        os.environ["SJRNG_THREADS"] = str(threads)
        try:
            produced = sjrng.prepare_random(seed, trials, requester)
        finally:
            os.environ.pop("SJRNG_THREADS", None)
        for name, got, ref in zip(
                ("istar", "erandom", "wanders", "accepts"), produced,
                reference):
            self.assertEqual(got.dtype, np.float64, f"{label}/{name} dtype")
            self.assertEqual(got.shape, ref.shape, f"{label}/{name} shape")
            self.assertEqual(got.tobytes(), ref.tobytes(),
                             f"{label}/{name}: not bit-identical")

    def test_field_widths_views_and_orders(self):
        for width in ("i4", "i8"):
            requester = np.array(
                [(0, 0), (1, 5), (2, 10), (17, 17611)],
                dtype=[("eid", width), ("ch", width)])
            self.check(f"{width} plain", 0, 4000, requester, 1)
            self.check(f"{width} strided view", 0, 4000, requester[::2], 4)
            self.check(f"{width} reversed view", 0, 4000, requester[::-1], 4)
            for trials in (0, 1):
                self.check(f"{width} trials={trials}", 0, trials, requester, 1)
            self.check(f"{width} empty requester", 0, 4000, requester[:0], 1)

    def test_pipeline_shaped_two_field_view(self):
        # fsmp.py line 129: part_index[["eid", "ch"]] — uint32/uint16 fields,
        # non-contiguous 2-field view of a wider structured array.
        index = np.zeros(6, dtype=np.dtype([
            ("eid", np.uint32), ("ch", np.uint16), ("offset", np.uint16),
            ("s0", np.uint16)]))
        index["eid"] = [3, 1, 7, 0, 2 ** 31, 5]
        index["ch"] = [0, 17611, 5, 999, 1, 65535]
        self.check("pipeline view", 0, 4000, index[["eid", "ch"]], 4)

    def test_large_identities_and_alternate_seed(self):
        large = np.array(
            [(2 ** 32, 17611), (2 ** 53 + 3, 12), (2 ** 63 - 1, 1),
             (0, 2 ** 32)],
            dtype=[("eid", "i8"), ("ch", "i8")])
        self.check("large ids, 1 thread", 0, 4000, large, 1)
        self.check("large ids reversed, 4 threads", 0, 4000, large[::-1], 4)
        self.check("alternate seed, 2 threads", 42, 513, large, 2)

    def test_many_streams_hit_normal_slow_paths(self):
        # 80 random streams x 4000 draws: enough samples that the rare
        # ziggurat tail and wedge branches are exercised (statistically);
        # any branch-structure error shows up as a bitwise mismatch.
        rng = np.random.default_rng(12345)
        many = np.empty(80, dtype=[("eid", "i8"), ("ch", "i8")])
        many["eid"] = rng.integers(0, 2 ** 54, size=80, dtype=np.int64)
        many["ch"] = rng.integers(0, 17612, size=80, dtype=np.int64)
        self.check("80 random streams, 4 threads", 0, 4000, many, 4)
        _, _, wanders, _ = sjrng.prepare_random(0, 4000, many)
        tail_hits = int(np.sum(np.abs(wanders) > 3.6541528853610088))
        self.assertGreater(tail_hits, 0, "ziggurat tail never sampled")


class TestRequesterValidation(unittest.TestCase):
    """Malformed requesters must raise on the Python side before any C++
    call — no silent truncation, wrap-around or float coercion
    (ZCODE_REVIEW_1 item 1)."""

    def test_float_fields_rejected(self):
        requester = np.array(
            [(0.0, 0.0), (1.0, 5.0)], dtype=[("eid", "f8"), ("ch", "f8")])
        with self.assertRaises(ValueError):
            sjrng.prepare_random(0, 100, requester)

    def test_uint64_above_int64_max_rejected(self):
        requester = np.array(
            [(2 ** 63, 1)], dtype=[("eid", "u8"), ("ch", "u8")])
        with self.assertRaises(ValueError):
            sjrng.prepare_random(0, 100, requester)

    def test_uint64_in_range_accepted(self):
        # 2**63-1 fits int64 and must flow through unchanged.
        requester = np.array(
            [(2 ** 63 - 1, 7)], dtype=[("eid", "u8"), ("ch", "u8")])
        got = sjrng.prepare_random(0, 50, requester)
        ref = prepare_random_reference(0, 50, make_requester([2 ** 63 - 1], [7]))
        for g, r in zip(got, ref):
            self.assertEqual(g.tobytes(), r.tobytes())

    def test_negative_ids_rejected(self):
        requester = make_requester([-1], [0])
        with self.assertRaises(ValueError):
            sjrng.prepare_random(0, 100, requester)
        requester = make_requester([0], [-3])
        with self.assertRaises(ValueError):
            sjrng.prepare_random(0, 100, requester)

    def test_non_integer_scalar_arguments_rejected(self):
        requester = make_requester([1], [2])
        for bad in ((0.5, 100, requester), (0, 4000.0, requester),
                    (True, 100, requester)):
            with self.assertRaises(ValueError):
                sjrng.prepare_random(*bad)


class TestSjrngMicroBenchmark(unittest.TestCase):
    """LOCAL micro-benchmark only. Not a pipeline measurement, not a score.

    Measures the replaced hotspot in isolation: upstream's per-waveform
    Python loop vs the C++ backend, at one representative shape.
    """

    def test_benchmark(self):
        n_streams, trials = 5000, 4000
        rng = np.random.default_rng(7)
        eids = rng.integers(0, 2**31, n_streams, dtype=np.int64)
        chs = rng.integers(0, 17612, n_streams, dtype=np.int64)
        requester = make_requester(eids, chs)

        t0 = time.perf_counter()
        ref = prepare_random_reference(0, trials, requester)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        got = sjrng.prepare_random(0, trials, requester)
        t_got = time.perf_counter() - t0

        for g, r in zip(got, ref):
            self.assertEqual(g.tobytes(), r.tobytes())

        print(
            f"[micro] prepare_random streams={n_streams} trials={trials}: "
            f"numpy loop {t_ref:.3f} s, C++ {t_got:.3f} s "
            f"({t_ref / t_got:.1f}x, LOCAL micro-benchmark, not official)"
        )


if __name__ == "__main__":
    print(f"[info] {sjrng.build_id()}")
    print(f"[info] numpy {np.__version__}")
    unittest.main(verbosity=2)
