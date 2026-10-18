"""Seam test: the opt-in pieces installed into upstream/src work from the
pipeline's import location and stay bit-exact.

Run after `make install`:  python3 cpp/test_install.py

Checks (all LOCAL):
1. upstream/src/sjrng.py + _sjrng_core.so import exactly the way
   prepare.py's opt-in branch does (`import sjrng` with sys.path[0]=src/),
   including for a requester built like fsmp.py's
   `part_index[["eid", "ch"]]` two-field view.
2. The env-gated patched files still parse and their default branch is the
   untouched upstream call (guarded by a literal source check).

Not covered locally: cupy-dependent execution (no CUDA locally) — that is
CLUSTER_PENDING via `python scripts/run.py one --compare`.
"""

import ast
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_SRC = os.path.join(ROOT, "upstream", "src")


class TestInstalledSeam(unittest.TestCase):
    def test_import_from_upstream_src(self):
        sys.path.insert(0, UPSTREAM_SRC)
        try:
            import sjrng

            self.assertIn(sjrng.__file__, [
                os.path.join(UPSTREAM_SRC, "sjrng.py"),
            ])
        finally:
            sys.path.pop(0)

    def test_pipeline_shaped_requester(self):
        # Build a full index-like structured array and slice ["eid","ch"]
        # exactly like src/fsmp.py line 129 does.
        dt = np.dtype([
            ("eid", np.uint32),
            ("ch", np.uint16),
            ("offset", np.uint16),
            ("s0", np.uint16),
        ])
        index = np.zeros(5, dtype=dt)
        index["eid"] = [3, 1, 7, 0, 2**31]
        index["ch"] = [0, 17611, 5, 999, 1]
        requester = index[["eid", "ch"]]

        sys.path.insert(0, UPSTREAM_SRC)
        try:
            import sjrng
            got = sjrng.prepare_random(0, 400, requester)
        finally:
            sys.path.pop(0)

        ref = [np.empty((400, 5)) for _ in range(4)]
        for i, (eid, ch) in enumerate(requester):
            rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([0, 1, int(eid), int(ch)])))
            ref[0][:, i] = rng.random(400)
            ref[1][:, i] = rng.random(400)
            ref[2][:, i] = rng.standard_normal(400, np.float64)
            ref[3][:, i] = np.log(rng.random(400))
        for g, r in zip(got, ref):
            self.assertEqual(g.tobytes(), r.tobytes())

    def test_patched_files_default_branch_intact(self):
        move = os.path.join(UPSTREAM_SRC, "analysis", "fsmp", "move.py")
        prep = os.path.join(UPSTREAM_SRC, "analysis", "fsmp", "prepare.py")
        for path in (move, prep):
            with open(path) as fh:
                tree = ast.parse(fh.read(), filename=path)  # parses cleanly
            self.assertTrue(tree is not None)
        with open(move) as fh:
            src = fh.read()
        self.assertIn("beta = cp.linalg.inv(beta_inv)", src)  # default branch
        self.assertIn('os.environ.get("SUPERJUNO_BETA_INV") == "1"', src)
        with open(prep) as fh:
            src = fh.read()
        self.assertIn('os.environ.get("SUPERJUNO_SJRNG") == "1"', src)
        # The upstream numpy loop is still present as the fallback body.
        self.assertIn("istar[:, i] = rng.random(TRIALS)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
