"""Small end-to-end checks for the contestant result validators."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from comparison import ATOL, MAX_ERRORS, RTOL, differences

SCRIPTS = Path(__file__).resolve().parent


def pe(rows=2):
    return pd.DataFrame([
        dict(eid=12, ch=3, step=50 + i, offset=0, PEt=[0., 2.],
             e=[0, 1], q=[0., 0.], nu_lc=0.)
        for i in range(rows)
    ])


def tve():
    dtype = [(name, "i8") for name in ("EventID", "step", "move", "count")]
    dtype += [(name, "f8") for name in
              ("x", "y", "z", "E", "t", "NPE", "Likelihood", "epsilon")]
    dtype += [("acceptv", "?")]
    result = np.zeros((2, 2, 2), dtype=dtype)
    result["EventID"] = np.array([12, 13])[None, :, None]
    result["step"] = [50, 51]
    result["count"] = 1
    result["NPE"] = 3
    result["x"][1] = 2
    return result


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, kind, name, data):
        path = self.directory / name
        if kind == "PE":
            data.to_parquet(path, index=False)
        else:
            with h5py.File(path, "w") as handle:
                handle["Recon"] = data
        return path

    def check(self, kind, reference, candidate, code):
        ref = self.write(kind, "reference", reference)
        cand = self.write(kind, "candidate", candidate)
        command = [sys.executable, str(SCRIPTS / f"compare_{kind}.py"),
                   str(ref), str(cand)]
        result = subprocess.run(command, cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        lines = result.stdout.splitlines()
        self.assertTrue(lines)
        self.assertLessEqual(len(lines), MAX_ERRORS if code == 1 else 1)
        self.assertTrue(all(line.startswith(f"{code}: ") for line in lines))
        self.assertTrue(result.stdout.isascii())
        report = self.directory / "output" / kind / f"{cand.name}.compare.txt"
        self.assertEqual(report.read_text(encoding="utf-8"), result.stdout)
        if code == 0:
            self.assertEqual(result.stdout, f"0: PASS (atol={ATOL:g}, rtol={RTOL:g})\n")
        return result.stdout

    def test_absolute_tolerance_boundary(self):
        for delta, code in ((0.5 * ATOL, 0), (ATOL, 0),
                            (np.nextafter(ATOL, np.inf), 1), (1.5 * ATOL, 1)):
            for kind, reference in (("PE", pe()), ("tvE", tve())):
                with self.subTest(kind=kind, delta=delta):
                    candidate = reference.copy()
                    if kind == "PE":
                        candidate.at[0, "PEt"] = [delta, 2.]
                    else:
                        candidate["x"][0, 0, 0] = delta
                    message = self.check(kind, reference, candidate, code)
                    if code:
                        for detail in ("step=50", "ref=", "cand=", "abs_diff=", f"tol={ATOL:g}"):
                            self.assertIn(detail, message)

    def test_relative_tolerance_for_signed_references(self):
        for value in (1e9, -1e9):
            limit = ATOL + RTOL * abs(value)
            for factor, code in ((0.5, 0), (1.5, 1)):
                for kind, reference in (("PE", pe()), ("tvE", tve())):
                    with self.subTest(value=value, factor=factor, kind=kind):
                        field = "nu_lc" if kind == "PE" else "Likelihood"
                        reference[field] = value
                        candidate = reference.copy()
                        candidate[field] += factor * limit
                        message = self.check(kind, reference, candidate, code)
                        if code:
                            self.assertIn(f"tol={limit:g}", message)

    def test_tolerance_uses_reference(self):
        # Binary-exact limits distinguish reference scaling from candidate scaling.
        with patch("comparison.ATOL", 0.5), patch("comparison.RTOL", 0.25):
            for sign in (1, -1):
                for candidate, fails in ((1., False), (3., False),
                                         (np.nextafter(3., np.inf), True)):
                    with self.subTest(sign=sign, candidate=candidate):
                        messages = list(differences(sign * 2., sign * candidate,
                                                    "x", "vertex=0"))
                        self.assertEqual(len(messages), int(fails))

    def test_event_copies_with_different_relative_limits(self):
        for field in ("Likelihood", "epsilon"):
            with self.subTest(field=field):
                reference = tve()
                reference[field][0, 0, 0] = 1e9
                candidate = reference.copy()
                candidate[field][0, 0, 0] += 0.5 * (ATOL + RTOL * 1e9)
                candidate[field][1, 0, 0] = 2 * ATOL
                message = self.check("tvE", reference, candidate, 1)
                self.assertEqual(len(message.splitlines()), 1)
                for detail in (field, "EventID=12", "step=50", "vertex=1",
                               f"tol={ATOL:g}"):
                    self.assertIn(detail, message)

    def test_zero_pe_states(self):
        reference = pe(1)
        for field in ("PEt", "e", "q"):
            reference.at[0, field] = []
        self.check("PE", reference, reference, 0)

    def test_record_order(self):
        reference = pe()
        self.check("PE", reference, reference.iloc[::-1], 0)
        reference = tve()
        self.check("tvE", reference, reference[:, ::-1, ::-1], 0)

    def test_first_max_errors_mismatches(self):
        reference = pe(MAX_ERRORS + 2)
        candidate = reference.copy()
        candidate["nu_lc"] = 2 * ATOL
        message = self.check("PE", reference, candidate.iloc[::-1], 1)
        self.assertEqual(len(message.splitlines()), MAX_ERRORS)
        for step, line in zip(range(50, 50 + MAX_ERRORS), message.splitlines()):
            self.assertIn(f"step={step}", line)
        reference = np.repeat(tve()[:, :1, :1], MAX_ERRORS + 2, axis=2)
        reference["step"] = np.arange(50, 50 + MAX_ERRORS + 2)
        candidate = reference.copy()
        candidate["x"][0] = 2 * ATOL
        message = self.check("tvE", reference, candidate, 1)
        self.assertEqual(len(message.splitlines()), MAX_ERRORS)
        for step, line in zip(range(50, 50 + MAX_ERRORS), message.splitlines()):
            self.assertIn(f"step={step}", line)

    def test_nonfinite_values_and_q_placeholders(self):
        reference = pe()
        reference.at[0, "q"] = [np.nan, np.nan]
        self.check("PE", reference, reference.copy(), 0)
        candidate = reference.copy()
        candidate.at[0, "q"] = [0., np.nan]
        self.check("PE", reference, candidate, 1)
        for kind, reference in (("PE", pe()), ("tvE", tve())):
            for value in (np.nan, np.inf):
                if kind == "PE":
                    reference["nu_lc"] = value
                else:
                    reference["x"][0] = value
                self.check(kind, reference, reference.copy(), 1)

    def test_inactive_vertices_and_vertex_order(self):
        reference = tve()
        candidate = reference.copy()
        candidate["x"][1] = np.nan
        self.check("tvE", reference, candidate, 0)
        for field in ("Likelihood", "epsilon"):
            candidate[field][0] = 0.9 * ATOL
            candidate[field][1] = -0.9 * ATOL
        self.check("tvE", reference, candidate, 0)
        candidate["Likelihood"][1, 0, 0] = 2 * ATOL
        self.check("tvE", reference, candidate, 1)
        candidate = reference.copy()
        candidate["NPE"][1, 0, 0] = np.nan
        self.check("tvE", reference, candidate, 1)
        reference["count"] = 2
        self.check("tvE", reference, reference[::-1], 1)
        reference = pe()
        candidate = reference.copy()
        candidate.at[0, "PEt"] = [2., 0.]
        candidate.at[0, "e"] = [1, 0]
        self.check("PE", reference, candidate, 1)

    def test_missing_records_and_exact_fields(self):
        self.check("PE", pe(), pe().iloc[:1], 1)
        self.check("tvE", tve(), tve()[:, :1], 1)
        candidate = tve()
        candidate["step"] += 10
        self.check("tvE", tve(), candidate, 1)
        candidate = pe()
        for field in ("PEt", "e", "q"):
            candidate.at[0, field] = candidate.at[0, field][:1]
        self.check("PE", pe(), candidate, 1)
        for field in ("offset", "e"):
            candidate = pe()
            candidate.at[0, field] = 1 if field == "offset" else [1, 1]
            self.check("PE", pe(), candidate, 1)
        for field in ("move", "count", "NPE", "acceptv"):
            candidate = tve()
            candidate[field] += True if field == "acceptv" else 1
            self.check("tvE", tve(), candidate, 1)

    def test_invalid_inputs(self):
        for candidate in (pe().iloc[:0], pd.concat([pe(), pe()]), pe().drop(columns="q")):
            self.check("PE", pe(), candidate, 2)
        reference = tve()
        duplicate = reference.copy()
        duplicate["step"] = 50
        for candidate in (reference[:, :0], duplicate, reference[["EventID", "step"]]):
            self.check("tvE", reference, candidate, 2)
        for kind in ("PE", "tvE"):
            for extra in ([], ["--rtol", "1e-3"], ["-o", "result.txt"]):
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / f"compare_{kind}.py"),
                     str(self.directory / "missing"), str(self.directory / "missing"),
                     *extra], cwd=self.directory, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stderr, "")
                self.assertEqual(len(result.stdout.splitlines()), 1)
                self.assertTrue(result.stdout.startswith("2: "))
                if not extra:
                    report = self.directory / "output" / kind / "missing.compare.txt"
                    self.assertEqual(report.read_text(encoding="utf-8"), result.stdout)


if __name__ == "__main__":
    unittest.main()
