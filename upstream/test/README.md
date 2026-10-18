# Output comparison

From the repository root, compare existing outputs with:

```bash
python test/compare_PE.py data/ana_PE/one.pq output/PE/one.pq
python test/compare_tvE.py data/ana_tvE/one.h5 output/tvE/one.h5
```

`one` is the example case and is not scored. The scored cases are `0`, `1`, and
`2`; replace `one` with the case ID to check each case's outputs.

| Script | Reference | Candidate |
| --- | --- | --- |
| `test/compare_PE.py` | `data/ana_PE/<id>.pq` | `output/PE/<id>.pq` |
| `test/compare_tvE.py` | `data/ana_tvE/<id>.h5` | `output/tvE/<id>.h5` |

These scripts use the selected Python interpreter and read existing files without
running GPU computations. Every invocation reruns the check, prints results
to the terminal, and saves the same text in `output/PE/<id>.pq.compare.txt` or
`output/tvE/<id>.h5.compare.txt`.

To run the pipeline and then both checks, use `python scripts/run.py <id> --compare`.
After a successful pipeline run, the runner executes both checks even if one fails,
and returns a nonzero exit code if either check fails.
The runner prints pipeline timing, comparison output, the corresponding
`output/logs/<id>-<runid>.time.csv` path in a `Timing:` line, and the final exit code
directly to stdout. Without `--compare`, it prints that comparison
was skipped. Pipeline timing excludes the comparisons.

To compare files at other paths, pass the reference and candidate as arguments:

```bash
python test/compare_PE.py reference.pq candidate.pq
python test/compare_tvE.py reference.h5 candidate.h5
```

Direct invocations also save reports under `output/PE/` or `output/tvE/`, relative
to the current working directory, using the candidate filename followed by
`.compare.txt`. The scripts create these directories as needed and overwrite the
report on each run, including when results differ or an input cannot be read.

## Results

Each output line has the form `CODE: MESSAGE`, with an English message. The
script's exit code is the same as `CODE`:

| Code | Meaning |
| --- | --- |
| `0` | All checks passed; one line. |
| `1` | Results differ; at most `MAX_ERRORS` issues, one per line. |
| `2` | The check could not run, such as an unreadable file or missing field; one line. |

Differences include the record key and field; numerical differences also include
the reference value, candidate value, absolute difference, and tolerance for that
reference value. Successful output is `0: PASS (atol=..., rtol=...)`, displaying
the configured absolute and relative tolerances.

## Comparison rules

Set `ATOL` and `RTOL` in [comparison.py](comparison.py) to choose the absolute and
relative tolerances, and `MAX_ERRORS` to set the maximum number of reported issues.
Floating-point values pass when
`abs(candidate - reference) <= ATOL + RTOL * abs(reference)`.
Record keys, counts, flags, and other discrete values must
match exactly. Missing or extra records fail; record storage order does not
matter.

- **PE:** Match records by `(eid, ch, step)`. Compare `offset`, `e`, and array
  lengths exactly; compare `PEt`, `q`, and `nu_lc` with the tolerance. PE arrays
  are compared in their saved slot order.
- **tvE:** Match records by `(EventID, step)`. Compare `move`, `count`, `NPE`, and
  `acceptv` exactly; compare `x`, `y`, `z`, `E`, `t`, `Likelihood`, and `epsilon`
  with the tolerance. Vertex fields (`x/y/z/E/t`) use only the first `count`
  slots in saved order. Event-level numeric fields are checked across all stored
  copies, reporting at most one issue per field per step.

Matching missing values in PE `q` are allowed because `q` is a placeholder when
MMSE is disabled. A missing value on only one side fails. Other NaN or infinite
values in compared data fail.

Run the comparator regression tests with:

```bash
python -m unittest discover -s test -p 'test_*.py'
```
