#!/bin/bash
# status.sh — summarize submitted jobs and their pipeline results.
#
# Usage (from the repository root): ./slurm/status.sh [job_id ...]
# Without arguments, all jobs recorded in output/logs/submissions.tsv.

set -euo pipefail

SJ_REPO_ROOT="${SJ_REPO_ROOT:-$HOME/run/superjuno}"
cd "$SJ_REPO_ROOT"
SUBMISSIONS=output/logs/submissions.tsv

[ -f "$SUBMISSIONS" ] || { echo "no submissions recorded yet"; exit 0; }

if [ "$#" -gt 0 ]; then
    job_ids=("$@")
else
    mapfile -t job_ids < <(tail -n +2 "$SUBMISSIONS" | cut -f3)
fi

for job in "${job_ids[@]}"; do
    out="output/logs/superjuno-$job.out"
    echo "=== job $job ==="
    if [ ! -f "$out" ]; then
        echo "  no stdout yet (queued or running): output/logs/superjuno-$job.out"
        continue
    fi
    grep -E 'Case:|^TOTAL:|Pipeline exit code|Timing:|Comparison exit code|Final exit code|PASS|FAIL|REFUSED' "$out" \
        | sed 's/^/  /' || echo "  (no result lines yet)"
done
