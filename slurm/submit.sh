#!/bin/bash
# submit.sh — safe submission wrapper for the SuperJUNO pipeline.
#
# Usage (from the repository root, with run.slurm also in the root):
#   SJ_PARTITION=GPU_PARTITION ./slurm/submit.sh one        # validate first
#   # after `one` passes (Final exit code: 0 in its .out):
#   SJ_PARTITION=GPU_PARTITION ./slurm/submit.sh 0 1 2      # scored cases
#
# Environment:
#   SJ_PARTITION        4090 partition assigned by the organizers (required
#                       unless SBATCH has a default; set to empty to omit).
#   SJ_GPUS             GPUs per job, 1 or 2. Default 1. The rule allows at
#                       most 2 per run.
#   SJ_REPO_ROOT        repository root (default $HOME/run/superjuno).
#   SJ_SKIP_ONE_GUARD=1 submit 0/1/2 even if no `one` run was recorded.
#   SJ_SKIP_DATA_CHECK=1 skip the input-data presence check (local control-
#                       flow tests only; never on the cluster).
#
# Per <id>, in this exact order (check first, archive only afterwards):
#   0. argument/resource validation (duplicates, unknown ids, missing
#      sbatch/squeue/run.slurm/scripts/run.py) — before ANY output path is
#      created, so a bad id like `../../x` can never reach a path;
#   1. acquire the per-id lock output/locks/<id>.lock (mkdir; a live
#      concurrent holder blocks, a dead holder's stale lock is taken over);
#   2. refuse if a previously recorded job for this id is still pending or
#      running according to squeue (so an active run is never archived or
#      double-submitted);
#   3. refuse if the official input files for this id are missing;
#   4. archive that id's previous artifacts and logs under
#      output/archive/<id>/<UTC timestamp>[-N]/ (moved, never deleted;
#      the -N suffix keeps directories unique even within one second);
#   5. pre-create output/logs and output/tmp/... so Slurm can open the
#      %x-%j.out/.err files before the job starts;
#   6. sbatch, then record (timestamp, id, job id, partition, GPUs, active
#      opt-in switches) in output/logs/submissions.tsv.
# A failure for one id (busy lock, active job, missing data, sbatch error)
# releases that id's lock and does NOT affect the other ids; the script
# exits nonzero if any id failed.
#
# State boundaries (documented, intentionally simple):
#   * Active-job detection lists the jobs currently queued or running for
#     this user (`squeue -h -u "$USER" -o %A`) and intersects that with the
#     job ids ever recorded for the id in submissions.tsv (a ";cluster"
#     suffix recorded from `sbatch --parsable` is stripped before comparing).
#     Historical ids are never passed to `squeue -j`: finished jobs the
#     controller has already purged are simply absent from the queue listing
#     and may be archived, whereas `squeue -j <purged id>` fails with
#     "Invalid job id" and would block re-submission of the case forever.
#     If the queue listing itself fails, the id is refused (fail-safe)
#     rather than guessed about.
#   * The lock is host-local (pid liveness). It guards concurrent
#     submit.sh runs on one login node; cluster state is guarded by the
#     squeue check, and finished artifacts are additionally protected by
#     run.slurm's own overwrite refusal.
#   * If submissions.tsv is lost, active-job detection degrades to the
#     file-existence checks in run.slurm (documented residual risk).
#
# Opt-in switches (SUPERJUNO_SJRNG, SUPERJUNO_BETA_INV) propagate through
# sbatch's default --export=ALL and are recorded for the audit trail.

set -euo pipefail

SJ_GPUS="${SJ_GPUS:-1}"
SJ_REPO_ROOT="${SJ_REPO_ROOT:-$HOME/run/superjuno}"

die() { echo "submit.sh: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0) Argument and resource validation — no output path is touched below
#    this block, and no path is ever built from an unvalidated id.
# ---------------------------------------------------------------------------
[ "$SJ_GPUS" = "1" ] || [ "$SJ_GPUS" = "2" ] || die "SJ_GPUS must be 1 or 2"

ids=("$@")
[ ${#ids[@]} -gt 0 ] || die "usage: SJ_PARTITION=<part> [SJ_GPUS=1] $0 <one|0|1|2> [...]"
declare -A seen=()
for id in "${ids[@]}"; do
    case "$id" in
        one|0|1|2) ;;
        *) die "invalid case id '$id' (one|0|1|2)" ;;
    esac
    [ -z "${seen[$id]:-}" ] || die "duplicate case id '$id' in one submission"
    seen[$id]=1
done

command -v sbatch >/dev/null 2>&1 || die "sbatch not found (submit from a Slurm login node)"
command -v squeue >/dev/null 2>&1 || die "squeue not found (submit from a Slurm login node)"
RUN_SLURM="$SJ_REPO_ROOT/run.slurm"
[ -f "$RUN_SLURM" ] || die "run.slurm not found at $RUN_SLURM (copy slurm/run.slurm to the repo root)"
[ -f "$SJ_REPO_ROOT/scripts/run.py" ] || die "scripts/run.py not found under $SJ_REPO_ROOT (upstream tree missing)"

# ---------------------------------------------------------------------------
cd "$SJ_REPO_ROOT"
mkdir -p output/logs output/sparsify output/PE output/tvE \
         output/tmp/runtime output/tmp/cupy output/tmp/cuda output/tmp/xdg \
         output/tmp/pycache output/archive output/locks
SUBMISSIONS=output/logs/submissions.tsv
[ -f "$SUBMISSIONS" ] || printf 'submit_utc\tid\tjob_id\tpartition\tgpus\tsjrng\tbeta_inv\n' > "$SUBMISSIONS"

# Guard: scored submissions should follow a passing `one` validation.
if [ "${SJ_SKIP_ONE_GUARD:-0}" != "1" ]; then
    has_scored=0
    for id in "${ids[@]}"; do [ "$id" != one ] && has_scored=1; done
    if [ "$has_scored" = 1 ] && ! grep -q $'\tone\t' "$SUBMISSIONS"; then
        die "no recorded 'one' validation run; submit and verify 'one' first (or SJ_SKIP_ONE_GUARD=1)"
    fi
fi

# Input files the official pipeline reads for this id (scripts/run.py
# build_pipeline + --compare references). Missing input = certain job
# failure, so refuse before submitting.
check_input_data() {  # $1 = id; returns 0 if all inputs present
    local id=$1 missing=0 f
    for f in "data/raw/$id.h5" "data/ser_truth_J25.7.4+.h5" "data/sim.h5" \
             "data/PMT_J26.4.0.h5" "data/junoid_J25.3.0.pq" \
             "data/J25.3.0.PMTPos_CD_LPMT.csv" \
             "data/ana_PE/$id.pq" "data/ana_tvE/$id.h5"; do
        if [ ! -e "$f" ]; then
            echo "submit.sh: missing official input $f for id '$id'" >&2
            missing=1
        fi
    done
    [ "$missing" = 0 ]
}

# --- per-id lock ------------------------------------------------------------
HELD_LOCK=""
lock_acquire() {  # $1 = id; 0 = acquired, 1 = busy
    local id=$1 lock="output/locks/$id.lock" owner_pid
    if mkdir "$lock" 2>/dev/null; then
        printf 'pid=%s started=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lock/owner"
        HELD_LOCK="$lock"
        return 0
    fi
    owner_pid=$(sed -n 's/^pid=//p' "$lock/owner" 2>/dev/null || true)
    if [ -n "$owner_pid" ] && [ "$owner_pid" != "$$" ] \
       && ! kill -0 "$owner_pid" 2>/dev/null; then
        # Holder process is gone: stale lock, safe to take over.
        rm -rf "$lock"
        if mkdir "$lock" 2>/dev/null; then
            printf 'pid=%s started=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lock/owner"
            HELD_LOCK="$lock"
            return 0
        fi
    fi
    echo "submit.sh: id '$id' is locked by another live submit.sh (pid ${owner_pid:-unknown})" >&2
    return 1
}
lock_release() {
    if [ -n "${HELD_LOCK:-}" ]; then
        rm -rf "$HELD_LOCK"
        HELD_LOCK=""
    fi
    return 0
}
trap 'lock_release' EXIT

# --- active-job check -------------------------------------------------------
active_check() {  # $1 = id; 0 = clear, 1 = active or unverifiable
    local id=$1 jobs out job active=""
    [ -f "$SUBMISSIONS" ] || return 0
    jobs=$(awk -F'\t' -v id="$id" 'NR>1 && $2==id {print $3}' "$SUBMISSIONS")
    [ -n "$jobs" ] || return 0
    # List this user's queued/running jobs once, then intersect with the
    # recorded ids. Querying the historical ids themselves (`squeue -j`)
    # fails with "Invalid job id" once the controller purges a finished
    # job, which would block archiving/re-running this case forever.
    if ! out=$(squeue -h -u "${USER:-$(id -un)}" -o %A 2>&1); then
        echo "submit.sh: cannot verify cluster state of previous '$id' job(s) $jobs (squeue failed); refusing" >&2
        return 1
    fi
    while IFS= read -r job; do
        [ -n "$job" ] || continue
        job=${job%%;*}   # sbatch --parsable may record "<id>;<cluster>"
        if grep -qxF -- "$job" <<<"$out"; then
            active="$active $job"
        fi
    done <<<"$jobs"
    if [ -n "$active" ]; then
        echo "submit.sh: id '$id' still has active Slurm job(s);$active not archiving, not submitting" >&2
        return 1
    fi
    return 0
}

# --- archive ----------------------------------------------------------------
archive_id() {  # $1 = id; moves prior artifacts aside, never deletes
    local id=$1 base dest n moved=0 f
    base="output/archive/$id/$(date -u +%Y%m%dT%H%M%SZ)"
    dest=$base
    n=0
    while [ -e "$dest" ]; do      # unique even within the same second
        n=$((n + 1))
        dest="$base-$n"
    done
    mkdir -p "$dest"
    for f in "output/sparsify/$id.h5" "output/sparsify/$id.h5.log" \
             "output/PE/$id.pq" "output/PE/$id.pq.log" \
             "output/PE/$id.pq.compare.txt" \
             "output/tvE/$id.h5" "output/tvE/$id.h5.log" \
             "output/tvE/$id.h5.compare.txt" \
             output/logs/"$id"-*.time.csv; do
        if [ -e "$f" ]; then
            if ! mv "$f" "$dest/"; then
                echo "submit.sh: archiving '$id' failed while moving $f (already-moved files stay in $dest; re-run archives again)" >&2
                return 1
            fi
            moved=1
        fi
    done
    if [ "$moved" = 1 ]; then
        echo "archived previous $id artifacts -> $dest"
    else
        rmdir "$dest"
    fi
    return 0
}

submit_one() {  # $1 = id; returns 0 only if the job was submitted
    local id=$1 job_id
    lock_acquire "$id" || return 1
    if ! active_check "$id"; then lock_release; return 1; fi
    if [ "${SJ_SKIP_DATA_CHECK:-0}" != "1" ]; then
        if ! check_input_data "$id"; then
            echo "submit.sh: id '$id' NOT submitted (missing official input data; SJ_SKIP_DATA_CHECK=1 overrides for local tests only)" >&2
            lock_release
            return 1
        fi
    fi
    if ! archive_id "$id"; then lock_release; return 1; fi
    local args=(run.slurm "$id")
    if [ -n "${SJ_PARTITION:-}" ]; then args=(--partition="$SJ_PARTITION" "${args[@]}"); fi
    if [ "$SJ_GPUS" != "1" ]; then args=(--gres="gpu:$SJ_GPUS" "${args[@]}"); fi
    if ! job_id=$(sbatch --parsable "${args[@]}"); then
        echo "submit.sh: sbatch failed for id '$id'" >&2
        lock_release
        return 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$id" "$job_id" \
        "${SJ_PARTITION:-<default>}" "$SJ_GPUS" \
        "${SUPERJUNO_SJRNG:-0}" "${SUPERJUNO_BETA_INV:-0}" >> "$SUBMISSIONS"
    echo "submitted $id -> job $job_id (output/logs/superjuno-$job_id.out)"
    lock_release
    return 0
}

failed=0
for id in "${ids[@]}"; do
    if ! submit_one "$id"; then
        echo "submit.sh: id '$id' NOT submitted (see messages above)" >&2
        failed=$((failed + 1))
    fi
done
if [ "$failed" -gt 0 ]; then
    die "$failed of ${#ids[@]} id(s) failed to submit"
fi

cat <<'EOF'
Next:
  squeue -u "$USER"                                   # watch the queue
  grep -E 'TOTAL|Pipeline exit code|Comparison exit code|Final exit code' \
      output/logs/superjuno-<JOBID>.out               # verify after the run
The Timing: line in each .out names that run's output/logs/<id>-*.time.csv;
all artifacts of the same run stay in place until the next submission of the
same id archives them (and only if no job for that id is still active).
EOF
