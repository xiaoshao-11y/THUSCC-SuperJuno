#!/bin/bash
# Local control-flow regression for slurm/submit.sh and slurm/run.slurm.
#
# sbatch and squeue are EXPLICIT STUBS on PATH (no cluster is contacted, no
# GPU is touched, nothing is submitted anywhere). This file proves the shell
# control flow demanded by ZCODE_REVIEW_2:
#   1. a first clean submission of `one` succeeds (exit 0);
#   2. an id with a still-queued/running job is refused — no archive, no
#      second sbatch;
#   3. resubmitting a finished id archives the previous artifacts first;
#   4. duplicate ids in one invocation are rejected before anything else;
#   5. invalid ids / SJ_GPUS>2 are rejected before any output path exists;
#   6. missing official input data is refused (and SJ_SKIP_DATA_CHECK=1
#      overrides for local tests);
#   7. a failure on one id does not block other ids in the same invocation;
#   8. archive directories never collide within the same second;
#   9. run.slurm validates its argument and Slurm environment BEFORE any
#      output path is created (bad id / no Slurm / 2 nodes / 3 GPUs / stale
#      products refusal paths);
#  10. historical job ids already purged by the controller (absent from the
#      per-user queue listing, other live jobs present) allow archiving and
#      re-submission — the `squeue -j <purged id>` "Invalid job id" trap;
#  11. ids recorded as "<job>;<cluster>" (sbatch --parsable suffix) still
#      refuse while that job is queued and re-submit after it leaves;
#  12. an squeue query failure conservatively refuses (no archive, no sbatch);
#  13. submit.sh never queries squeue with -j; it lists the user's live jobs
#      (-u <user> -o %A) and intersects them with the recorded ids.
# This is LOCAL control-flow verification only: it is NOT a record of any
# real Slurm submission, PE/tvE comparison or GPU run.

set -uo pipefail   # no -e: we deliberately run commands that must fail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); echo "ok   - $1"; }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL - $1"; }

expect_exit_zero() {  # desc cmd...
    local desc=$1; shift
    if "$@" >"$TMP/last_stdout" 2>"$TMP/last_stderr"; then ok "$desc"; else
        bad "$desc (exit $?, stderr: $(tail -2 "$TMP/last_stderr" | tr '\n' ' '))"
    fi
}
expect_exit_nonzero() {  # desc cmd...
    local desc=$1; shift
    if "$@" >/dev/null 2>"$TMP/last_stderr"; then
        bad "$desc (unexpectedly exited 0)"
    else ok "$desc"; fi
}
assert_file() { [ -e "$2" ] && ok "$1" || bad "$1 ($2 missing)"; }
assert_no_file() { [ ! -e "$2" ] && ok "$1" || bad "$1 ($2 exists)"; }
sbatch_calls() { grep -c '^sbatch' "$STUB_LOG"; }

# ---------------------------------------------------------------- stubs ----
export STUB_STATE="$TMP/stubstate"
export STUB_LOG="$TMP/stub.log"
export STUB_QUEUE="$TMP/stub.queue"   # lines: "<jobid>|<STATE>"
mkdir -p "$STUB_STATE" "$TMP/bin"
: > "$STUB_LOG"; : > "$STUB_QUEUE"

cat > "$TMP/bin/sbatch" <<'EOF'
#!/bin/bash
# stub: record invocation, return a deterministic new job id (100, 101, ...).
# STUB_SBATCH_SUFFIX emulates `sbatch --parsable` on a multi-cluster system
# returning "<jobid>;<cluster>".
echo "sbatch $*" >> "$STUB_LOG"
n=$(cat "$STUB_STATE/next_jobid" 2>/dev/null || echo 100)
echo $((n + 1)) > "$STUB_STATE/next_jobid"
echo "${n}${STUB_SBATCH_SUFFIX:-}"
EOF
cat > "$TMP/bin/squeue" <<'EOF'
#!/bin/bash
# stub: supports the per-user listing submit.sh uses (`-h -u <user> -o %A`);
# prints one job id per line for jobs still queued/running, from $STUB_QUEUE
# (lines "<jobid>|<STATE>"). `-j` is refused outright: submit.sh must never
# query historical ids directly — real squeue fails "Invalid job id" for jobs
# the controller has purged. STUB_SQUEUE_FAIL=1 emulates a query failure.
echo "squeue $*" >> "$STUB_LOG"
if [ "${STUB_SQUEUE_FAIL:-0}" = "1" ]; then
    echo "squeue-stub: simulated scheduler query failure" >&2
    exit 1
fi
while [ $# -gt 0 ]; do
    case "$1" in
        -h) ;;
        -u) user=$2; shift ;;
        -o) fmt=$2; shift ;;
        -j) echo "squeue-stub: -j not supported (historical-id queries break on purged jobs)" >&2; exit 1 ;;
        *) echo "squeue-stub: unsupported argument '$1'" >&2; exit 1 ;;
    esac
    shift
done
cut -d'|' -f1 "$STUB_QUEUE" 2>/dev/null || true
exit 0
EOF
chmod +x "$TMP/bin/sbatch" "$TMP/bin/squeue"
export PATH="$TMP/bin:$PATH"

# ------------------------------------------------------------- fixtures ----
make_fixture() {  # $1 = root dir; $2 = "with-data" | "no-data"
    local root=$1 mode=$2 id
    mkdir -p "$root/scripts"
    touch "$root/scripts/run.py"
    cp "$REPO/slurm/run.slurm" "$root/run.slurm"
    if [ "$mode" = with-data ]; then
        mkdir -p "$root/data/raw" "$root/data/ana_PE" "$root/data/ana_tvE"
        touch "$root/data/ser_truth_J25.7.4+.h5" "$root/data/sim.h5" \
              "$root/data/PMT_J26.4.0.h5" "$root/data/junoid_J25.3.0.pq" \
              "$root/data/J25.3.0.PMTPos_CD_LPMT.csv"
        for id in one 0 1 2; do
            touch "$root/data/raw/$id.h5" "$root/data/ana_PE/$id.pq" \
                  "$root/data/ana_tvE/$id.h5"
        done
    fi
}

submit() {  # $1 = fixture root, rest = command (forwards current PATH)
    local root=$1; shift
    env SJ_REPO_ROOT="$root" "$@"
}

# =========================================================== submit.sh =====
ROOT="$TMP/rootA"
make_fixture "$ROOT" with-data
SUB="$REPO/slurm/submit.sh"

# 1) first clean submission of `one` succeeds (exit 0) — the bug fixed by
#    ZCODE_REVIEW_2 item 1 was exactly this case exiting 1.
expect_exit_zero "clean first submission of one exits 0" \
    submit "$ROOT" bash "$SUB" one
assert_file "submissions.tsv recorded" "$ROOT/output/logs/submissions.tsv"
[ "$(awk 'END{print NR}' "$ROOT/output/logs/submissions.tsv")" = 2 ] \
    && ok "submissions.tsv has header + 1 row (job 100)" \
    || bad "submissions.tsv row count"
assert_file "output/logs pre-created for Slurm stdout" "$ROOT/output/logs"
assert_no_file "lock released after success" "$ROOT/output/locks/one.lock"

# 2) same id with an ACTIVE job (job 100 still RUNNING): refuse; do not
#    archive existing products; no second sbatch.
mkdir -p "$ROOT/output/PE"
echo fake > "$ROOT/output/PE/one.pq"
echo "100|RUNNING" > "$STUB_QUEUE"
before=$(sbatch_calls)
expect_exit_nonzero "active same-id submission refused" \
    submit "$ROOT" bash "$SUB" one
assert_file "active job: existing product NOT archived" "$ROOT/output/PE/one.pq"
[ "$(sbatch_calls)" = "$before" ] && ok "active job: no new sbatch" \
    || bad "active job: extra sbatch issued"
: > "$STUB_QUEUE"   # job 100 leaves the queue

# 3) resubmit after the job finished: old products archived, sbatch again.
expect_exit_zero "finished id resubmits and archives" \
    submit "$ROOT" bash "$SUB" one
assert_no_file "old product moved out of PE/" "$ROOT/output/PE/one.pq"
[ -n "$(find "$ROOT/output/archive/one" -name one.pq 2>/dev/null)" ] \
    && ok "old product preserved in output/archive/one/" \
    || bad "archived copy not found"

# 4) duplicate ids rejected before any sbatch.
before=$(sbatch_calls)
expect_exit_nonzero "duplicate ids (one one) rejected" \
    submit "$ROOT" bash "$SUB" one one
[ "$(sbatch_calls)" = "$before" ] && ok "duplicates: no sbatch" \
    || bad "duplicates: sbatch still issued"

# 5) invalid id and SJ_GPUS>2 rejected before any output path is created.
ROOT2="$TMP/rootB"
make_fixture "$ROOT2" with-data
expect_exit_nonzero "invalid id ../../x rejected" \
    submit "$ROOT2" bash "$SUB" ../../x
assert_no_file "invalid id: output/ never created" "$ROOT2/output"
expect_exit_nonzero "SJ_GPUS=3 rejected" \
    submit "$ROOT2" env SJ_GPUS=3 bash "$SUB" one
assert_no_file "SJ_GPUS=3: output/ never created" "$ROOT2/output"

# 6) missing official input data refused; SJ_SKIP_DATA_CHECK=1 overrides.
ROOT3="$TMP/rootC"
make_fixture "$ROOT3" no-data
before=$(sbatch_calls)
expect_exit_nonzero "missing data inputs refused" submit "$ROOT3" bash "$SUB" one
[ "$(sbatch_calls)" = "$before" ] && ok "missing data: no sbatch" \
    || bad "missing data: sbatch issued"
expect_exit_zero "SJ_SKIP_DATA_CHECK=1 overrides for local tests" \
    submit "$ROOT3" env SJ_SKIP_DATA_CHECK=1 bash "$SUB" one

# 7) one busy id does not block other ids in the same invocation.
ROOT4="$TMP/rootD"
make_fixture "$ROOT4" with-data
submit "$ROOT4" env SJ_SKIP_ONE_GUARD=1 bash "$SUB" one 0 1 2 >/dev/null 2>&1 || true   # jobs 103-106
: > "$STUB_QUEUE"
echo "104|PENDING" > "$STUB_QUEUE"      # job 104 = id `0`, still queued
before=$(sbatch_calls)
expect_exit_nonzero "mixed invocation with one active id exits nonzero" \
    submit "$ROOT4" bash "$SUB" 0 1
[ "$(sbatch_calls)" = "$((before + 1))" ] \
    && ok "non-active id (1) still submitted exactly once" \
    || bad "expected exactly one extra sbatch for id 1"
awk -F'\t' 'NR>1 && $2=="1" {found=1} END{exit !found}' \
    "$ROOT4/output/logs/submissions.tsv" \
    && ok "id 1 recorded in submissions.tsv" || bad "id 1 not recorded"
: > "$STUB_QUEUE"

# 8) archive directories never collide within the same second.
mkdir -p "$ROOT/output/tvE" "$ROOT/output/archive/one/$(date -u +%Y%m%dT%H%M%SZ)"
touch "$ROOT/output/tvE/one.h5" "$ROOT/output/tvE/one.h5.log"
expect_exit_zero "resubmission with colliding archive timestamp" \
    submit "$ROOT" bash "$SUB" one
[ "$(find "$ROOT/output/archive/one" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ge 2 ] \
    && ok "archive dirs kept distinct under collision" \
    || bad "archive dirs collided"
[ -n "$(find "$ROOT/output/archive/one" -name 'one.h5' 2>/dev/null)" ] \
    && ok "tvE products preserved in archive" || bad "tvE products lost"

# 9) one-guard: scored submission without any recorded `one` refused.
ROOT5="$TMP/rootE"
make_fixture "$ROOT5" with-data
expect_exit_nonzero "scored ids without prior one refused" \
    submit "$ROOT5" bash "$SUB" 0

# 10) purged historical jobs: history exists, but the per-user queue listing
#     no longer contains those ids (Slurm purges finished jobs) while OTHER
#     jobs of the same user are live. Re-submission must be allowed —
#     `squeue -j <purged id>` would fail "Invalid job id" forever.
ROOT6="$TMP/rootF2"
make_fixture "$ROOT6" with-data
submit "$ROOT6" bash "$SUB" one >/dev/null 2>&1 || true   # recorded history
mkdir -p "$ROOT6/output/PE" && echo old > "$ROOT6/output/PE/one.pq"
echo "999|RUNNING" > "$STUB_QUEUE"   # unrelated live job of the same user
before=$(sbatch_calls)
expect_exit_zero "purged historical ids allow archiving re-submission" \
    submit "$ROOT6" bash "$SUB" one
[ "$(sbatch_calls)" = "$((before + 1))" ] \
    && ok "purged ids: re-submitted exactly once" \
    || bad "purged ids: sbatch count wrong"
[ -n "$(find "$ROOT6/output/archive/one" -name one.pq 2>/dev/null)" ] \
    && ok "purged ids: previous artifact still archived" \
    || bad "purged ids: artifact not archived"
: > "$STUB_QUEUE"

# 11) ";cluster" suffix from `sbatch --parsable`: the recorded id carries a
#     suffix; comparison must strip it — active job still refused, then (once
#     it leaves the queue) the id re-submits.
ROOT7="$TMP/rootG"
make_fixture "$ROOT7" with-data
submit "$ROOT7" env STUB_SBATCH_SUFFIX=';gpu-a' bash "$SUB" one >/dev/null 2>&1 || true
rec=$(awk -F'\t' 'NR>1 && $2=="one" {print $3}' "$ROOT7/output/logs/submissions.tsv" | tail -1)
case "$rec" in
    *\;*) ok "job recorded with parsable suffix ($rec)" ;;
    *) bad "expected '<job>;<cluster>' record, got '$rec'"; rec=";" ;;
esac
echo "${rec%%;*}|RUNNING" > "$STUB_QUEUE"
before=$(sbatch_calls)
expect_exit_nonzero "active job refused despite ;cluster suffix in record" \
    submit "$ROOT7" bash "$SUB" one
[ "$(sbatch_calls)" = "$before" ] && ok "suffix case: no new sbatch while active" \
    || bad "suffix case: sbatch issued while active"
: > "$STUB_QUEUE"
expect_exit_zero "suffix-recorded id re-submits after job leaves queue" \
    submit "$ROOT7" bash "$SUB" one

# 12) squeue query failure: conservative refusal — no archive, no sbatch.
mkdir -p "$ROOT7/output/PE" && echo keep > "$ROOT7/output/PE/one.pq"
before=$(sbatch_calls)
expect_exit_nonzero "squeue failure conservatively refuses" \
    submit "$ROOT7" env STUB_SQUEUE_FAIL=1 bash "$SUB" one
assert_file "squeue failure: product NOT archived" "$ROOT7/output/PE/one.pq"
[ "$(sbatch_calls)" = "$before" ] && ok "squeue failure: no sbatch" \
    || bad "squeue failure: sbatch issued"

# 13) global guard: submit.sh only ever listed the user's live jobs
#     (`-u <user> -o %A`); no squeue -j with historical ids anywhere.
if grep -q -- ' -j' "$STUB_LOG"; then
    bad "squeue was called with -j (historical-id query)"
else ok "no squeue -j anywhere in the run"; fi
if grep -q -- ' -o %A' "$STUB_LOG" && grep -q -- ' -u ' "$STUB_LOG"; then
    ok "squeue queried via per-user listing (-u <user> -o %A)"
else bad "per-user queue listing form never used"; fi

# =========================================================== run.slurm =====
# Validation must refuse BEFORE any output path is created. The stub
# environment provides SLURM_* variables; a valid case stops later at the
# cluster-only `module load` (exit != 97/98/99) — that boundary is expected
# locally and documented; it is not a pipeline run.
R="$TMP/rootF"
make_fixture "$R" with-data
no_output_created() { assert_no_file "run.slurm refusal created no output/" "$R/output"; }

rc=$(cd "$R" && env SJ_REPO_ROOT="$R" bash run.slurm >/dev/null 2>&1; echo $?)
[ "$rc" = 99 ] && ok "run.slurm without argument -> 99" || bad "no-arg exit $rc"
no_output_created

rc=$(cd "$R" && env SJ_REPO_ROOT="$R" bash run.slurm ../../x >/dev/null 2>&1; echo $?)
[ "$rc" = 99 ] && ok "run.slurm invalid id -> 99" || bad "bad-id exit $rc"
no_output_created

rc=$(cd "$R" && env SJ_REPO_ROOT="$R" SLURM_JOB_ID= bash run.slurm one >/dev/null 2>&1; echo $?)
[ "$rc" = 97 ] && ok "run.slurm outside Slurm -> 97" || bad "no-slurm exit $rc"
no_output_created

rc=$(cd "$R" && env SJ_REPO_ROOT="$R" SLURM_JOB_ID=1 SLURM_JOB_NUM_NODES=2 \
        bash run.slurm one >/dev/null 2>&1; echo $?)
[ "$rc" = 97 ] && ok "run.slurm multi-node -> 97" || bad "multi-node exit $rc"
no_output_created

rc=$(cd "$R" && env SJ_REPO_ROOT="$R" SLURM_JOB_ID=1 SLURM_JOB_NUM_NODES=1 \
        SLURM_GPUS_ON_NODE=3 bash run.slurm one >/dev/null 2>&1; echo $?)
[ "$rc" = 97 ] && ok "run.slurm 3 GPUs -> 97" || bad "3-gpu exit $rc"
no_output_created

# Valid environment passes validation (gets past the overwrite gate and the
# output/ mkdir; locally it then stops at `module load`, exit 127 here).
rc=$(cd "$R" && env SJ_REPO_ROOT="$R" SLURM_JOB_ID=1 SLURM_JOB_NUM_NODES=1 \
        SLURM_GPUS_ON_NODE=1 bash run.slurm one >/dev/null 2>&1; echo $?)
case "$rc" in
    97|98|99) bad "run.slurm valid env wrongly refused (exit $rc)" ;;
    *) ok "run.slurm valid env passes validation (local stop at module load, exit $rc)" ;;
esac
assert_file "run.slurm valid env created output/logs" "$R/output/logs"

# Stale products refusal (products exist, SJ_OVERWRITE unset).
mkdir -p "$R/output/PE" && touch "$R/output/PE/one.pq"
rc=$(cd "$R" && env SJ_REPO_ROOT="$R" SLURM_JOB_ID=1 SLURM_JOB_NUM_NODES=1 \
        SLURM_GPUS_ON_NODE=1 bash run.slurm one >/dev/null 2>&1; echo $?)
[ "$rc" = 98 ] && ok "run.slurm stale products -> 98" || bad "stale exit $rc"

# ================================================================== end =====
echo
echo "control-flow regression: $PASS passed, $FAIL failed (stub sbatch/squeue only; no cluster contacted)"
[ "$FAIL" -eq 0 ]
