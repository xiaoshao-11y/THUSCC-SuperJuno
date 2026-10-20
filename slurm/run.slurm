#!/bin/bash
#SBATCH --job-name=superjuno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/logs/%x-%j.out
#SBATCH --error=output/logs/%x-%j.err
#
# Official three-stage run with correctness comparison.
# Copy this file to the repository root (~/run/superjuno/) next to scripts/.
#
# Submit with (override --gres to use up to 2 GPUs):
#   sbatch --partition=$GPU_PARTITION run.slurm one          # validation case
#   sbatch --partition=$GPU_PARTITION run.slurm 0            # scored cases
# or via slurm/submit.sh, which archives previous artifacts first.
#
# Environment:
#   SJ_REPO_ROOT   repository root (default $HOME/run/superjuno)
#   SJ_OVERWRITE   set to 1 to allow running when output files already
#                  exist; default is to fail instead of overwriting, so a
#                  double submission of the same <id> cannot clobber a
#                  finished run. submit.sh archives before submitting, so
#                  normal flow never needs this.
# Opt-in optimization switches propagated from the submitting shell
# (echoed at the top of the job log by the line below; submit.sh records
# the first two in output/logs/submissions.tsv):
#   SUPERJUNO_SJRNG=1      C++ bit-exact FSMP random preparation
#   SUPERJUNO_BETA_INV=1   upstream beta_inversion kernel instead of
#                          cp.linalg.inv in vmove1 (needs --compare PASS
#                          before being trusted)
#   SJ_BLOCKS=<n>          reconstruction event-block count (run.py -n;
#                          officially adjustable, reference value 10)
#   SUPERJUNO_VRAM_DIV=<d> fsmp GPU batch sizing divisor on total VRAM
#                          (upstream heuristic: /5; measured: /2 regressed
#                          badly on one — memory-bound)
#   SJ_SHARDS=<n>          stage-2 part sharding across n GPUs (round-
#                          robin parts, parquet merge; needs SJ_GPUS=n)

set -eo pipefail

# --- validation: the case id is checked BEFORE any output path is created
# or even constructed, so `../../x`-style arguments can never enter a path.
# Exit codes: 99 = bad usage/id, 97 = environment refusal (not under Slurm,
# multi-node, or >2 GPUs), 98 = overwrite refusal.
die() { echo "run.slurm: $*" >&2; exit "${2:-99}"; }

[ "$#" -eq 1 ] || die "exactly one case id required: sbatch ... run.slurm <one|0|1|2>"
case "$1" in
    one|0|1|2) input_id=$1 ;;
    *) die "invalid case id '$1' (one|0|1|2)" ;;
esac
[ -n "${SLURM_JOB_ID:-}" ] || die "SLURM_JOB_ID unset: refusing to run outside Slurm (no compute on login nodes)" 97
nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
[ "$nodes" = "1" ] || die "single-node jobs only (SLURM_JOB_NUM_NODES=$nodes)" 97
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    case "$SLURM_GPUS_ON_NODE" in
        1|2) ;;
        *) die "the rules allow at most 2 GPUs per run (SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE)" 97 ;;
    esac
fi

SJ_REPO_ROOT="${SJ_REPO_ROOT:-$HOME/run/superjuno}"
cd "$SJ_REPO_ROOT"
[ -f scripts/run.py ] || die "scripts/run.py not found under $SJ_REPO_ROOT" 97

echo "Job ID: $SLURM_JOB_ID, Node: $(hostname), Start Time: $(date -Is)"
echo "Repo: $SJ_REPO_ROOT, Case: $input_id"
echo "Alloc: nodes=$nodes gpus_on_node=${SLURM_GPUS_ON_NODE:-<unset>}"
trap 'status=$?; echo "End Time: $(date -Is), Exit code: $status"; exit "$status"' EXIT

# Overwrite protection: submit.sh archives old artifacts, so leftover files
# mean a double submission or manual residue — refuse instead of clobbering.
for f in "output/sparsify/${input_id}.h5" "output/PE/${input_id}.pq" \
         "output/tvE/${input_id}.h5"; do
    if [ -e "$f" ] && [ "${SJ_OVERWRITE:-0}" != "1" ]; then
        echo "run.slurm: REFUSED: $f already exists (double submission?). Archive it" \
             "first or set SJ_OVERWRITE=1." >&2
        exit 98
    fi
done

# Slurm opens the %x-%j.out/.err files before the job starts, so output/logs
# must exist at submission time; submit.sh also mkdirs it. Keep it here for
# manual sbatch invocations of archived repos.
mkdir -p output/logs output/sparsify output/PE output/tvE output/tmp

# sbatch batch shells are non-login: Environment Modules is not initialized
# there (observed on BSCC-N16R4 ln01/g-nodes: "module: command not found"),
# so source the modules init before the first module load.
if ! command -v module >/dev/null 2>&1; then
    . /etc/profile.d/modules.sh 2>/dev/null \
        || . /usr/share/modules/init/bash 2>/dev/null || true
fi
module load miniforge3/24.11
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate juno

# Provenance dump (before the timed command; run.py does its own timing).
echo "Python: $(python -c 'import sys; print(sys.version.split()[0])')"
python - <<'PY'
import cupy, numpy, scipy, h5py
print("cupy", cupy.__version__, "CUDA", cupy.cuda.runtime.runtimeGetVersion())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "h5py", h5py.__version__)
print("GPU:", cupy.cuda.runtime.getDeviceProperties(0)["name"])
PY
echo "Opt-in switches: SUPERJUNO_SJRNG=${SUPERJUNO_SJRNG:-0} SUPERJUNO_BETA_INV=${SUPERJUNO_BETA_INV:-0} SJ_BLOCKS=${SJ_BLOCKS:-10} SUPERJUNO_VRAM_DIV=${SUPERJUNO_VRAM_DIV:-5} SJ_SHARDS=${SJ_SHARDS:-1} SJ_RECON_SHARDS=${SJ_RECON_SHARDS:-1}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python scripts/run.py "$input_id" --compare
