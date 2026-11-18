#!/bin/bash
#
# Tail-welfare experiments W0-W6 for the risk-aware clearing paper.
#
#   sbatch slurm_tail.sh
#
# Produces, under outputs/results/ and figures/tail/:
#   - tail_<tag>_main.json      W0-W6 at the headline configuration
#   - tail_<tag>_rampext.json   same, WITH the start-up constraint (extension)
#   - tail_<tag>_beta99.json    robustness at beta = 0.99
#   - six PDF/PNG figure pairs
# and pushes all of it back to the hrithik_final branch.

#SBATCH --job-name=zavala_tail
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --time=14:00:00
#SBATCH --partition=batch
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

set -euo pipefail

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"
mkdir -p outputs/logs outputs/results logs figures/tail

LOG=outputs/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}, workdir ${WORKDIR}"

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

if ! conda env list | grep -q "^zavala-market "; then
    log "Creating conda env (NOTE: compute nodes have no internet; if this"
    log "fails, create the env on the login node first)."
    conda create -n zavala-market python=3.11 -y
    conda run -n zavala-market pip install -r requirements.txt
fi
conda activate zavala-market

export PYTHONUNBUFFERED=TRUE
export GRB_LICENSE_FILE=$HOME/gurobi.lic
# Gurobi threads are set per solve; keep BLAS from oversubscribing the node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

log "Preflight: Gurobi licence"
python - <<'PYEOF'
import sys
try:
    import gurobipy as gp
    m = gp.Model(); m.setParam("OutputFlag", 0)
    x = m.addVar(ub=1); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize()
    print(f"[preflight] Gurobi OK (v{gp.gurobi.version()}), obj = {m.objVal:.1f}")
except Exception as e:
    print(f"[preflight][FATAL] Gurobi unusable: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

TAG="tail_${SLURM_JOB_ID}"
S=500
SEEDS=30
SWEEP_SEEDS=15
BETA=0.95
EPS=0.3

# ---------------------------------------------------------------------------
# 1. Headline run. --check-reference asserts the vectorised model in
#    tail_model.py reproduces the verified zavala_cvar before anything is
#    reported from it.
# ---------------------------------------------------------------------------
log "W0-W7 headline (S=${S}, ${SEEDS} seeds, eps=${EPS}, beta=${BETA})"
python -u run_tail_experiments.py \
  --experiments W0 W1 W2 W3 W4 W5 W6 W7 \
  --scenarios $S --seeds $SEEDS --sweep-seeds $SWEEP_SEEDS \
  --eps $EPS --beta $BETA --solver GUROBI \
  --ramp-premium 1.0 --backdown-discount 0.40 --adequacy 1.12 --voll 1000 \
  --asymmetry-levels 0.0 0.25 0.5 1.0 2.0 4.0 \
  --adequacy-levels 0.95 1.00 1.05 1.12 1.20 1.30 \
  --voll-levels 200 500 1000 2000 5000 \
  --eps-levels 0.0 0.1 0.2 0.3 0.5 0.7 0.9 \
  --stress-prob-levels 0.02 0.06 0.12 0.22 0.35 \
  --stress-depth-levels -0.5 -1.2 -1.9 -2.8 -3.6 \
  --correlation-levels 0.1 0.4 0.7 0.9 0.98 \
  --outage-levels 0.0 0.1 0.3 0.5 0.7 \
  --check-reference \
  --tag "${TAG}_main"

# ---------------------------------------------------------------------------
# 2. Start-up (ramp-limit) extension. This adds G_i(w) - g_i <= ramp_cap_i,
#    which is NOT in the paper's Eq. (63)-(65). It is the only channel through
#    which the day-ahead position can change unserved energy, so it is run and
#    reported separately rather than mixed into the headline.
# ---------------------------------------------------------------------------
log "W0-W6 with the start-up constraint (EXTENSION, reported separately)"
python -u run_tail_experiments.py \
  --experiments W0 W1 W2 W5 \
  --scenarios $S --seeds $SEEDS --sweep-seeds $SWEEP_SEEDS \
  --eps $EPS --beta $BETA --solver GUROBI \
  --ramp-premium 1.0 --backdown-discount 0.40 --adequacy 1.12 --voll 1000 \
  --ramp-limit-frac 0.03 \
  --asymmetry-levels 0.0 0.5 1.0 2.0 4.0 \
  --tag "${TAG}_rampext"

# ---------------------------------------------------------------------------
# 3. Tail-level robustness: the whole story should not hinge on beta = 0.95.
# ---------------------------------------------------------------------------
log "Robustness at beta = 0.99"
python -u run_tail_experiments.py \
  --experiments W0 W1 W2 \
  --scenarios $S --seeds $SWEEP_SEEDS --sweep-seeds $SWEEP_SEEDS \
  --eps $EPS --beta 0.99 --solver GUROBI \
  --ramp-premium 1.0 --backdown-discount 0.40 --adequacy 1.12 --voll 1000 \
  --asymmetry-levels 0.0 0.5 1.0 2.0 4.0 \
  --tag "${TAG}_beta99"

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
log "Rendering figures"
python -u plot_tail_figures.py \
  --result "outputs/results/tail_${TAG}_main.json" \
  --outdir figures/tail
python -u plot_tail_figures.py \
  --result "outputs/results/tail_${TAG}_rampext.json" \
  --outdir figures/tail_rampext

# ---------------------------------------------------------------------------
# 5. Commit results. Compute nodes have no outbound network, so `git push`
#    CANNOT work here; ./push_results.sh does that from a login node.
# ---------------------------------------------------------------------------
log "Committing results (push must happen on a login node)"
git config user.email "hrithik_nambiar@brown.edu"
git config user.name "Hrithik Nambiar"
git add outputs/results/ outputs/logs/ logs/ figures/
git commit -m "OSCAR tail-welfare results: job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}

W0-W7 at S=${S}, ${SEEDS} paired seeds, eps=${EPS}, beta=${BETA}; plus the
start-up-constraint extension and a beta=0.99 robustness run. Figures included." \
  || log "[WARN] nothing new to commit"

log ""
log "  ############################################################"
log "  #  RESULTS ARE COMMITTED BUT NOT PUSHED.                   #"
log "  #  From a LOGIN node, run:                                 #"
log "  #      cd ${WORKDIR}"
log "  #      ./push_results.sh                                   #"
log "  ############################################################"

log "Done."
