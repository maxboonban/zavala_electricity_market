#!/bin/bash

#SBATCH --job-name=zavala_im3
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

WORKDIR=/users/$USER/scratch/zavala_electricity_market
LOG=outputs/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

cd "$WORKDIR"
mkdir -p outputs/logs outputs/results logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Job started — ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME"

log "Loading modules..."
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

if ! conda env list | grep -q "^zavala-market "; then
    log "Conda env 'zavala-market' not found — creating from requirements.txt..."
    conda create -n zavala-market python=3.11 -y
    conda run -n zavala-market pip install -r requirements.txt
    log "Conda env created."
else
    log "Conda env 'zavala-market' already exists, skipping creation."
fi

conda activate zavala-market
log "Conda environment activated"

export PYTHONUNBUFFERED=TRUE
export GRB_LICENSE_FILE=/users/$USER/gurobi.lic

log "Preflight: Gurobi licence check"
python - <<'PYEOF'
import sys
try:
    import gurobipy as gp
    m = gp.Model(); m.setParam("OutputFlag", 0)
    x = m.addVar(ub=1); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize()
    print(f"[preflight] Gurobi OK (v{gp.gurobi.version()}), trivial LP obj = {m.objVal:.1f}")
except Exception as e:
    print(f"[preflight][FATAL] Gurobi unusable: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# --- 1. Theory errata: adjudicate every suspected mistake under Gurobi -------
log "Theory verification (docs/THEORY_ERRATA.md)"
for EPS in 0.1 0.3 0.5; do
  log "  verify_theory eps=$EPS"
  python -u verify_theory.py --solver GUROBI --scenarios 200 --eps $EPS \
    2>&1 | tee -a logs/verify_theory_gurobi.log
done

# --- 2. CVaR pricing invariants + decentralisation suite D1-D5 --------------
log "CVaR pricing invariants"
python -u verify_cvar_pricing.py --solver GUROBI --scenarios 100 \
  2>&1 | tee logs/verify_cvar_pricing_gurobi.log

log "D1-D5 decentralisation suite (S=500)"
python -u run_decentralized_experiments.py \
  --experiments D1 D2 D3 D4 D5 --scenarios 500 --eps 0.1 --beta 0.95 \
  --seed 2025 --solver GUROBI --tag "dec_S500_${SLURM_JOB_ID}"

# --- 3. IM3 GO-WEST economic experiments E1-E6 ------------------------------
SC=500
TAG="im3_S${SC}_${SLURM_JOB_ID}"

log "E1-E3,E6 at the headline configuration (summer, delivery hour 18)"
python -u run_im3_experiments.py \
  --experiments E1 E2 E3 E6 \
  --scenarios $SC --eps 0.3 --beta 0.95 --seed 2025 --solver GUROBI \
  --hour-of-day 18 --season summer \
  --shares 0.15 0.30 0.45 0.60 0.75 \
  --deltas 0.02 0.05 0.10 0.20 0.40 \
  --tag "${TAG}_main"

log "E4 out-of-sample validation (30 replications)"
python -u run_im3_experiments.py \
  --experiments E4 \
  --scenarios $SC --eps 0.3 --beta 0.95 --seed 2025 --solver GUROBI \
  --hour-of-day 18 --season summer --train 400 --test 400 --reps 30 \
  --tag "${TAG}_oos"

log "E5 paired statistics (30 replications)"
python -u run_im3_experiments.py \
  --experiments E5 \
  --scenarios 300 --beta 0.95 --seed 2025 --solver GUROBI \
  --hour-of-day 18 --season summer --reps 30 \
  --eps-list 0.1 0.3 0.5 1.0 \
  --tag "${TAG}_paired"

# --- 4. Seasonal / delivery-hour robustness ---------------------------------
for SEASON in winter summer; do
  for HOUR in 8 18; do
    log "E1 robustness: season=$SEASON hour=$HOUR"
    python -u run_im3_experiments.py \
      --experiments E1 E2 \
      --scenarios $SC --eps 0.3 --beta 0.95 --seed 2025 --solver GUROBI \
      --hour-of-day $HOUR --season $SEASON \
      --tag "${TAG}_${SEASON}_h${HOUR}"
  done
done

log "Job completed successfully."
