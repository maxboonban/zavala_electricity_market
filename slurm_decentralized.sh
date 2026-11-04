#!/bin/bash

#SBATCH --job-name=zavala_decentralized
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

WORKDIR=/users/$USER/scratch/zavala_electricity_market
LOG=outputs/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

cd "$WORKDIR"
mkdir -p outputs/logs outputs/results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Job started — ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME"
log "Working directory: $WORKDIR"

log "Loading modules..."
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

# Create conda env from requirements.txt if it doesn't exist yet
if ! conda env list | grep -q "^zavala-market "; then
    log "Conda env 'zavala-market' not found — creating from requirements.txt..."
    conda create -n zavala-market python=3.11 -y
    conda run -n zavala-market pip install -r requirements.txt
    log "Conda env created."
else
    log "Conda env 'zavala-market' already exists, skipping creation."
fi

conda activate zavala-market
log "Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"

export PYTHONUNBUFFERED=TRUE
export GRB_LICENSE_FILE=/users/$USER/gurobi.lic

# --- Experiment parameters ---
SCENARIOS=500
BETA=0.95
SEED=2025
TAG="dec_S${SCENARIOS}_b${BETA}_${SLURM_JOB_ID}"

log "Sanity: CVaR pricing invariants under Gurobi"
python -u verify_cvar_pricing.py --solver GUROBI --scenarios 40 \
  2>&1 | tee logs/verify_cvar_pricing_gurobi.log

log "D1-D5 at eps=0.1 (headline configuration)"
python -u run_decentralized_experiments.py \
  --experiments D1 D2 D3 D4 D5 \
  --scenarios $SCENARIOS --eps 0.1 --beta $BETA --seed $SEED \
  --solver GUROBI --tag "${TAG}_eps0.1"

log "Risk-weight sweep: D1 and D3 at each eps (Appendix B configuration)"
for EPS in 0.3 0.5 1.0; do
  log "  eps=$EPS"
  python -u run_decentralized_experiments.py \
    --experiments D1 D3 \
    --scenarios $SCENARIOS --eps $EPS --beta $BETA --seed $SEED \
    --solver GUROBI --tag "${TAG}_eps${EPS}"
done

log "Job completed successfully."
