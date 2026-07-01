#!/bin/bash

#SBATCH --job-name=zavala_visualize_distributions
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

log "Starting visualize_distributions.py"
python -u visualize_distributions.py

log "Job completed successfully."
