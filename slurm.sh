#!/bin/bash

#SBATCH --job-name=zavala_market
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /users/$USER/scratch/zavala_electricity_market

mkdir -p outputs/logs outputs/results

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate zavala-market

export PYTHONUNBUFFERED=TRUE

python -u zavala_real_data.py \
  --data-dir dataset/IM-3-GO-WEST \
  --experiment-name baseline_stochastic