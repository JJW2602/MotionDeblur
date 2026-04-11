#!/bin/bash
#SBATCH --job-name=eval_smoke5
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch2/james2602/playground/Motionblur/results/_logs_eval_smoke5_parallel/%A_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/results/_logs_eval_smoke5_parallel/%A_%a.err

# Parallel smoke test: 5 jobs, each processes 1 GoPro test image.
# Each array task gets image_idx = SLURM_ARRAY_TASK_ID (0~4).
#
# Submit:
#   sbatch script/eval_smoke5_parallel.slurm

set -euo pipefail

CONDA_ENV="diffusion"
PROJECT_DIR="/scratch2/james2602/playground/Motionblur"
JOB_ID="${SLURM_JOB_ID:-local}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[warn] HF_TOKEN is not set. SD3.5 Large download may fail (gated repo)."
fi

mkdir -p "${PROJECT_DIR}/results/_logs_eval_smoke5_parallel"

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_DIR}"

echo "======================================================"
echo "[$(date +%F_%T)] Job ${JOB_ID} / Task ${TASK_ID} — image_idx=${TASK_ID}"
echo "======================================================"

python -m eval.restormer_sd35_eval \
  ++image_idx="${TASK_ID}" \
  eval.output_dir="results/baseline_restormer_sd35_smoke5_parallel"

echo "[DONE] Job ${JOB_ID} / Task ${TASK_ID} finished at $(date)"
