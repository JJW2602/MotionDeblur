#!/bin/bash
#SBATCH --job-name=eval_smoke5
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch2/james2602/playground/Motionblur/results/_logs_eval_smoke5/%j.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/results/_logs_eval_smoke5/%j.err

# Smoke test: Restormer + SD3.5 Blur ControlNet on 5 GoPro test images.
#
# Submit:
#   sbatch script/eval_smoke5.slurm

set -euo pipefail

CONDA_ENV="diffusion"
PROJECT_DIR="/scratch2/james2602/playground/Motionblur"
JOB_ID="${SLURM_JOB_ID:-local}"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[warn] HF_TOKEN is not set. SD3.5 Large download may fail (gated repo)."
fi

mkdir -p "${PROJECT_DIR}/results/_logs_eval_smoke5"

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_DIR}"

echo "======================================================"
echo "[$(date +%F_%T)] Smoke test (5 images) — Job ${JOB_ID}"
echo "======================================================"

python -m eval.restormer_sd35_eval \
  ++smoke=true \
  eval.output_dir=results/baseline_restormer_sd35_smoke5

echo "[DONE] Job ${JOB_ID} finished at $(date)"
