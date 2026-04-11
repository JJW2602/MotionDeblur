#!/bin/bash
#SBATCH --job-name=eval_restormer_sd35
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/james2602/playground/Motionblur/results/_logs_eval_restormer_sd35/%j.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/results/_logs_eval_restormer_sd35/%j.err

# Baseline evaluation: Restormer (Stage1) + SD3.5 Large Blur ControlNet (Stage2).
# No fine-tuning. Computes PSNR/SSIM/LPIPS on the GoPro test split.
#
# Submit:
#   sbatch script/eval_restormer_sd35.slurm
# Smoke test (first 5 images) — pass --smoke flag through env:
#   SMOKE=1 sbatch script/eval_restormer_sd35.slurm

set -euo pipefail

CONDA_ENV="diffusion"
PROJECT_DIR="/scratch2/james2602/playground/Motionblur"
JOB_ID="${SLURM_JOB_ID:-local}"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# SD3.5 Large is a gated repo on HF. Set HF_TOKEN in your env (e.g. via ~/.bashrc)
# before submitting, or pre-download the weights into ${HF_HOME}.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[warn] HF_TOKEN is not set. SD3.5 Large download may fail (gated repo)."
fi

mkdir -p "${PROJECT_DIR}/results/_logs_eval_restormer_sd35"

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_DIR}"

EXTRA_OVERRIDES=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_OVERRIDES+=("++smoke=true")
  EXTRA_OVERRIDES+=("eval.output_dir=results/baseline_restormer_sd35_smoke")
fi

echo "======================================================"
echo "[$(date +%F_%T)] Job ${JOB_ID}"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  overrides:  ${EXTRA_OVERRIDES[*]:-<none>}"
echo "======================================================"

python -m eval.restormer_sd35_eval "${EXTRA_OVERRIDES[@]}"

echo "[DONE] Job ${JOB_ID} finished at $(date)"
