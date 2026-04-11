#!/bin/bash
# SDEdit parameter tuning sweep — 4-D grid over (strength × guidance × steps × input_images).
#
# Each SLURM array task runs ONE combo via `python run.py mode=inference ...`.
# Results land under:
#   SDEdit/result/param_tuning/s<strength>_g<guidance>_n<steps>/inference/<stem>_grid.png
#
# Usage:
#   mkdir -p /scratch2/james2602/playground/Motionblur/SDEdit/result/param_tuning/_logs
#   sbatch parameter_tuning.slurm
#
# NOTE: If you change the STRENGTH / GUIDANCE_SCALE / STEPS / INPUT_IMAGES lists
#       below, UPDATE the `#SBATCH --array=` line so its upper bound ≥
#       (NUM_S * NUM_G * NUM_T * NUM_I - 1). Out-of-range tasks exit cleanly.

#SBATCH --job-name=sdedit_sweep
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node18,node04,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-625%50
#SBATCH --output=/scratch2/james2602/playground/Motionblur/SDEdit/result/param_tuning/_logs/%A/task_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/SDEdit/result/param_tuning/_logs/%A/task_%a.err

set -euo pipefail

# ========================= 사용자 설정 =========================
CONDA_ENV="diffusion"
WORKSPACE_DIR="/scratch2/james2602/playground/Motionblur/SDEdit"
INPUT_DIR="/scratch2/james2602/playground/Motionblur/dataset/gopro/blur/images"
EXP_NAME="param_tuning"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

# ===================== 스윕 하이퍼파라미터 =====================
STRENGTH=(0.3 0.4 0.5 0.6 0.7)
GUIDANCE_SCALE=(2.0 4.0 6.0 8.0 10.0)
STEPS=(10 20 30 40 50)
INPUT_IMAGES=(
  "${INPUT_DIR}/000007.png"
  "${INPUT_DIR}/000028.png"
  "${INPUT_DIR}/000031.png"
  "${INPUT_DIR}/000033.png"
  "${INPUT_DIR}/000045.png"
)

NUM_S=${#STRENGTH[@]}
NUM_G=${#GUIDANCE_SCALE[@]}
NUM_T=${#STEPS[@]}
NUM_I=${#INPUT_IMAGES[@]}
TOTAL_COMB=$(( NUM_S * NUM_G * NUM_T * NUM_I ))

echo "[SWEEP] dims: S=${NUM_S} G=${NUM_G} T=${NUM_T} I=${NUM_I}  total=${TOTAL_COMB}"

# ===== Runtime env =====
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export WANDB_MODE="offline"

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

cd "${WORKSPACE_DIR}"

# -------------------------------------------------------------------------
# TASK_ID → (strength, guidance, steps, image)
# Mapping: strength outermost, image innermost
# -------------------------------------------------------------------------
if (( TASK_ID < 0 || TASK_ID >= TOTAL_COMB )); then
  echo "[SKIP] TASK_ID=${TASK_ID} out of range (total=${TOTAL_COMB}). Exiting cleanly."
  exit 0
fi

rem=${TASK_ID}
SIDX=$(( rem / (NUM_G * NUM_T * NUM_I) )); rem=$(( rem % (NUM_G * NUM_T * NUM_I) ))
GIDX=$(( rem / (NUM_T * NUM_I) ));         rem=$(( rem % (NUM_T * NUM_I) ))
TIDX=$(( rem / NUM_I ))
IIDX=$(( rem % NUM_I ))

strength="${STRENGTH[$SIDX]}"
guidance="${GUIDANCE_SCALE[$GIDX]}"
steps="${STEPS[$TIDX]}"
INPUT_IMAGE="${INPUT_IMAGES[$IIDX]}"
INPUT_NAME="$(basename "${INPUT_IMAGE}" .png)"

# cfg.name uses a nested path so each combo gets its own result dir:
#   result/param_tuning/s<strength>_g<guidance>_n<steps>/inference/<stem>_grid.png
RUN_NAME="${EXP_NAME}/s${strength}_g${guidance}_n${steps}"

echo "[LAUNCH] job=${JOB_ID} task=${TASK_ID}"
echo "[LAUNCH] strength=${strength}  guidance=${guidance}  steps=${steps}  image=${INPUT_NAME}"
echo "[LAUNCH] name=${RUN_NAME}"

python run.py \
  mode=inference \
  name="${RUN_NAME}" \
  input="${INPUT_IMAGE}" \
  strength="${strength}" \
  guidance_scale="${guidance}" \
  steps="${steps}" \
  use_wandb=false

echo "[DONE] task=${TASK_ID}"
