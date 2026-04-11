#!/bin/bash
# SDEdit test-split sweep — 3-D grid over (strength × guidance × steps).
#
# Each SLURM array task runs ONE combo via `python run.py mode=test ...`,
# which iterates the full GoPro test split (100 images) for that single
# (strength, guidance, steps) setting and reports PSNR/SSIM/LPIPS aggregates.
#
# Per-combo results land under:
#   SDEdit/result/param_tuning_test/s<strength>_g<guidance>_n<steps>/test/
#     ├── sweep_summary.csv             (1 row — this combo's averaged metrics)
#     ├── sweep_config.json
#     └── s<str>_g<gui>_n<stp>/         (internal sweep dir, one combo)
#          ├── per_image.csv            (100 rows)
#          └── summary.json
#
# Usage:
#   mkdir -p /scratch2/james2602/playground/Motionblur/SDEdit/result/param_tuning_test/_logs
#   sbatch parameter_tuning_test.slurm
#
# NOTE: If you change the STRENGTH / GUIDANCE_SCALE / STEPS lists below, UPDATE
#       `#SBATCH --array=` so its upper bound ≥ (NUM_S * NUM_G * NUM_T - 1).

#SBATCH --job-name=sdedit_test_sweep
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --exclude=node18,node04,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-125%50
#SBATCH --output=/scratch2/james2602/playground/Motionblur/SDEdit/result/param_tuning/_logs/%A/task_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/SDEdit/result/param_tuning/_logs/%A/task_%a.err

set -euo pipefail

# ========================= 사용자 설정 =========================
CONDA_ENV="diffusion"
WORKSPACE_DIR="/scratch2/james2602/playground/Motionblur/SDEdit"
EXP_NAME="param_tuning"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

# ===================== 스윕 하이퍼파라미터 =====================
STRENGTH=(0.3 0.4 0.5 0.6 0.7)
GUIDANCE_SCALE=(2.0 4.0 6.0 8.0 10.0)
STEPS=(10 20 30 40 50)

NUM_S=${#STRENGTH[@]}
NUM_G=${#GUIDANCE_SCALE[@]}
NUM_T=${#STEPS[@]}
TOTAL_COMB=$(( NUM_S * NUM_G * NUM_T ))

echo "[SWEEP] dims: S=${NUM_S} G=${NUM_G} T=${NUM_T}  total=${TOTAL_COMB}"

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
# TASK_ID → (strength, guidance, steps)
# Mapping: strength outermost, steps innermost
# -------------------------------------------------------------------------
if (( TASK_ID < 0 || TASK_ID >= TOTAL_COMB )); then
  echo "[SKIP] TASK_ID=${TASK_ID} out of range (total=${TOTAL_COMB}). Exiting cleanly."
  exit 0
fi

rem=${TASK_ID}
SIDX=$(( rem / (NUM_G * NUM_T) )); rem=$(( rem % (NUM_G * NUM_T) ))
GIDX=$(( rem / NUM_T ))
TIDX=$(( rem % NUM_T ))

strength="${STRENGTH[$SIDX]}"
guidance="${GUIDANCE_SCALE[$GIDX]}"
steps="${STEPS[$TIDX]}"

RUN_NAME="${EXP_NAME}/s${strength}_g${guidance}_n${steps}"

echo "[LAUNCH] job=${JOB_ID} task=${TASK_ID}"
echo "[LAUNCH] strength=${strength}  guidance=${guidance}  steps=${steps}"
echo "[LAUNCH] name=${RUN_NAME}"

# Pass a 1-element list for each sweep dimension → run.py's internal sweep
# degenerates to a single combo run over the full test split.
python run.py \
  mode=test \
  name="${RUN_NAME}" \
  "test.strengths=[${strength}]" \
  "test.guidance_scales=[${guidance}]" \
  "test.steps_list=[${steps}]" \
  use_wandb=false

echo "[DONE] task=${TASK_ID}"
