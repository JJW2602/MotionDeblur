#!/bin/bash
# OURS parameter tuning — 1-D sweeps around a default pivot.
#
# Default: controlnet_scale=0.5, guidance_scale=6.0, steps=40
# Sweep one axis at a time, fixing the other two at default:
#   Jobs  0-4:  controlnet_scale ∈ {0.3, 0.4, 0.5, 0.6, 0.7}  (gs=6.0, steps=40)
#   Jobs  5-9:  guidance_scale   ∈ {2.0, 4.0, 6.0, 8.0, 10.0}  (cs=0.5, steps=40)
#   Jobs 10-14: steps            ∈ {10, 20, 30, 40, 50}         (cs=0.5, gs=6.0)
#
# Total: 15 jobs.  Each job runs inference (single image) THEN test (full test split).
#
# Results:  OURS/result/param_tuning/cs<cs>_g<gs>_n<steps>/{stage1,stage2}/...
#
# Usage:
#   mkdir -p /scratch2/james2602/playground/Motionblur/OURS/result/param_tuning/_logs
#   sbatch script/tuning_inference_test.slurm

#SBATCH --job-name=ours_sweep
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node18,node04,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-14%10
#SBATCH --output=/scratch2/james2602/playground/Motionblur/OURS/result/param_tuning/_logs/%A/task_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/OURS/result/param_tuning/_logs/%A/task_%a.err

set -euo pipefail

# ========================= 사용자 설정 =========================
CONDA_ENV="diffusion"
WORKSPACE_DIR="/scratch2/james2602/playground/Motionblur/OURS"
EXP_NAME="param_tuning"
INPUT_IMAGE="/scratch2/james2602/playground/Motionblur/dataset/gopro/blur/images/000007.png"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

# ===================== Default pivot =====================
DEFAULT_CS=0.5
DEFAULT_GS=6.0
DEFAULT_STEPS=40

# ===================== Sweep values =====================
CS_VALUES=(0.3 0.4 0.5 0.6 0.7)
GS_VALUES=(2.0 4.0 6.0 8.0 10.0)
STEPS_VALUES=(10 20 30 40 50)

# ===================== Map TASK_ID → (cs, gs, steps) =====================
if (( TASK_ID >= 0 && TASK_ID <= 4 )); then
    # Sweep controlnet_scale, fix gs & steps
    cs="${CS_VALUES[$TASK_ID]}"
    gs="${DEFAULT_GS}"
    steps="${DEFAULT_STEPS}"
elif (( TASK_ID >= 5 && TASK_ID <= 9 )); then
    # Sweep guidance_scale, fix cs & steps
    cs="${DEFAULT_CS}"
    gs="${GS_VALUES[$((TASK_ID - 5))]}"
    steps="${DEFAULT_STEPS}"
elif (( TASK_ID >= 10 && TASK_ID <= 14 )); then
    # Sweep steps, fix cs & gs
    cs="${DEFAULT_CS}"
    gs="${DEFAULT_GS}"
    steps="${STEPS_VALUES[$((TASK_ID - 10))]}"
else
    echo "[SKIP] TASK_ID=${TASK_ID} out of range. Exiting cleanly."
    exit 0
fi

RUN_NAME="${EXP_NAME}/cs${cs}_g${gs}_n${steps}"

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

echo "============================================"
echo "[LAUNCH] job=${JOB_ID} task=${TASK_ID}"
echo "[LAUNCH] controlnet_scale=${cs}  guidance_scale=${gs}  steps=${steps}"
echo "[LAUNCH] name=${RUN_NAME}"
echo "============================================"

# ===== 1. Inference (single image) =====
echo "[RUN] mode=inference"
python run.py \
    mode=inference \
    name="${RUN_NAME}" \
    input="${INPUT_IMAGE}" \
    stage2.controlnet_scale="${cs}" \
    stage2.guidance_scale="${gs}" \
    stage2.steps="${steps}" \
    use_wandb=false

# ===== 2. Test (full test split) =====
echo "[RUN] mode=test"
python run.py \
    mode=test \
    name="${RUN_NAME}" \
    stage2.controlnet_scale="${cs}" \
    stage2.guidance_scale="${gs}" \
    stage2.steps="${steps}" \
    test.num_images=20 \
    use_wandb=false

echo "[DONE] task=${TASK_ID}  name=${RUN_NAME}"
