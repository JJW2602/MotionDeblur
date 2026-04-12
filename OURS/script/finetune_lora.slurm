#!/bin/bash
# OURS LoRA fine-tuning — Top 5 configurations sweep.
#
# Each config varies (rank, target_modules, learning_rate) to find
# the sweet spot for 900 training images.
#
# ┌────┬──────┬───────┬─────────────────────────┬────────┬───────────┬──────────┐
# │ ID │ rank │ alpha │ target_modules          │ lr     │ ~LoRA     │ par/img  │
# ├────┼──────┼───────┼─────────────────────────┼────────┼───────────┼──────────┤
# │  0 │  2   │  2    │ to_q, to_v              │ 2e-4   │   166K    │   184    │
# │  1 │  4   │  4    │ to_q, to_v              │ 1e-4   │   332K    │   369    │
# │  2 │  4   │  4    │ to_q, to_k, to_v, to_out│ 1e-4   │   663K    │   737    │
# │  3 │  8   │  8    │ to_q, to_v              │ 5e-5   │   663K    │   737    │
# │  4 │  8   │  8    │ to_q, to_k, to_v, to_out│ 5e-5   │  1,327K   │  1,475   │
# └────┴──────┴───────┴─────────────────────────┴────────┴───────────┴──────────┘
#
# Configs 2 & 3 have same param count but different structure
# (more targets vs deeper rank) — direct comparison.
#
# wandb project: OURS_LoRA_Sweep (online mode, all 5 runs comparable)
#
# Usage:
#   mkdir -p /scratch2/james2602/playground/Motionblur/OURS/result/lora_sweep/_logs
#   sbatch script/finetune_lora.slurm

#SBATCH --job-name=ours_lora
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node18,node04,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch2/james2602/playground/Motionblur/OURS/result/lora_sweep/_logs/%A/task_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/OURS/result/lora_sweep/_logs/%A/task_%a.err

set -euo pipefail

# ========================= 사용자 설정 =========================
CONDA_ENV="diffusion"
WORKSPACE_DIR="/scratch2/james2602/playground/Motionblur/OURS"
EXP_NAME="lora_sweep"
MAX_STEPS=50000
SAVE_EVERY=5000
VAL_EVERY=1000

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

# ===================== Config definitions =====================
case ${TASK_ID} in
    0)
        RANK=2;  ALPHA=2;  LR="2e-4"; CP_LR="1e-3"
        TGT_LABEL="qv"
        TGT_HYDRA="[to_q,to_v]"
        ;;
    1)
        RANK=4;  ALPHA=4;  LR="1e-4"; CP_LR="1e-3"
        TGT_LABEL="qv"
        TGT_HYDRA="[to_q,to_v]"
        ;;
    2)
        RANK=4;  ALPHA=4;  LR="1e-4"; CP_LR="1e-3"
        TGT_LABEL="qkvo"
        TGT_HYDRA="[to_q,to_k,to_v,to_out.0]"
        ;;
    3)
        RANK=8;  ALPHA=8;  LR="5e-5"; CP_LR="5e-4"
        TGT_LABEL="qv"
        TGT_HYDRA="[to_q,to_v]"
        ;;
    4)
        RANK=8;  ALPHA=8;  LR="5e-5"; CP_LR="5e-4"
        TGT_LABEL="qkvo"
        TGT_HYDRA="[to_q,to_k,to_v,to_out.0]"
        ;;
    *)
        echo "[SKIP] TASK_ID=${TASK_ID} out of range. Exiting."
        exit 0
        ;;
esac

RUN_NAME="${EXP_NAME}/r${RANK}_${TGT_LABEL}_lr${LR}"

# ===== Runtime env =====
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

cd "${WORKSPACE_DIR}"

echo "============================================================"
echo "[CONFIG] job=${JOB_ID} task=${TASK_ID}"
echo "[CONFIG] rank=${RANK} alpha=${ALPHA} targets=${TGT_LABEL}"
echo "[CONFIG] lr=${LR} concat_proj_lr=${CP_LR}"
echo "[CONFIG] max_steps=${MAX_STEPS} val_every=${VAL_EVERY}"
echo "[CONFIG] name=${RUN_NAME}"
echo "============================================================"

python run.py \
    mode=train \
    name="${RUN_NAME}" \
    use_wandb=true \
    wandb_project=OURS_LoRA_Sweep \
    train.rank="${RANK}" \
    train.alpha="${ALPHA}" \
    train.learning_rate="${LR}" \
    train.concat_proj_lr="${CP_LR}" \
    "train.target_modules=${TGT_HYDRA}" \
    train.max_train_steps="${MAX_STEPS}" \
    train.save_every="${SAVE_EVERY}" \
    train.val_every="${VAL_EVERY}" \
    train.precompute_stage1=true \
    train.gradient_checkpointing=true

echo "[DONE] task=${TASK_ID}  name=${RUN_NAME}"
