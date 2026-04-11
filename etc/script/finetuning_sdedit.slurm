#!/bin/bash
#SBATCH --job-name=sdedit_finetune
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch2/james2602/playground/Motionblur/Motionblur/results/_logs_finetune_sdedit/%A/%j_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/Motionblur/results/_logs_finetune_sdedit/%A/%j_%a.err

set -euo pipefail

# ========================= 사용자 설정 =========================
CONDA_ENV="diffusion"
PROJECT_DIR="/scratch2/james2602/playground/Motionblur/Motionblur"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_JOB_ID}"

# ENV
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE="online"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_SERVICE_FORCE_INTEL=1

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_DIR}/results/_logs_finetune_sdedit/${JOB_ID}"

# ========================= 3가지 Fine-tuning 방법 =========================
#
# 방법 0: baseline_r8_lr1e4
#   - 기본 설정 (rank 8, lr 1e-4) 그대로, 개선된 logging으로 학습 추이 확인
#
# 방법 1: r32_lr1e4
#   - rank 32로 LoRA capacity 증가, 나머지 동일
#
# 방법 2: r16_lr5e5_textenc
#   - rank 16 + text encoder도 함께 fine-tuning
#   - lr 절반으로 안정적 학습

case ${TASK_ID} in
  0)
    METHOD_NAME="baseline_r8_lr1e4"
    RANK=8
    ALPHA=8
    LR="1.0e-4"
    TRAIN_TEXT_ENC=false
    TEXT_ENC_RANK=4
    TEXT_ENC_ALPHA=4
    MAX_STEPS=10000
    ;;
  1)
    METHOD_NAME="r32_lr1e4"
    RANK=32
    ALPHA=32
    LR="1.0e-4"
    TRAIN_TEXT_ENC=false
    TEXT_ENC_RANK=4
    TEXT_ENC_ALPHA=4
    MAX_STEPS=10000
    ;;
  2)
    METHOD_NAME="r16_lr5e5_textenc"
    RANK=16
    ALPHA=16
    LR="5.0e-5"
    TRAIN_TEXT_ENC=true
    TEXT_ENC_RANK=8
    TEXT_ENC_ALPHA=8
    MAX_STEPS=10000
    ;;
esac

OUTPUT_DIR="${PROJECT_DIR}/results/lora/sdedit_${METHOD_NAME}"
WEIGHT_NAME="sd3_lora_${METHOD_NAME}.safetensors"

echo "======================================================"
echo "[${TS}] Task ${TASK_ID}: ${METHOD_NAME}"
echo "  rank=${RANK} alpha=${ALPHA} lr=${LR}"
echo "  train_text_enc=${TRAIN_TEXT_ENC} max_steps=${MAX_STEPS}"
echo "  output=${OUTPUT_DIR}/${WEIGHT_NAME}"
echo "======================================================"

python run.py \
  mode=train \
  method=sdedit \
  use_wandb=true \
  wandb_entity="MotionBLur" \
  wandb_project="Finetuning_LoRA_${METHOD_NAME}" \
  train.tuning_architecture=SD3 \
  train.weighting_scheme=none \
  train.rank="${RANK}" \
  train.alpha="${ALPHA}" \
  train.learning_rate="${LR}" \
  train.weight_decay=0.0 \
  train.max_train_steps="${MAX_STEPS}" \
  train.max_grad_norm=1.0 \
  train.gradient_checkpointing=true \
  train.train_text_encoders="${TRAIN_TEXT_ENC}" \
  train.text_encoder_rank="${TEXT_ENC_RANK}" \
  train.text_encoder_alpha="${TEXT_ENC_ALPHA}" \
  train.dropout=0.0 \
  train.log_every=50 \
  train.output_dir="${OUTPUT_DIR}" \
  train.weight_name="${WEIGHT_NAME}" \
  hydra.run.dir="${OUTPUT_DIR}"

echo "[DONE] Task ${TASK_ID}: ${METHOD_NAME} finished at $(date)"
