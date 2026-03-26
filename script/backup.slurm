#!/bin/bash
#SBATCH --job-name=pretrain_rest
#SBATCH --partition=big_suma_rtx3090
#SBATCH --exclude=node19
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-99%30
#SBATCH --output=/scratch2/james2602/playground/Motionblur/Motionblur/results/_logs_baseline_pretrain/%A/%j_%a.out
#SBATCH --error=/scratch2/james2602/playground/Motionblur/Motionblur/results/_logs_baseline_pretrain/%A/%j_%a.err

set -euo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"

# ========================= 사용자 설정 =========================

TASKS_PER_ARRAY=3     # 한 GPU에서 동시에 3개
CONDA_ENV="diffusion"
PROJECT_DIR="/scratch2/james2602/playground/Motionblur/Motionblur"
TASK_ID="${SLURM_ARRAY_TASK_ID}"     
JOB_ID="${SLURM_JOB_ID}"

# ===================== 스윕 하이퍼파라미터 =====================

GUIDANCE_SCALE=(6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0)
STRENGTH=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
STEP=(10 20 30 40 50)

NUM_G=${#GUIDANCE_SCALE[@]}
NUM_S=${#STRENGTH[@]}
NUM_T=${#STEP[@]}

TOTAL_COMB=$(( NUM_G * NUM_S * NUM_T ))
ARRAY_MAX=$(( TOTAL_COMB - 1 ))

# ENV
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export WANDB_MODE="offline"
export WANDB_CACHE_DIR=/scratch2/$USER/wandb-cache/mimdice/baseline_pretrain
export WANDB_DIR=/scratch2/$USER/wandb-runs/mimdice/baseline_pretrain
export WANDB_DATA_DIR=/scratch2/$USER/wandb-staging/mimdice/baseline_pretrain
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR"
echo "[W&B] cache=$WANDB_CACHE_DIR dir=$WANDB_DIR data=$WANDB_DATA_DIR"

set -eo pipefail

# ENV
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MUJOCO_GL=egl
export MKL_SERVICE_FORCE_INTEL=1
export WANDB_START_METHOD=${WANDB_START_METHOD}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED}

# 리소스/공유 동작 옵션
ENABLE_MPS="${ENABLE_MPS:-1}"         # 1이면 NVIDIA MPS 사용 시도(권장)
CPUS_PER_RUN="${CPUS_PER_RUN:-3}"     # 각 프로세스당 CPU 스레드 수 (4/2=2)
GPU_SHARE="${GPU_SHARE:-100}"          # MPS에서 각 프로세스 GPU 지분(%)

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
conda activate "${CONDA_ENV}"
set -u

TS_DATE="$(date +%Y.%m.%d)"
TS_TIME="$(date +%H%M%S)"
TS_HM="$(date +%H%M)"

cd "${PROJECT_DIR}"

start_mps () {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_$TASK_ID/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_$TASK_ID/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[MPS] starting ..."
    nvidia-cuda-mps-control -d || true
  else
    echo "[MPS] nvidia-cuda-mps-control not found. Continuing without MPS."
    ENABLE_MPS=0
  fi
}
stop_mps () {
  if [[ "${ENABLE_MPS:-1}" == "1" ]]; then
    echo "[MPS] stopping ..."
    echo quit | nvidia-cuda-mps-control || true
    rm -rf "/tmp/mps_${JOB_ID}_${TASK_ID}" || true
  fi
}
trap stop_mps EXIT
if [[ "${ENABLE_MPS}" == "1" ]]; then start_mps; fi

setup_for_index () {
  local IDX="$1"
  if (( IDX < 0 || IDX > ARRAY_MAX || IDX >= TOTAL_COMB )); then
    echo "[SKIP] IDX=${IDX} out of scheduled/total range"; return 1
  fi

  local g_stride=1
  local s_stride=$((g_stride * NUM_G))
  local t_stride=$((s_stride * NUM_S))
  #local a_stride=$((s_stride * NUM_S))
  #local d_stride=$((a_stride * NUM_A))

  # 인덱싱: A(빠름) → R(d_s_z_lr) → P(phi_lr) → N(num_layers) → D(d_hidden)(느림)
  GIDX=$(( (IDX / g_stride) % NUM_G ))
  SIDX=$(( (IDX / s_stride) % NUM_S ))
  TIDX=$(( (IDX / t_stride) % NUM_T ))
  
  guidance_scale="${GUIDANCE_SCALE[$GIDX]}"
  strength="${STRENGTH[$SIDX]}"
  step="${STEP[$TIDX]}"
  RUN_TAG="${JOB_ID}_${IDX}"
}

run_one () {
  local IDX="$1"
  local LOG_BASE="/scratch2/james2602/playground/Motionblur/Motionblur/results/demo/output/strength_${strength}/guidance_scale_${guidance_scale}/step_${step}"
  local LOG_DIR="${LOG_BASE}/logs"
  mkdir -p "$LOG_DIR"
  local OUT="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_idx${IDX}.out"
  local ERR="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_idx${IDX}.err"
  
  echo "[LAUNCH] guidance_scale=${GUIDANCE_SCALE[$GIDX]} → ${OUT}"

  (
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${GPU_SHARE}"
    export OMP_NUM_THREADS="${CPUS_PER_RUN}"
    export MKL_NUM_THREADS="${CPUS_PER_RUN}"

    cd "${PROJECT_DIR}"

    exec > >(stdbuf -oL tee -a "${OUT}") 2> >(stdbuf -oL tee -a "${ERR}" >&2)

    exec python run.py \
      guidance_scale="${GUIDANCE_SCALE[$GIDX]}" \
      strength="${STRENGTH[$SIDX]}" \
      output="/scratch2/james2602/playground/Motionblur/Motionblur/results/demo/output/strength_${strength}_guidance_scale_${guidance_scale}_step_${step}_000001.png" \
      mode="run" \
      method="demo" \
      save_grid=true \
      use_wandb=false \
      wandb_project="urlbaseline_pretraining_260101" \
      hydra.run.dir=${LOG_BASE} 
  ) &
}

CANDIDATES=("${TASK_ID}")
VALID=()
for id in "${CANDIDATES[@]}"; do
  if (( id >= 0 && id <= ARRAY_MAX && id < TOTAL_COMB )); then
    VALID+=("${id}")
  fi
done

N_ACTIVE=${#VALID[@]}
if (( N_ACTIVE == 0 )); then
  echo "[WARN] No valid indices to run. Exiting."; exit 0
fi


for idx in "${VALID[@]}"; do
  setup_for_index "${idx}" || continue
  run_one "${idx}"
done

wait
stop_mps
echo "Done: Job=${JOB_ID}, Task=${TASK_ID}, ran indices: ${VALID[*]}"
