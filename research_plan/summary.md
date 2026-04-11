# Project: Motionblur

**Path**: `/scratch2/james2602/playground/Motionblur`  
**Last updated**: 2026-04-11

---

## 1. Project Overview

Motion deblurring 프로젝트. DiffBIR 프레임워크 기반으로 **Stage 1 (기존 deblurring model)** + **Stage 2 (diffusion ControlNet refinement)** 2-stage pipeline 구현. 최종 목적은 DiffBlur처럼 stage1에서 deblur 후, stage2에서 deblurred image를 diffusion ControlNet input으로 넣어 고품질 복원.

---

## 2. 핵심 구조 (2026-04-11 재편)

```
Motionblur/
├── research_plan/          # summary.md + YYYY-MM-DD.md 일일 로그 (Claude memory root)
├── dataset/gopro/          # GoPro motion blur dataset (train 900 / test 100)
├── DiffBIR/                # 원본 DiffBIR framework (유지)
├── SDEdit/                 # Workspace 1: SD3.5 Img2Img
│   ├── run.py / run.yaml
│   ├── script/parameter_tuning.slurm         # inference 4-D sweep
│   ├── script/parameter_tuning_test.slurm    # test 3-D sweep
│   └── result/<cfg.name>/<mode>/
├── ControlNet/             # Workspace 2: SD3.5 + selectable ControlNet
│   ├── run.py / run.yaml
│   └── result/<cfg.name>/<mode>/
├── OURS/                   # Workspace 3: Restormer Stage1 + SD3.5 ControlNet Stage2
│   ├── run.py / run.yaml
│   └── result/<cfg.name>/<mode>/
└── etc/                    # Legacy: old run.py, eval/, prior_aware/, script/, toy_experiment/, flow_grpo/, configs/, results/, wandb/
```

각 workspace `run.py` 는 self-contained (공통 helper 복제)하며 `mode ∈ {train, test, inference}` 를 지원. 결과는 `<workspace>/result/<cfg.name>/<mode>/` 에 저장. `cfg.name` 에 slash 를 쓰면 nested dir 로 자동 해석.

---

## 3. Two-Stage Pipeline

### Stage 1: Degradation Removal (Deblurring)

| 항목 | 내용 |
|------|------|
| **모델** | SwinIR (default), BSRNet, SCUNet |
| **구조** | Swin Transformer, embed_dim=180, window=8, 8 stages x 6 blocks |
| **Loss** | MSE (predicted vs GT) |
| **Metrics** | LPIPS, PSNR |
| **학습** | 150k steps, batch 96, lr 1e-4 |
| **Degradation** | Gaussian blur(sigma 0.1-12) + downsample(1-12x) + noise(0-15) + JPEG(30-100) |

### Stage 2: Diffusion ControlNet Refinement

| 항목 | 내용 |
|------|------|
| **모델** | ControlLDM (Stable Diffusion v2.1 + ControlNet) |
| **UNet** | 320 base channels, 4 attention levels, context_dim=1024 |
| **VAE** | 4-channel latent, scale=0.18215 |
| **ControlNet** | Stage1 output을 VAE encode -> 13개 control signals -> UNet에 주입 |
| **Diffusion** | Linear schedule (beta 0.00085~0.012), 1000 timesteps, eps prediction |
| **학습** | Phase1: lr 1e-4 (30k steps), Phase2: lr 1e-5 (50k steps), batch 256 |
| **Inference** | EDM DPM++ 3M SDE sampler, 10-50 steps, CFG 4.0-8.0 |

### SD3.5 기반 Workspace 3종 (2026-04-11 재편)
- **Base model**: `stabilityai/stable-diffusion-3.5-large` + `enable_model_cpu_offload()` (RTX 3090 24GB 대응)
- **Prompt** (공통): `"a sharp, clean, natural photo, no motion blur, high detail"`
- **LoRA**: rank=8, alpha=8, target_modules=`[to_q, to_k, to_v, to_out.0]`
- **SDEdit**: transformer LoRA, sharp 이미지에 SD3 flow-match objective (blur 미참여, 분포 편향 기대)
- **ControlNet**: ControlNet LoRA. `control_source ∈ {blur, canny}` 로 control image 생성
- **OURS**: ControlNet LoRA only (Restormer 는 frozen). Control = Restormer(blur). Restormer arch 은 `etc/prior_aware/.../restormer_arch.py` 를 importlib 로 standalone 로드

---

## 4. 주요 파일

### SD3.5 Workspace (Main)
| 파일 | 설명 |
|------|------|
| `SDEdit/run.py` | SD3.5 Img2Img — train/test/inference. test 는 strength×guidance×steps grid sweep |
| `SDEdit/run.yaml` | SDEdit Hydra config |
| `SDEdit/script/parameter_tuning.slurm` | Inference 4-D sweep (strength × guidance × steps × input_images) |
| `SDEdit/script/parameter_tuning_test.slurm` | Test 3-D sweep, 각 task 가 1 combo × 100장 test split |
| `ControlNet/run.py` | SD3.5 + selectable ControlNet (blur / canny) |
| `ControlNet/run.yaml` | ControlNet Hydra config |
| `OURS/run.py` | Restormer Stage1 + SD3.5 ControlNet Stage2 (LoRA on ControlNet only) |
| `OURS/run.yaml` | OURS Hydra config |

### Legacy / 참조용 (`etc/`)
| 파일 | 설명 |
|------|------|
| `etc/run.py` | 옛 monolithic run.py (SDEdit + ControlNet 통합 버전) |
| `etc/eval/restormer_sd35_eval.py` | 2026-04-07 Restormer + SD3.5 baseline 평가 스크립트 |
| `etc/prior_aware/stage1/Restormer/basicsr/models/archs/restormer_arch.py` | Restormer 아키텍처 (OURS 가 importlib 로 참조) |

### DiffBIR (원본 framework, 유지)
| 파일 | 설명 |
|------|------|
| `DiffBIR/train_stage1.py` | Stage 1 학습 (SwinIR) |
| `DiffBIR/train_stage2.py` | Stage 2 학습 (ControlNet) |
| `DiffBIR/inference.py` | 통합 inference |
| `DiffBIR/diffbir/model/cldm.py` | ControlLDM |

---

## 6. Dataset

- **GoPro**: blur/sharp paired images, **train 900장 / test 100장** (split.json 기준, 2026-04-11 test 129→100 축소)
- 실제 파일 보유량: `blur/images/`, `sharp/images/` 각 **1029장** (2026-04-11 `gopro_deblur.zip` 압축 해제)
- **Synthetic degradation**: Gaussian blur + downsample + noise + JPEG (CodeformerDataset, DiffBIR 쪽만 사용)

---

## 7. 대안 Stage 1 모델 (`etc/prior_aware/stage1/`)

- **FFTformer**: Frequency domain transformer
- **Restormer**: Efficient transformer for high-resolution restoration
  - `OURS/run.py` 가 `etc/prior_aware/.../restormer_arch.py` 를 `importlib.util.spec_from_file_location` 로 standalone 로드 (basicsr 패키지 의존성 회피)
  - Checkpoint: `etc/prior_aware/stage1/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth` (다운로드 필요)

---

## 8. 평가 / 학습 파이프라인

### SDEdit Parameter Sweep (2026-04-11 신설)
- **Inference sweep**: `SDEdit/script/parameter_tuning.slurm` — 4-D grid (strength × guidance × steps × input_images). 각 SLURM array task = 1 inference. 결과는 `SDEdit/result/param_tuning/s<s>_g<g>_n<n>/inference/<stem>_grid.png`
- **Test sweep**: `SDEdit/script/parameter_tuning_test.slurm` — 3-D grid. 각 task 가 1 (s,g,n) combo 로 100장 test split 을 돌려 PSNR/SSIM/LPIPS 집계. 결과는 `SDEdit/result/param_tuning_test/s<s>_g<g>_n<n>/test/`
- **Hydra 1-combo override 트릭**: test slurm 이 `"test.strengths=[0.5]"` 형태로 1-element list 를 전달 → `run.py` 의 internal sweep 이 degenerate 하게 동작, run.py 수정 없이 array parallelize

### 공통 metric
- PSNR/SSIM (skimage) + LPIPS (VGG)
- VRAM: SD3.5 Large fp16 + `enable_model_cpu_offload()` → RTX 3090 24GB 안에서 동작
- diffusion conda env 에 `einops`, `lpips` 설치됨

---

## 9. 현재 상태

- **DiffBIR 기반 2-stage pipeline**: 구현 완료 (SwinIR + ControlNet), 2026-03~04 wandb 로그 참조
- **2026-04-07**: Restormer + SD3.5 Large Blur ControlNet fine-tuning-free baseline 평가 스크립트 (`etc/eval/restormer_sd35_eval.py`) 구축
- **2026-04-11 (NEW)**: 프로젝트 구조 전면 재편 → 3-workspace (SDEdit / ControlNet / OURS). SDEdit parameter sweep SLURM 2종 준비 완료. `gopro_deblur.zip` 압축 해제 완료. **SLURM 실제 제출은 아직** — HF_TOKEN, Restormer ckpt, `mkdir -p result/_logs` 가 선행 필요
- **알려진 wire-up 이슈**: `SDEdit/run.yaml` 의 `test.num_images` 필드가 아직 `run.py` 에 반영 안 됨 (smoke flag 만 사용 중)
