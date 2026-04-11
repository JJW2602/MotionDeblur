"""Baseline evaluation: Restormer (Stage1) + SD3.5 Large Blur ControlNet (Stage2).

Pipeline (per image in GoPro test split):
    blur ──► Restormer ──► db_img
                              │
                              ├─(Run A) SD3.5 ControlNet(control=db_img)  ──► out_A
                              └─(Run B) SD3.5 ControlNet(control=blur)    ──► out_B

Metrics (vs sharp GT) computed for: blur, db_img, out_A, out_B.
PSNR/SSIM via skimage; LPIPS via the `lpips` package (VGG backbone).

Run:
    cd /scratch2/james2602/playground/Motionblur
    python -m eval.restormer_sd35_eval \
        --config-path=../configs/eval --config-name=restormer_sd35

    # Smoke (first 5 images):
    python -m eval.restormer_sd35_eval ... ++smoke=true

This is fine-tuning-free. Outputs:
    <output_dir>/per_image.csv
    <output_dir>/summary.json
    <output_dir>/images/<filename>/{blur,db,out_A,out_B,gt}.png
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the existing GoProPairDataset from run.py (canny field is ignored).
from run import GoProPairDataset  # noqa: E402

RESTORMER_ARCH_PATH = (
    PROJECT_ROOT
    / "prior_aware"
    / "stage1"
    / "Restormer"
    / "basicsr"
    / "models"
    / "archs"
    / "restormer_arch.py"
)


def _load_restormer_arch_module():
    """Import restormer_arch.py without dragging in the basicsr package."""
    spec = importlib.util.spec_from_file_location(
        "restormer_arch_standalone", RESTORMER_ARCH_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Restormer arch from {RESTORMER_ARCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Restormer Stage1
# ---------------------------------------------------------------------------
def build_restormer(ckpt_path: str, device: str) -> torch.nn.Module:
    arch = _load_restormer_arch_module()
    # Default Restormer config matches the released motion_deblurring.pth.
    model = arch.Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Restormer checkpoint not found at: {ckpt_path}\n"
            "Download motion_deblurring.pth from the Restormer release page and "
            "place it under "
            "prior_aware/stage1/Restormer/Motion_Deblurring/pretrained_models/."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("params", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def restormer_forward(
    model: torch.nn.Module, pil_img: Image.Image, device: str
) -> Image.Image:
    """Run Restormer on a PIL image. Pads to multiples of 8 then unpads."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    out = model(tensor)
    out = out[..., :h, :w].clamp(0.0, 1.0)
    out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_np)


# ---------------------------------------------------------------------------
# SD3.5 Large + Blur ControlNet
# ---------------------------------------------------------------------------
def build_sd35_pipeline(cfg: DictConfig):
    """Lazy import diffusers so the module is importable without it installed."""
    from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel

    controlnet = SD3ControlNetModel.from_pretrained(
        cfg.sd35.controlnet, torch_dtype=torch.float16
    )
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        cfg.sd35.base,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    return pipe


def run_controlnet(
    pipe,
    cfg: DictConfig,
    control_image: Image.Image,
    height: int,
    width: int,
    seed: int,
) -> Image.Image:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=cfg.sd35.prompt,
        negative_prompt=cfg.sd35.negative_prompt,
        control_image=control_image,
        height=height,
        width=width,
        num_inference_steps=int(cfg.sd35.num_inference_steps),
        guidance_scale=float(cfg.sd35.guidance_scale),
        controlnet_conditioning_scale=float(cfg.sd35.controlnet_conditioning_scale),
        generator=generator,
    ).images[0]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _to_uint8_rgb(img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Resize to target (W, H) and return uint8 HWC RGB array."""
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    if crop_border > 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border]
        gt = gt[crop_border:-crop_border, crop_border:-crop_border]
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(
        structural_similarity(gt, pred, channel_axis=-1, data_range=255)
    )
    return psnr, ssim


class LpipsScorer:
    def __init__(self, device: str = "cuda") -> None:
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net="vgg").to(device).eval()

    @torch.no_grad()
    def __call__(self, pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
        def to_tensor(arr):
            t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            return t.to(self.device)

        d = self.model(to_tensor(pred_uint8), to_tensor(gt_uint8))
        return float(d.item())


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict]) -> dict:
    keys = [k for k in rows[0].keys() if k != "filename"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        summary[k] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
            "n": int(len(vals)),
        }
    return summary


@hydra.main(version_base=None, config_path="../configs/eval", config_name="restormer_sd35")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This eval requires CUDA (SD3.5 Large is too large for CPU).")

    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if cfg.eval.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset
    dataset = GoProPairDataset(
        blur_dir=cfg.data.blur_dir,
        sharp_dir=cfg.data.sharp_dir,
        split_file=cfg.data.split_file,
        split=cfg.data.split,
        resolution=int(cfg.data.resolution),
    )
    n_total = len(dataset)
    indices = list(range(n_total))
    if cfg.get("image_idx") is not None:
        idx = int(cfg.image_idx)
        assert 0 <= idx < n_total, f"image_idx={idx} out of range [0, {n_total})"
        indices = [idx]
    elif bool(cfg.get("smoke", False)):
        indices = indices[:5]
    print(f"[data] split={cfg.data.split}  total={n_total}  using={len(indices)}")

    # ---- Models
    print("[models] loading Restormer ...")
    restormer = build_restormer(cfg.restormer.ckpt, device)

    print("[models] loading SD3.5 + Blur ControlNet ...")
    pipe = build_sd35_pipeline(cfg)

    print("[models] loading LPIPS (VGG) ...")
    lpips_scorer = LpipsScorer(device=device)

    # ---- Loop
    rows: list[dict] = []
    t0 = time.time()
    for k, idx in enumerate(tqdm(indices, desc="eval")):
        sample = dataset[idx]
        fname: str = sample["filename"]
        blur: Image.Image = sample["blur"]
        sharp: Image.Image = sample["sharp"]
        W, H = blur.size  # GoProPairDataset already aligns to multiples of 64

        # Stage 1: Restormer
        db_img = restormer_forward(restormer, blur, device)

        # Stage 2: ControlNet — Run A (control = Restormer output)
        out_a: Optional[Image.Image] = None
        if bool(cfg.eval.run_a):
            out_a = run_controlnet(pipe, cfg, db_img, H, W, int(cfg.sd35.seed))
            torch.cuda.empty_cache()

        # Stage 2: ControlNet — Run B (control = raw blur)
        out_b: Optional[Image.Image] = None
        if bool(cfg.eval.run_b):
            out_b = run_controlnet(pipe, cfg, blur, H, W, int(cfg.sd35.seed))
            torch.cuda.empty_cache()

        # Align all to GT resolution for metrics
        target_size = sharp.size  # (W, H)
        gt_arr = _to_uint8_rgb(sharp, target_size)
        blur_arr = _to_uint8_rgb(blur, target_size)
        db_arr = _to_uint8_rgb(db_img, target_size)
        crop = int(cfg.eval.crop_border)

        row: dict = {"filename": fname}

        psnr_blur, ssim_blur = compute_psnr_ssim(blur_arr, gt_arr, crop)
        row["psnr_blur"] = psnr_blur
        row["ssim_blur"] = ssim_blur
        row["lpips_blur"] = lpips_scorer(blur_arr, gt_arr)

        psnr_db, ssim_db = compute_psnr_ssim(db_arr, gt_arr, crop)
        row["psnr_db"] = psnr_db
        row["ssim_db"] = ssim_db
        row["lpips_db"] = lpips_scorer(db_arr, gt_arr)

        if out_a is not None:
            a_arr = _to_uint8_rgb(out_a, target_size)
            psnr_a, ssim_a = compute_psnr_ssim(a_arr, gt_arr, crop)
            row["psnr_a"] = psnr_a
            row["ssim_a"] = ssim_a
            row["lpips_a"] = lpips_scorer(a_arr, gt_arr)

        if out_b is not None:
            b_arr = _to_uint8_rgb(out_b, target_size)
            psnr_b, ssim_b = compute_psnr_ssim(b_arr, gt_arr, crop)
            row["psnr_b"] = psnr_b
            row["ssim_b"] = ssim_b
            row["lpips_b"] = lpips_scorer(b_arr, gt_arr)

        rows.append(row)

        # Save imagery (resized versions used for the metrics)
        if cfg.eval.save_images:
            stem = Path(fname).stem
            sub = images_dir / stem
            sub.mkdir(parents=True, exist_ok=True)
            Image.fromarray(blur_arr).save(sub / "blur.png")
            Image.fromarray(db_arr).save(sub / "db.png")
            Image.fromarray(gt_arr).save(sub / "gt.png")
            if out_a is not None:
                Image.fromarray(_to_uint8_rgb(out_a, target_size)).save(sub / "out_A.png")
            if out_b is not None:
                Image.fromarray(_to_uint8_rgb(out_b, target_size)).save(sub / "out_B.png")

        # Periodic incremental dump (so SLURM crashes don't lose all progress)
        if (k + 1) % 10 == 0 or (k + 1) == len(indices):
            _dump_csv(rows, output_dir / "per_image.csv")

    elapsed = time.time() - t0
    summary = aggregate(rows)
    summary["_meta"] = {
        "elapsed_sec": round(elapsed, 1),
        "n_images": len(rows),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] {len(rows)} images in {elapsed/60:.1f} min")
    print(f"[done] summary -> {output_dir / 'summary.json'}")
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        print(f"  {k:14s}  mean={v['mean']:.4f}  std={v['std']:.4f}  n={v['n']}")


def _dump_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    # Ensure all rows share the same key set (some rows may lack a/b)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    keys = ["filename"] + sorted(k for k in all_keys if k != "filename")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
