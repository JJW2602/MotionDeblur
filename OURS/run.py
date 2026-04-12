"""OURS workspace: Stage1 (Restormer) + Stage2 (SD3.5 ControlNet, DiffBIR-style concat).

Pipeline
--------
    blur -> [Stage1: Restormer subprocess] -> db_img
                                                |
                                           VAE encode -> prior_latent (16ch)
                                                |
    noise z --- cat(z, prior) -> ConcatProj(32->16) -> ControlNet --> block_samples
      |                                                    ^               |
      |                                          controlnet_cond=prior  (additive)
      |                                                                    |
      +-------------------------------------------> Transformer <----------+
                                                        |
                                                   noise_pred -> scheduler.step -> ...

Modes
-----
- inference: Stage1 on single image -> Stage2 -> save [blur, stage1, stage2] grid.
- test:      Stage1 on test split + metrics -> Stage2 on all + metrics -> summary.
- train:     Pre-compute Stage1 outputs -> LoRA on ControlNet + ConcatProjection.

Results land in ``OURS/result/<cfg.name>/{stage1,stage2}/<mode>/``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# ConcatProjection — DiffBIR-style latent concat for ControlNet
# ---------------------------------------------------------------------------
class ConcatProjection(nn.Module):
    """Project cat(noisy_z, prior_latent) from 2C to C channels.

    Initialised as identity on first C channels (noise passthrough) and
    zeros on the last C channels (prior).  At step 0 the output equals
    the original noisy_z, so the pretrained ControlNet starts from its
    normal behaviour.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self._init_weights(channels)

    def _init_weights(self, channels: int) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)  # [C, 2C, 1, 1]
            for i in range(channels):
                self.proj.weight[i, i, 0, 0] = 1.0

    def forward(self, noisy_z: torch.Tensor, prior_latent: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([noisy_z, prior_latent], dim=1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GoProPairDataset(Dataset):
    """GoPro blur/sharp pair dataset with optional pre-computed Stage1 dir."""

    def __init__(
        self,
        blur_dir: str,
        sharp_dir: str,
        split_file: str,
        split: str,
        resolution: int,
        stage1_dir: str | None = None,
    ):
        with open(split_file, "r") as f:
            self.filenames = json.load(f)[split]
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.resolution = resolution
        self.stage1_dir = stage1_dir

    def __len__(self) -> int:
        return len(self.filenames)

    def _load(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.resolution / min(w, h)
        new_w = max(64, int(w * scale) // 64 * 64)
        new_h = max(64, int(h * scale) // 64 * 64)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        stem = Path(fname).stem
        result = {
            "blur": self._load(os.path.join(self.blur_dir, fname)),
            "sharp": self._load(os.path.join(self.sharp_dir, fname)),
            "filename": fname,
        }
        if self.stage1_dir is not None:
            s1_path = os.path.join(self.stage1_dir, f"{stem}.png")
            result["stage1"] = self._load(s1_path)
        return result


def pil_collate(batch):
    return {k: [d[k] for d in batch] for k in batch[0]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _to_uint8(img: Image.Image, target: tuple[int, int]) -> np.ndarray:
    if img.size != target:
        img = img.resize(target, Image.LANCZOS)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=-1, data_range=255))
    return psnr, ssim


class LpipsScorer:
    def __init__(self, device: str) -> None:
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net="vgg").to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        def to_t(a):
            return (
                torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            ).to(self.device)

        return float(self.model(to_t(pred), to_t(gt)).item())


def snap_to_64(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize(
        (max(64, (w // 64) * 64), max(64, (h // 64) * 64)), Image.LANCZOS
    )


def make_grid(images: list[Image.Image]) -> Image.Image:
    h = max(im.size[1] for im in images)
    w_total = sum(im.size[0] for im in images)
    grid = Image.new("RGB", (w_total, h), (0, 0, 0))
    x = 0
    for im in images:
        grid.paste(im, (x, (h - im.size[1]) // 2))
        x += im.size[0]
    return grid


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def _result_dir(self, *parts: str) -> Path:
        d = SCRIPT_DIR / "result" / self.cfg.name
        for p in parts:
            d = d / p
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ==================================================================
    # Stage 1 — Restormer via subprocess
    # ==================================================================
    def run_stage1(self, input_dir: str, result_dir: str) -> Path:
        """Run Restormer demo.py via subprocess.

        Returns the directory containing restored images
        (demo.py saves to ``<result_dir>/<task>/``).
        """
        s1 = self.cfg.stage1
        restormer_dir = str(s1.restormer_dir)
        task = str(s1.task)

        output_path = Path(result_dir) / task
        if output_path.exists() and any(output_path.iterdir()):
            print(f"[stage1] cached outputs found at {output_path}, skipping")
            return output_path

        cmd = [
            sys.executable,
            "demo.py",
            "--task", task,
            "--input_dir", str(input_dir),
            "--result_dir", str(result_dir),
        ]
        if s1.get("tile") is not None:
            cmd += ["--tile", str(s1.tile)]
            cmd += ["--tile_overlap", str(s1.get("tile_overlap", 32))]

        print(f"[stage1] running: {' '.join(cmd)}")
        print(f"[stage1] cwd: {restormer_dir}")
        subprocess.run(cmd, cwd=restormer_dir, check=True)
        return output_path

    def _compute_stage1_metrics(
        self, stage1_img_dir: Path, filenames: list[str], out_dir: Path
    ) -> dict:
        """Compute PSNR/SSIM/LPIPS between Stage1 outputs and GT sharp images."""
        lpips_scorer = LpipsScorer(self.device)
        rows: list[dict] = []
        grid_dir = out_dir / "grids"
        grid_dir.mkdir(parents=True, exist_ok=True)

        for fname in tqdm(filenames, desc="[stage1 metrics]"):
            stem = Path(fname).stem
            s1_path = stage1_img_dir / f"{stem}.png"
            if not s1_path.exists():
                print(f"[stage1 metrics] WARNING: missing {s1_path}")
                continue
            s1_img = Image.open(s1_path).convert("RGB")
            sharp = Image.open(os.path.join(self.cfg.data.sharp_dir, fname)).convert("RGB")
            blur = Image.open(os.path.join(self.cfg.data.blur_dir, fname)).convert("RGB")

            target = sharp.size
            gt_a = _to_uint8(sharp, target)
            s1_a = _to_uint8(s1_img, target)
            blur_a = _to_uint8(blur, target)

            p_b, s_b = compute_psnr_ssim(blur_a, gt_a)
            p_s1, s_s1 = compute_psnr_ssim(s1_a, gt_a)
            rows.append({
                "filename": fname,
                "psnr_blur": p_b, "ssim_blur": s_b, "lpips_blur": lpips_scorer(blur_a, gt_a),
                "psnr_stage1": p_s1, "ssim_stage1": s_s1, "lpips_stage1": lpips_scorer(s1_a, gt_a),
            })

            grid = make_grid([
                Image.fromarray(blur_a),
                Image.fromarray(s1_a),
                Image.fromarray(gt_a),
            ])
            grid.save(grid_dir / f"{stem}.png")

        agg = _aggregate(rows)
        metrics = {"per_image": rows, "aggregate": agg, "n_images": len(rows)}
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"[stage1] psnr_stage1={agg['psnr_stage1']['mean']:.3f}  "
            f"lpips_stage1={agg['lpips_stage1']['mean']:.4f}"
        )
        return agg

    # ==================================================================
    # Stage 2 — SD3.5 ControlNet with DiffBIR-style concat
    # ==================================================================
    def _build_pipeline(self):
        from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline

        s2 = self.cfg.stage2
        controlnet = SD3ControlNetModel.from_pretrained(
            s2.controlnet_model, torch_dtype=self.dtype,
        )
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            s2.base_model, controlnet=controlnet, torch_dtype=self.dtype,
        )
        pipe.set_progress_bar_config(disable=True)
        if s2.get("cpu_offload", True):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe, controlnet

    def _load_lora(self, controlnet) -> None:
        ckpt = self.cfg.stage2.get("lora_checkpoint")
        if ckpt is None:
            return
        ckpt = str(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt}")
        from safetensors.torch import load_file

        controlnet.load_state_dict(load_file(ckpt), strict=False)
        print(f"[ckpt] loaded ControlNet LoRA -> {ckpt}")

    def _build_concat_proj(self) -> ConcatProjection:
        proj = ConcatProjection(channels=16)
        ckpt = self.cfg.stage2.get("concat_proj_checkpoint")
        if ckpt is not None:
            ckpt = str(ckpt)
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"ConcatProjection checkpoint not found: {ckpt}")
            proj.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"[ckpt] loaded ConcatProjection -> {ckpt}")
        proj.to(self.device, dtype=self.dtype)
        return proj

    def _encode_prompt(self, pipe):
        s2 = self.cfg.stage2
        pe, _, pp, _ = pipe.encode_prompt(
            prompt=s2.prompt,
            prompt_2=s2.get("prompt_2"),
            prompt_3=s2.get("prompt_3"),
            negative_prompt=s2.negative_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return pe, pp

    def _encode_image(self, pipe, img: Image.Image) -> torch.Tensor:
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import (
            retrieve_latents,
        )

        t = pipe.image_processor.preprocess(img).to(
            device=self.device, dtype=pipe.vae.dtype,
        )
        latents = retrieve_latents(pipe.vae.encode(t))
        return (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    @torch.no_grad()
    def _generate(
        self,
        pipe,
        controlnet,
        concat_proj: ConcatProjection,
        stage1_img: Image.Image,
        height: int,
        width: int,
        seed: int,
    ) -> Image.Image:
        """Custom denoising loop with DiffBIR-style concat conditioning."""
        s2 = self.cfg.stage2
        g = torch.Generator(device=self.device).manual_seed(seed)

        # Encode Stage1 output -> prior latent
        prior_latent = self._encode_image(pipe, stage1_img)

        # Encode prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(pipe)

        # Prepare noise latents
        num_channels = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            1, num_channels, height, width,
            prompt_embeds.dtype, self.device, g,
        )

        # Timesteps
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            retrieve_timesteps,
        )

        steps = int(s2.steps)
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, steps, self.device)

        # ControlNet config
        cn_config = controlnet.config
        if getattr(cn_config, "force_zeros_for_pooled_projection", True):
            cn_pooled = torch.zeros_like(pooled_prompt_embeds)
        else:
            cn_pooled = pooled_prompt_embeds
        if getattr(cn_config, "joint_attention_dim", None) is not None:
            cn_encoder_hidden = prompt_embeds
        else:
            cn_encoder_hidden = None

        cond_scale = float(s2.controlnet_scale)

        # Denoising loop
        for t in tqdm(timesteps, desc="[stage2 denoise]"):
            timestep = t.expand(latents.shape[0])

            # DiffBIR concat: project cat(noisy_z, prior) -> 16ch for ControlNet
            cn_hidden = concat_proj(latents, prior_latent)

            # ControlNet: double conditioning (concat path + additive controlnet_cond)
            control_block_samples = controlnet(
                hidden_states=cn_hidden,
                controlnet_cond=prior_latent,
                conditioning_scale=cond_scale,
                timestep=timestep,
                encoder_hidden_states=cn_encoder_hidden,
                pooled_projections=cn_pooled,
                return_dict=False,
            )[0]

            # Transformer: sees original noisy z + control signals
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
                return_dict=False,
            )[0]

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # VAE decode
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    # ==================================================================
    # INFERENCE
    # ==================================================================
    def inference(self) -> None:
        input_path = self.cfg.input
        stem = Path(input_path).stem

        # --- Stage 1 ---
        s1_result_dir = self._result_dir("stage1")
        s1_img_dir = self.run_stage1(input_path, str(s1_result_dir))
        s1_img = Image.open(s1_img_dir / f"{stem}.png").convert("RGB")

        # Stage 1 grid: [blur, stage1]
        blur_orig = Image.open(input_path).convert("RGB")
        s1_grid_dir = s1_result_dir / "grids"
        s1_grid_dir.mkdir(parents=True, exist_ok=True)
        s1_grid = make_grid([blur_orig, s1_img])
        s1_grid.save(s1_grid_dir / f"{stem}.png")
        print(f"[stage1] grid -> {s1_grid_dir / f'{stem}.png'}")

        # --- Stage 2 ---
        s2_out = self._result_dir("stage2", "inference")
        blur = snap_to_64(Image.open(input_path).convert("RGB"))
        s1_img_resized = snap_to_64(s1_img)
        W, H = s1_img_resized.size

        pipe, controlnet = self._build_pipeline()
        self._load_lora(controlnet)
        concat_proj = self._build_concat_proj()
        concat_proj.eval()

        out = self._generate(pipe, controlnet, concat_proj, s1_img_resized, H, W, self.cfg.stage2.seed)

        grid = make_grid([blur, s1_img_resized, out])
        grid.save(s2_out / f"{stem}_grid.png")
        out.save(s2_out / f"{stem}_out.png")
        s1_img.save(s2_out / f"{stem}_stage1.png")
        print(f"[inference] grid -> {s2_out / f'{stem}_grid.png'}")

    # ==================================================================
    # TEST
    # ==================================================================
    def test(self) -> None:
        dataset = GoProPairDataset(
            self.cfg.data.blur_dir,
            self.cfg.data.sharp_dir,
            self.cfg.data.split_file,
            "test",
            self.cfg.data.resolution,
        )
        if bool(self.cfg.test.get("smoke", False)):
            dataset.filenames = dataset.filenames[:5]
        num_images = self.cfg.test.get("num_images")
        if num_images is not None:
            dataset.filenames = dataset.filenames[:int(num_images)]
        filenames = list(dataset.filenames)
        print(f"[test] n_images={len(filenames)}")

        # --- Stage 1 ---
        s1_result_dir = self._result_dir("stage1")
        s1_img_dir = self.run_stage1(self.cfg.data.blur_dir, str(s1_result_dir))
        self._compute_stage1_metrics(s1_img_dir, filenames, s1_result_dir)

        # --- Stage 2 ---
        s2_out = self._result_dir("stage2", "test")
        pipe, controlnet = self._build_pipeline()
        self._load_lora(controlnet)
        concat_proj = self._build_concat_proj()
        concat_proj.eval()
        lpips_scorer = LpipsScorer(self.device)

        rows: list[dict] = []
        for i in tqdm(range(len(filenames)), desc="[stage2 test]"):
            sample = dataset[i]
            fname = sample["filename"]
            stem = Path(fname).stem
            blur = sample["blur"]
            sharp = sample["sharp"]

            s1_path = s1_img_dir / f"{stem}.png"
            s1_img = snap_to_64(Image.open(s1_path).convert("RGB"))
            W, H = s1_img.size

            out_img = self._generate(
                pipe, controlnet, concat_proj, s1_img, H, W, self.cfg.stage2.seed,
            )

            target = sharp.size
            gt_a = _to_uint8(sharp, target)
            blur_a = _to_uint8(blur, target)
            s1_a = _to_uint8(s1_img, target)
            out_a = _to_uint8(out_img, target)

            p_b, s_b = compute_psnr_ssim(blur_a, gt_a)
            p_s1, s_s1 = compute_psnr_ssim(s1_a, gt_a)
            p_o, s_o = compute_psnr_ssim(out_a, gt_a)
            rows.append({
                "filename": fname,
                "psnr_blur": p_b, "ssim_blur": s_b,
                "lpips_blur": lpips_scorer(blur_a, gt_a),
                "psnr_stage1": p_s1, "ssim_stage1": s_s1,
                "lpips_stage1": lpips_scorer(s1_a, gt_a),
                "psnr_out": p_o, "ssim_out": s_o,
                "lpips_out": lpips_scorer(out_a, gt_a),
            })

            if bool(self.cfg.test.get("save_images", False)):
                sub = s2_out / "images" / stem
                sub.mkdir(parents=True, exist_ok=True)
                Image.fromarray(blur_a).save(sub / "blur.png")
                Image.fromarray(s1_a).save(sub / "stage1.png")
                Image.fromarray(out_a).save(sub / "out.png")
                Image.fromarray(gt_a).save(sub / "gt.png")

            if (i + 1) % 10 == 0 or (i + 1) == len(filenames):
                _dump_csv(rows, s2_out / "per_image.csv")

        agg = _aggregate(rows)
        s2_cfg = self.cfg.stage2
        with open(s2_out / "summary.json", "w") as f:
            json.dump(
                {
                    "config": {
                        "controlnet_model": s2_cfg.controlnet_model,
                        "guidance_scale": s2_cfg.guidance_scale,
                        "controlnet_scale": s2_cfg.controlnet_scale,
                        "steps": s2_cfg.steps,
                        "lora_checkpoint": str(s2_cfg.get("lora_checkpoint")),
                        "concat_proj_checkpoint": str(s2_cfg.get("concat_proj_checkpoint")),
                    },
                    "metrics": agg,
                    "n_images": len(rows),
                },
                f,
                indent=2,
            )
        print(
            f"[test] psnr_stage1={agg['psnr_stage1']['mean']:.3f}  "
            f"psnr_out={agg['psnr_out']['mean']:.3f}  "
            f"lpips_out={agg['lpips_out']['mean']:.4f}"
        )
        print(f"[test] summary -> {s2_out / 'summary.json'}")

    # ==================================================================
    # TRAIN (LoRA on ControlNet + full params on ConcatProjection)
    # ==================================================================
    def train(self) -> None:
        from diffusers.training_utils import (
            cast_training_params,
            compute_density_for_timestep_sampling,
            compute_loss_weighting_for_sd3,
        )
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        tcfg = self.cfg.train
        s2_out = self._result_dir("stage2", "train")

        # --- Pre-compute Stage 1 outputs ---
        stage1_dir: str | None = None
        if bool(tcfg.get("precompute_stage1", True)):
            s1_result_dir = self._result_dir("stage1")
            s1_img_dir = self.run_stage1(self.cfg.data.blur_dir, str(s1_result_dir))
            stage1_dir = str(s1_img_dir)
            print(f"[train] stage1 precomputed -> {stage1_dir}")

        # --- Build pipeline ---
        pipe, controlnet = self._build_pipeline()
        pipe.vae.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.requires_grad_(False)
        controlnet.requires_grad_(False)
        if tcfg.gradient_checkpointing:
            controlnet.enable_gradient_checkpointing()

        # LoRA on ControlNet
        lora_cfg = LoraConfig(
            r=tcfg.rank,
            lora_alpha=tcfg.alpha,
            lora_dropout=tcfg.dropout,
            target_modules=list(tcfg.target_modules),
        )
        controlnet.add_adapter(lora_cfg)
        cast_training_params([controlnet], dtype=torch.float32)
        controlnet.train()
        pipe.transformer.eval()

        # ConcatProjection (full params, float32)
        concat_proj = ConcatProjection(channels=16).to(self.device, dtype=torch.float32)
        concat_proj.train()

        # Optimizer with two param groups
        cn_params = [p for p in controlnet.parameters() if p.requires_grad]
        cp_params = list(concat_proj.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": cn_params, "lr": tcfg.learning_rate},
                {"params": cp_params, "lr": tcfg.get("concat_proj_lr", 1e-3)},
            ],
            weight_decay=tcfg.weight_decay,
        )

        # Prompt embeddings (cached)
        with torch.no_grad():
            prompt_embeds, pooled_embeds = self._encode_prompt(pipe)

        # ControlNet pooled projections
        cn_config = controlnet.config
        if getattr(cn_config, "force_zeros_for_pooled_projection", True):
            cn_pooled = torch.zeros_like(pooled_embeds)
        else:
            cn_pooled = pooled_embeds
        if getattr(cn_config, "joint_attention_dim", None) is not None:
            cn_encoder_hidden = prompt_embeds
        else:
            cn_encoder_hidden = None

        # Datasets
        train_ds = GoProPairDataset(
            self.cfg.data.blur_dir,
            self.cfg.data.sharp_dir,
            self.cfg.data.split_file,
            "train",
            self.cfg.data.resolution,
            stage1_dir=stage1_dir,
        )
        val_ds = GoProPairDataset(
            self.cfg.data.blur_dir,
            self.cfg.data.sharp_dir,
            self.cfg.data.split_file,
            "test",
            self.cfg.data.resolution,
            stage1_dir=stage1_dir,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            drop_last=True,
            collate_fn=pil_collate,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            drop_last=False,
            collate_fn=pil_collate,
        )

        scheduler = pipe.scheduler
        num_train_ts = scheduler.config.num_train_timesteps
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.stage2.seed)

        # Param counts for logging
        n_lora = sum(p.numel() for p in cn_params)
        n_cp = sum(p.numel() for p in cp_params)
        n_total = n_lora + n_cp
        print(f"[train] trainable params: lora={n_lora:,}  concat_proj={n_cp:,}  total={n_total:,}")
        print(f"[train] params/image ratio: {n_total / len(train_ds):.1f}")

        use_wandb = bool(self.cfg.use_wandb)
        if use_wandb:
            import wandb

            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.get("wandb_entity"),
                name=f"OURS_{self.cfg.name}",
                config={
                    **OmegaConf.to_container(self.cfg, resolve=True),
                    "trainable_params": {
                        "lora": n_lora,
                        "concat_proj": n_cp,
                        "total": n_total,
                        "params_per_image": round(n_total / len(train_ds), 1),
                    },
                },
            )

        def one_step(blur_img: Image.Image, sharp_img: Image.Image, s1_img: Image.Image):
            with torch.no_grad():
                target_lat = self._encode_image(pipe, sharp_img)
                prior_lat = self._encode_image(pipe, s1_img)
            noise = torch.randn(
                target_lat.shape, generator=generator, device=self.device, dtype=self.dtype,
            )
            u = compute_density_for_timestep_sampling(
                weighting_scheme=tcfg.weighting_scheme,
                batch_size=1,
                device=self.device,
                generator=generator,
            )
            idx = (u * num_train_ts).long().clamp(max=num_train_ts - 1).cpu()
            ts = scheduler.timesteps[idx].to(self.device)
            sigmas = scheduler.sigmas[idx].to(device=self.device, dtype=self.dtype)
            noisy = scheduler.scale_noise(target_lat, ts, noise)
            target = noise - target_lat

            # DiffBIR concat projection
            cn_hidden = concat_proj(noisy, prior_lat)

            # ControlNet with double conditioning
            cn_out = controlnet(
                hidden_states=cn_hidden,
                controlnet_cond=prior_lat,
                timestep=ts,
                encoder_hidden_states=cn_encoder_hidden,
                pooled_projections=cn_pooled,
                return_dict=False,
            )

            # Transformer sees original noisy latent
            pred = pipe.transformer(
                hidden_states=noisy,
                timestep=ts,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                block_controlnet_hidden_states=cn_out[0],
                return_dict=False,
            )[0]
            w = compute_loss_weighting_for_sd3(
                tcfg.weighting_scheme, sigmas=sigmas,
            ).view(-1, 1, 1, 1)
            return (w * (pred.float() - target.float()).pow(2)).mean()

        @torch.no_grad()
        def validate() -> float:
            controlnet.eval()
            concat_proj.eval()
            losses = []
            for b in val_dl:
                losses.append(
                    one_step(b["blur"][0], b["sharp"][0], b["stage1"][0]).item()
                )
            controlnet.train()
            concat_proj.train()
            return float(np.mean(losses)) if losses else float("nan")

        def save_ckpt(label):
            # LoRA
            sd = get_peft_model_state_dict(controlnet)
            lora_name = f"ours_lora_step{label}.safetensors"
            save_file(sd, str(s2_out / lora_name))
            # ConcatProjection
            cp_name = f"ours_concat_proj_step{label}.pt"
            torch.save(concat_proj.state_dict(), str(s2_out / cp_name))
            print(f"[ckpt] -> {s2_out / lora_name}, {s2_out / cp_name}")

        max_steps = tcfg.max_train_steps
        steps_per_epoch = len(train_dl)
        print(
            f"[train] dataset={len(train_ds)}  val={len(val_ds)}  "
            f"max_steps={max_steps}  steps_per_epoch={steps_per_epoch}"
        )
        global_step = 0
        epoch = 0
        best_val_loss = float("inf")
        window: deque = deque(maxlen=100)
        pbar = tqdm(total=max_steps, desc="[train]", unit="step")
        done = False
        while not done:
            epoch += 1
            for batch in train_dl:
                global_step += 1
                loss = one_step(batch["blur"][0], batch["sharp"][0], batch["stage1"][0])
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    cn_params + cp_params, tcfg.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                lv = loss.item()
                window.append(lv)
                pbar.update(1)
                pbar.set_postfix(loss=f"{lv:.5f}", avg=f"{np.mean(window):.5f}")

                if use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "loss/raw": lv,
                            "loss/window_avg_100": float(np.mean(window)),
                            "train/epoch": epoch,
                            "train/epoch_frac": global_step / steps_per_epoch,
                            "optim/lr_cn": optimizer.param_groups[0]["lr"],
                            "optim/lr_cp": optimizer.param_groups[1]["lr"],
                            "optim/grad_norm": (
                                grad.item() if torch.is_tensor(grad) else float(grad)
                            ),
                        },
                        step=global_step,
                    )

                if global_step % tcfg.val_every == 0:
                    vl = validate()
                    is_best = vl < best_val_loss
                    if is_best:
                        best_val_loss = vl
                        save_ckpt("best")
                    print(
                        f"[val] step={global_step} epoch={epoch} "
                        f"val_loss={vl:.5f} best={best_val_loss:.5f}"
                        f"{' *' if is_best else ''}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "val/loss": vl,
                                "val/best_loss": best_val_loss,
                            },
                            step=global_step,
                        )
                if global_step % tcfg.save_every == 0:
                    save_ckpt(global_step)
                if global_step >= max_steps:
                    done = True
                    break
        pbar.close()
        save_ckpt("final")
        if use_wandb:
            wandb.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dump_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _aggregate(rows: list[dict]) -> dict:
    keys = [k for k in rows[0].keys() if k != "filename"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
            "n": int(len(vals)),
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=".", config_name="run")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    ws = Workspace(cfg)
    if cfg.mode == "train":
        ws.train()
    elif cfg.mode == "test":
        ws.test()
    elif cfg.mode == "inference":
        ws.inference()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
