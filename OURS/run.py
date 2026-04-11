"""OURS workspace: Stage1 (Restormer) + Stage2 (SD3.5 ControlNet).

Pipeline
--------
    blur ─► Restormer (frozen) ─► db_img ─► SD3.5 ControlNet(control=db_img) ─► out

Modes
-----
- train:     LoRA fine-tuning on the ControlNet (Restormer + transformer frozen).
             Control = Restormer(blur), target = sharp.
- test:      run the full pipeline on the GoPro test split and log avg
             PSNR/SSIM/LPIPS for blur / db_img / final out vs sharp GT.
- inference: run on a single image, save [blur, db_img, out] grid.

Results land in `OURS/result/<cfg.name>/<mode>/`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESTORMER_ARCH_PATH = (
    PROJECT_ROOT / "etc" / "prior_aware" / "stage1" / "Restormer"
    / "basicsr" / "models" / "archs" / "restormer_arch.py"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GoProPairDataset(Dataset):
    def __init__(self, blur_dir: str, sharp_dir: str, split_file: str, split: str, resolution: int):
        with open(split_file, "r") as f:
            self.filenames = json.load(f)[split]
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.resolution = resolution

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
        return {
            "blur": self._load(os.path.join(self.blur_dir, fname)),
            "sharp": self._load(os.path.join(self.sharp_dir, fname)),
            "filename": fname,
        }


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
            return (torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).to(self.device)

        return float(self.model(to_t(pred), to_t(gt)).item())


def snap_to_64(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((max(64, (w // 64) * 64), max(64, (h // 64) * 64)), Image.LANCZOS)


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
# Restormer (Stage1) — loaded standalone to avoid basicsr package conflicts
# ---------------------------------------------------------------------------
def _load_restormer_arch_module():
    spec = importlib.util.spec_from_file_location("restormer_arch_standalone", RESTORMER_ARCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Restormer arch from {RESTORMER_ARCH_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_restormer(ckpt_path: str, device: str) -> torch.nn.Module:
    arch = _load_restormer_arch_module()
    model = arch.Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Restormer checkpoint not found: {ckpt_path}\n"
            "Download motion_deblurring.pth from the Restormer release page."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("params", ckpt), strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def restormer_forward(model: torch.nn.Module, pil_img: Image.Image, device: str) -> Image.Image:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    out = model(t)[..., :h, :w].clamp(0.0, 1.0)
    out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_np)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._restormer: torch.nn.Module | None = None

    def _result_dir(self, mode: str) -> Path:
        d = SCRIPT_DIR / "result" / self.cfg.name / mode
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_restormer(self) -> torch.nn.Module:
        if self._restormer is None:
            self._restormer = build_restormer(self.cfg.restormer_ckpt, self.device)
        return self._restormer

    # -- pipeline -----------------------------------------------------------
    def _build_pipeline(self):
        from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline

        controlnet = SD3ControlNetModel.from_pretrained(self.cfg.controlnet_model, torch_dtype=self.dtype)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.cfg.base_model, controlnet=controlnet, torch_dtype=self.dtype,
        )
        pipe.set_progress_bar_config(disable=True)
        if self.cfg.get("cpu_offload", True):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe, controlnet

    def _load_lora(self, pipe, controlnet) -> None:
        ckpt = self.cfg.get("checkpoint")
        if ckpt is None:
            return
        ckpt = str(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt}")
        from safetensors.torch import load_file

        controlnet.load_state_dict(load_file(ckpt), strict=False)
        print(f"[ckpt] loaded ControlNet LoRA -> {ckpt}")

    def _encode_prompt(self, pipe):
        pe, _, pp, _ = pipe.encode_prompt(
            prompt=self.cfg.prompt,
            prompt_2=self.cfg.get("prompt_2"),
            prompt_3=self.cfg.get("prompt_3"),
            negative_prompt=self.cfg.negative_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return pe, pp

    def _encode_image(self, pipe, img: Image.Image) -> torch.Tensor:
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents

        t = pipe.image_processor.preprocess(img).to(device=self.device, dtype=pipe.vae.dtype)
        latents = retrieve_latents(pipe.vae.encode(t))
        return (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    def _generate(self, pipe, control_image: Image.Image, height: int, width: int, seed: int) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        return pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            control_image=control_image,
            height=height, width=width,
            num_inference_steps=int(self.cfg.steps),
            guidance_scale=float(self.cfg.guidance_scale),
            controlnet_conditioning_scale=float(self.cfg.controlnet_scale),
            generator=g,
        ).images[0]

    # ========================================================================
    # INFERENCE
    # ========================================================================
    def inference(self) -> None:
        out_dir = self._result_dir("inference")
        blur = snap_to_64(Image.open(self.cfg.input).convert("RGB"))

        restormer = self._get_restormer()
        db_img = restormer_forward(restormer, blur, self.device)

        pipe, controlnet = self._build_pipeline()
        self._load_lora(pipe, controlnet)
        W, H = blur.size
        out = self._generate(pipe, db_img, H, W, self.cfg.seed)

        grid = make_grid([blur, db_img, out])
        stem = Path(self.cfg.input).stem
        grid.save(out_dir / f"{stem}_grid.png")
        out.save(out_dir / f"{stem}_out.png")
        db_img.save(out_dir / f"{stem}_stage1.png")
        print(f"[inference] grid -> {out_dir / f'{stem}_grid.png'}")

    # ========================================================================
    # TEST
    # ========================================================================
    def test(self) -> None:
        out_dir = self._result_dir("test")
        dataset = GoProPairDataset(
            self.cfg.data.blur_dir, self.cfg.data.sharp_dir,
            self.cfg.data.split_file, "test", self.cfg.data.resolution,
        )
        if bool(self.cfg.get("smoke", False)):
            dataset.filenames = dataset.filenames[:5]
        print(f"[test] n_images={len(dataset)}")

        restormer = self._get_restormer()
        pipe, controlnet = self._build_pipeline()
        self._load_lora(pipe, controlnet)
        lpips_scorer = LpipsScorer(self.device)

        rows: list[dict] = []
        for i in tqdm(range(len(dataset)), desc="[test]"):
            sample = dataset[i]
            blur = sample["blur"]
            sharp = sample["sharp"]

            db_img = restormer_forward(restormer, blur, self.device)
            W, H = blur.size
            out_img = self._generate(pipe, db_img, H, W, self.cfg.seed)

            target = sharp.size
            gt_a = _to_uint8(sharp, target)
            blur_a = _to_uint8(blur, target)
            db_a = _to_uint8(db_img, target)
            out_a = _to_uint8(out_img, target)

            p_b, s_b = compute_psnr_ssim(blur_a, gt_a)
            p_d, s_d = compute_psnr_ssim(db_a, gt_a)
            p_o, s_o = compute_psnr_ssim(out_a, gt_a)
            rows.append({
                "filename": sample["filename"],
                "psnr_blur": p_b, "ssim_blur": s_b, "lpips_blur": lpips_scorer(blur_a, gt_a),
                "psnr_db": p_d, "ssim_db": s_d, "lpips_db": lpips_scorer(db_a, gt_a),
                "psnr_out": p_o, "ssim_out": s_o, "lpips_out": lpips_scorer(out_a, gt_a),
            })

            if bool(self.cfg.test.get("save_images", False)):
                stem = Path(sample["filename"]).stem
                sub = out_dir / "images" / stem
                sub.mkdir(parents=True, exist_ok=True)
                Image.fromarray(blur_a).save(sub / "blur.png")
                Image.fromarray(db_a).save(sub / "db.png")
                Image.fromarray(out_a).save(sub / "out.png")
                Image.fromarray(gt_a).save(sub / "gt.png")

            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                _dump_csv(rows, out_dir / "per_image.csv")

        agg = _aggregate(rows)
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"config": {
                "restormer_ckpt": str(self.cfg.restormer_ckpt),
                "controlnet_model": self.cfg.controlnet_model,
                "guidance_scale": self.cfg.guidance_scale,
                "controlnet_scale": self.cfg.controlnet_scale,
                "steps": self.cfg.steps,
            }, "metrics": agg, "n_images": len(rows)}, f, indent=2)
        print(f"[test] psnr_db={agg['psnr_db']['mean']:.3f}  psnr_out={agg['psnr_out']['mean']:.3f}  "
              f"lpips_out={agg['lpips_out']['mean']:.4f}")
        print(f"[test] summary -> {out_dir / 'summary.json'}")

    # ========================================================================
    # TRAIN  (LoRA on ControlNet; Restormer + transformer frozen)
    # ========================================================================
    def train(self) -> None:
        from diffusers.training_utils import (
            cast_training_params,
            compute_density_for_timestep_sampling,
            compute_loss_weighting_for_sd3,
        )
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        out_dir = self._result_dir("train")
        tcfg = self.cfg.train

        restormer = self._get_restormer()
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

        lora_cfg = LoraConfig(
            r=tcfg.rank, lora_alpha=tcfg.alpha, lora_dropout=tcfg.dropout,
            target_modules=list(tcfg.target_modules),
        )
        controlnet.add_adapter(lora_cfg)
        cast_training_params([controlnet], dtype=torch.float32)
        controlnet.train()
        pipe.transformer.eval()

        optimizer = torch.optim.AdamW(
            [p for p in controlnet.parameters() if p.requires_grad],
            lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay,
        )

        with torch.no_grad():
            prompt_embeds, pooled_embeds = self._encode_prompt(pipe)

        train_ds = GoProPairDataset(
            self.cfg.data.blur_dir, self.cfg.data.sharp_dir,
            self.cfg.data.split_file, "train", self.cfg.data.resolution,
        )
        val_ds = GoProPairDataset(
            self.cfg.data.blur_dir, self.cfg.data.sharp_dir,
            self.cfg.data.split_file, "test", self.cfg.data.resolution,
        )
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=self.cfg.data.num_workers,
                              drop_last=True, collate_fn=pil_collate)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=self.cfg.data.num_workers,
                            drop_last=False, collate_fn=pil_collate)

        scheduler = pipe.scheduler
        num_train_ts = scheduler.config.num_train_timesteps
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        use_wandb = bool(self.cfg.use_wandb)
        if use_wandb:
            import wandb
            wandb.init(project=self.cfg.wandb_project, entity=self.cfg.get("wandb_entity"),
                       name=f"OURS_{self.cfg.name}",
                       config=OmegaConf.to_container(self.cfg, resolve=True))

        def one_step(blur: Image.Image, sharp: Image.Image):
            with torch.no_grad():
                db_img = restormer_forward(restormer, blur, self.device)
                target_lat = self._encode_image(pipe, sharp)
            noise = torch.randn(target_lat.shape, generator=generator, device=self.device, dtype=self.dtype)
            u = compute_density_for_timestep_sampling(
                weighting_scheme=tcfg.weighting_scheme, batch_size=1,
                device=self.device, generator=generator,
            )
            idx = (u * num_train_ts).long().clamp(max=num_train_ts - 1).cpu()
            ts = scheduler.timesteps[idx].to(self.device)
            sigmas = scheduler.sigmas[idx].to(device=self.device, dtype=self.dtype)
            noisy = scheduler.scale_noise(target_lat, ts, noise)
            target = noise - target_lat

            control_t = pipe.image_processor.preprocess(db_img).to(device=self.device, dtype=self.dtype)
            cn_out = controlnet(
                hidden_states=noisy, timestep=ts,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                controlnet_cond=control_t, return_dict=False,
            )
            pred = pipe.transformer(
                hidden_states=noisy, timestep=ts,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                block_controlnet_hidden_states=cn_out[0],
                return_dict=False,
            )[0]
            w = compute_loss_weighting_for_sd3(tcfg.weighting_scheme, sigmas=sigmas).view(-1, 1, 1, 1)
            return (w * (pred.float() - target.float()).pow(2)).mean()

        @torch.no_grad()
        def validate() -> float:
            controlnet.eval()
            losses = []
            for b in val_dl:
                losses.append(one_step(b["blur"][0], b["sharp"][0]).item())
            controlnet.train()
            return float(np.mean(losses)) if losses else float("nan")

        def save_ckpt(label):
            sd = get_peft_model_state_dict(controlnet)
            name = f"ours_lora_step{label}.safetensors"
            save_file(sd, str(out_dir / name))
            print(f"[ckpt] -> {out_dir / name}")

        max_steps = tcfg.max_train_steps
        print(f"[train] dataset={len(train_ds)}  val={len(val_ds)}  max_steps={max_steps}")
        global_step = 0
        window = deque(maxlen=100)
        pbar = tqdm(total=max_steps, desc="[train]", unit="step")
        done = False
        while not done:
            for batch in train_dl:
                global_step += 1
                loss = one_step(batch["blur"][0], batch["sharp"][0])
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    [p for p in controlnet.parameters() if p.requires_grad], tcfg.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                lv = loss.item()
                window.append(lv)
                pbar.update(1)
                pbar.set_postfix(loss=f"{lv:.5f}", avg=f"{np.mean(window):.5f}")

                if use_wandb:
                    wandb.log({"loss/raw": lv, "loss/window_avg_100": float(np.mean(window)),
                               "optim/lr": optimizer.param_groups[0]["lr"],
                               "optim/grad_norm": grad.item() if torch.is_tensor(grad) else float(grad)},
                              step=global_step)

                if global_step % tcfg.val_every == 0:
                    vl = validate()
                    print(f"[val] step={global_step} val_loss={vl:.5f}")
                    if use_wandb:
                        wandb.log({"val/loss": vl}, step=global_step)
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
        out[k] = {"mean": float(vals.mean()) if len(vals) else float("nan"),
                  "std": float(vals.std()) if len(vals) else float("nan"),
                  "n": int(len(vals))}
    return out


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
