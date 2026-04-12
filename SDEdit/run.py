"""SDEdit workspace: SD3.5 Img2Img for motion deblurring.

Modes
-----
- train:     LoRA fine-tuning on transformer (sharp images, SD3 flow-match objective)
- test:      sweep (strength × guidance_scale × num_inference_steps) over the GoPro
             test split. Logs avg PSNR/SSIM/LPIPS per combo + per-image CSVs.
- inference: run SD3.5 Img2Img on a single input image and save a before/after grid.

Results land in `SDEdit/result/<cfg.name>/<mode>/`.
"""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GoProPairDataset(Dataset):
    """Blur / Sharp paired GoPro dataset. Resizes so min side == resolution, snapped to /64."""

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
def _to_uint8(img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
# Workspace
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    # -- paths --------------------------------------------------------------
    def _result_dir(self, mode: str) -> Path:
        d = SCRIPT_DIR / "result" / self.cfg.name / mode
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- pipeline -----------------------------------------------------------
    def _build_pipeline(self):
        from diffusers import StableDiffusion3Img2ImgPipeline

        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            self.cfg.base_model, torch_dtype=self.dtype
        )
        if self.cfg.get("cpu_offload", True):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe

    def _load_lora(self, pipe) -> None:
        ckpt = self.cfg.get("checkpoint")
        if ckpt is None:
            return
        ckpt = str(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt}")
        pipe.load_lora_weights(os.path.dirname(ckpt), weight_name=os.path.basename(ckpt))
        print(f"[ckpt] loaded LoRA -> {ckpt}")

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

    # -- generation ---------------------------------------------------------
    def _img2img(self, pipe, image: Image.Image, strength: float, guidance: float, steps: int, seed: int) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        w, h = image.size
        return pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=image,
            height=h,
            width=w,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=g,
        ).images[0]

    # ========================================================================
    # INFERENCE  (single image → before/after grid)
    # ========================================================================
    def inference(self) -> None:
        out_dir = self._result_dir("inference")
        input_path = self.cfg.input
        img = snap_to_64(Image.open(input_path).convert("RGB")) # For VAE, PatchSize

        pipe = self._build_pipeline()
        self._load_lora(pipe)
        out = self._img2img(
            pipe, img, self.cfg.strength, self.cfg.guidance_scale, self.cfg.steps, self.cfg.seed
        )

        grid = make_grid([img, out])
        stem = Path(input_path).stem
        grid_path = out_dir / f"{stem}_grid.png"
        grid.save(grid_path)
        out.save(out_dir / f"{stem}_out.png")
        print(f"[inference] grid -> {grid_path}")

    # ========================================================================
    # TEST  (sweep strength × guidance × steps over GoPro test split)
    # ========================================================================
    def test(self) -> None:
        out_dir = self._result_dir("test")
        dataset = GoProPairDataset(
            self.cfg.data.blur_dir,
            self.cfg.data.sharp_dir,
            self.cfg.data.split_file,
            "test",
            self.cfg.data.resolution,
        )
        if bool(self.cfg.get("smoke", False)):
            dataset.filenames = dataset.filenames[:5]
        else:
            n_cap = self.cfg.test.get("num_images")
            if n_cap is not None:
                dataset.filenames = dataset.filenames[: int(n_cap)]
        print(f"[test] n_images={len(dataset)}")

        pipe = self._build_pipeline()
        self._load_lora(pipe)
        # Silence diffusers per-step pbar in test (outer tqdm handles progress)
        pipe.set_progress_bar_config(disable=True)
        lpips_scorer = LpipsScorer(self.device)

        strengths = list(self.cfg.test.strengths)
        guidances = list(self.cfg.test.guidance_scales)
        steps_list = list(self.cfg.test.steps_list)
        combos = [(s, g, n) for s in strengths for g in guidances for n in steps_list]
        print(f"[test] combos={len(combos)}  images={len(dataset)}  total_runs={len(combos) * len(dataset)}")

        combo_rows = []
        for strength, guidance, steps in tqdm(combos, desc="[combos]", position=0):
            tag = f"s{strength:.2f}_g{guidance:.1f}_n{int(steps)}"
            combo_dir = out_dir / tag
            combo_dir.mkdir(parents=True, exist_ok=True)
            per_rows: list[dict] = []

            for i in tqdm(range(len(dataset)), desc=f"[{tag}]", position=1, leave=False):
                sample = dataset[i]
                blur = sample["blur"]
                sharp = sample["sharp"]
                out_img = self._img2img(pipe, blur, strength, guidance, steps, self.cfg.seed)

                target = sharp.size
                gt_a = _to_uint8(sharp, target)
                blur_a = _to_uint8(blur, target)
                out_a = _to_uint8(out_img, target)

                p_blur, s_blur = compute_psnr_ssim(blur_a, gt_a)
                p_out, s_out = compute_psnr_ssim(out_a, gt_a)
                row = {
                    "filename": sample["filename"],
                    "psnr_blur": p_blur, "ssim_blur": s_blur, "lpips_blur": lpips_scorer(blur_a, gt_a),
                    "psnr_out": p_out, "ssim_out": s_out, "lpips_out": lpips_scorer(out_a, gt_a),
                }
                per_rows.append(row)

                if bool(self.cfg.test.get("save_images", False)):
                    out_img.save(combo_dir / f"{Path(sample['filename']).stem}.png")

                if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                    _dump_csv(per_rows, combo_dir / "per_image.csv")

            agg = _aggregate(per_rows)
            agg_row = {"strength": strength, "guidance": guidance, "steps": steps,
                       **{k: agg[k]["mean"] for k in agg}}
            combo_rows.append(agg_row)

            with open(combo_dir / "summary.json", "w") as f:
                json.dump({"config": {"strength": strength, "guidance": guidance, "steps": steps},
                           "metrics": agg, "n_images": len(per_rows)}, f, indent=2)

            print(f"[done] {tag}  psnr_out={agg['psnr_out']['mean']:.3f}  "
                  f"ssim_out={agg['ssim_out']['mean']:.4f}  lpips_out={agg['lpips_out']['mean']:.4f}")
            torch.cuda.empty_cache()

        # Sweep-level summary
        _dump_csv(combo_rows, out_dir / "sweep_summary.csv")
        with open(out_dir / "sweep_config.json", "w") as f:
            json.dump({"strengths": strengths, "guidances": guidances, "steps": steps_list,
                       "n_images": len(dataset)}, f, indent=2)
        print(f"[test] sweep summary -> {out_dir / 'sweep_summary.csv'}")

    # ========================================================================
    # TRAIN  (LoRA on transformer, SD3 flow-match objective on sharp images)
    # ========================================================================
    def train(self) -> None:
        from diffusers.training_utils import (
            cast_training_params,
            compute_density_for_timestep_sampling,
            compute_loss_weighting_for_sd3,
        )
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        out_dir = self._result_dir("train")
        tcfg = self.cfg.train

        pipe = self._build_pipeline()
        pipe.vae.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.requires_grad_(False)
        if tcfg.gradient_checkpointing:
            pipe.transformer.enable_gradient_checkpointing()

        lora_cfg = LoraConfig(
            r=tcfg.rank, lora_alpha=tcfg.alpha, lora_dropout=tcfg.dropout,
            target_modules=list(tcfg.target_modules),
        )
        pipe.transformer.add_adapter(lora_cfg)
        cast_training_params([pipe.transformer], dtype=torch.float32)
        pipe.transformer.train()

        optimizer = torch.optim.AdamW(
            [p for p in pipe.transformer.parameters() if p.requires_grad],
            lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay,
        )

        with torch.no_grad():
            prompt_embeds, pooled_embeds = self._encode_prompt(pipe)

        # Data
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

        # wandb
        use_wandb = bool(self.cfg.use_wandb)
        if use_wandb:
            import wandb
            wandb.init(project=self.cfg.wandb_project, entity=self.cfg.get("wandb_entity"),
                       name=f"SDEdit_{self.cfg.name}",
                       config=OmegaConf.to_container(self.cfg, resolve=True))

        def one_step(img: Image.Image):
            with torch.no_grad():
                target_lat = self._encode_image(pipe, img)
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
            pred = pipe.transformer(
                hidden_states=noisy, timestep=ts,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            w = compute_loss_weighting_for_sd3(tcfg.weighting_scheme, sigmas=sigmas).view(-1, 1, 1, 1)
            return (w * (pred.float() - target.float()).pow(2)).mean()

        @torch.no_grad()
        def validate() -> float:
            pipe.transformer.eval()
            losses = []
            for b in val_dl:
                losses.append(one_step(b["sharp"][0]).item())
            pipe.transformer.train()
            return float(np.mean(losses)) if losses else float("nan")

        def save_ckpt(label):
            lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.transformer))
            name = f"sdedit_lora_step{label}.safetensors"
            pipe.save_lora_weights(save_directory=str(out_dir), transformer_lora_layers=lora_layers, weight_name=name)
            print(f"[ckpt] -> {out_dir / name}")

        # loop
        max_steps = tcfg.max_train_steps
        print(f"[train] dataset={len(train_ds)}  val={len(val_ds)}  max_steps={max_steps}")
        global_step = 0
        window = deque(maxlen=100)
        pbar = tqdm(total=max_steps, desc="[train]", unit="step")
        done = False
        while not done:
            for batch in train_dl:
                global_step += 1
                loss = one_step(batch["sharp"][0])
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    [p for p in pipe.transformer.parameters() if p.requires_grad], tcfg.max_grad_norm,
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
# CSV / aggregation helpers
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
