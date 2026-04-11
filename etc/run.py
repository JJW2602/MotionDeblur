import json
import os
os.environ['HF_TOKEN'] = ''
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hydra
from diffusers import StableDiffusion3ControlNetPipeline, StableDiffusion3Img2ImgPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers, make_image_grid
from omegaconf import DictConfig
from PIL import Image

try:
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
except ImportError:
    LoraConfig = None
    get_peft_model_state_dict = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GoProPairDataset(Dataset):
    """Blur / Sharp paired dataset for GoPro motion-blur training."""

    def __init__(
        self,
        blur_dir: str,
        sharp_dir: str,
        split_file: str,
        split: str = "train",
        resolution: int = 512,
        canny_low: int = 100,
        canny_high: int = 200,
    ):
        with open(split_file, "r") as f:
            split_data = json.load(f)
        self.filenames = split_data[split]
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.resolution = resolution
        self.canny_low = canny_low
        self.canny_high = canny_high

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_and_resize(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.resolution / min(w, h)
        new_w = max(64, int(w * scale) // 64 * 64)
        new_h = max(64, int(h * scale) // 64 * 64)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        blur = self._load_and_resize(os.path.join(self.blur_dir, fname))
        sharp = self._load_and_resize(os.path.join(self.sharp_dir, fname))
        canny = build_canny_control_image(blur, self.canny_low, self.canny_high)
        return {"blur": blur, "sharp": sharp, "canny": canny, "filename": fname}

def get_device_and_dtype(device_arg: str | None) -> tuple[str, torch.dtype]:
    if device_arg is not None:
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return device, dtype


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_image_to_multiple_of_64(image: Image.Image) -> Image.Image:
    w, h = image.size
    w = max(64, (w // 64) * 64)
    h = max(64, (h // 64) * 64)
    return image.resize((w, h))


def build_canny_control_image(image: Image.Image, low: int, high: int) -> Image.Image:
    image_np = np.array(image)
    edges = cv2.Canny(image_np, low, high)
    return Image.fromarray(edges).convert("RGB")


def validate_method(method: str) -> None:
    if method not in {"sdedit", "controlnet"}:
        raise ValueError(f"Unsupported method '{method}'. Expected one of: sdedit, controlnet")


class Workspace:
    def __init__(self, cfg: DictConfig) -> None:
        validate_method(cfg.method)
        self.cfg = cfg
        self.device, self.dtype = get_device_and_dtype(cfg.device)
        self.canny_image: Image.Image | None = None
        # input image is only needed for inference (run) mode
        if cfg.mode != "train":
            self.image = load_image(cfg.input)
            self.train_image = resize_image_to_multiple_of_64(self.image)
        else:
            self.image = None
            self.train_image = None


    def _get_canny_image(self) -> Image.Image:
        if self.canny_image is None:
            self.canny_image = build_canny_control_image(
                self.image,
                self.cfg.canny_low,
                self.cfg.canny_high,
            )
        return self.canny_image

    def _build_img2img_pipeline(self) -> StableDiffusion3Img2ImgPipeline:
        return StableDiffusion3Img2ImgPipeline.from_pretrained(
            self.cfg.base_model,
            torch_dtype=self.dtype,
        ).to(self.device)

    def _encode_train_image(self, pipe: StableDiffusion3Img2ImgPipeline) -> torch.Tensor:
        image_tensor = pipe.image_processor.preprocess(self.train_image).to(device=self.device, dtype=pipe.vae.dtype)
        latents = retrieve_latents(pipe.vae.encode(image_tensor))
        return (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    def _encode_train_prompt(
        self, pipe: StableDiffusion3Img2ImgPipeline
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=self.cfg.prompt,
            prompt_2=self.cfg.get("prompt_2"),
            prompt_3=self.cfg.get("prompt_3"),
            negative_prompt=self.cfg.negative_prompt,
            negative_prompt_2=self.cfg.get("negative_prompt_2"),
            negative_prompt_3=self.cfg.get("negative_prompt_3"),
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds, pooled_prompt_embeds
    
    def _load_lora_checkpoint(self, pipe, controlnet=None) -> None:
        """Load LoRA weights from a checkpoint file if configured."""
        ckpt_path = self.cfg.get("checkpoint")
        if ckpt_path is None:
            return
        ckpt_path = str(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if self.cfg.method == "controlnet" and controlnet is not None:
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
            controlnet.load_state_dict(state_dict, strict=False)
            print(f"[ckpt] Loaded ControlNet LoRA from {ckpt_path}")
        else:
            pipe.load_lora_weights(
                os.path.dirname(ckpt_path),
                weight_name=os.path.basename(ckpt_path),
            )
            print(f"[ckpt] Loaded LoRA from {ckpt_path}")

    # I2I
    def run_sdedit(self) -> Image.Image:
        pipe = self._build_img2img_pipeline()
        self._load_lora_checkpoint(pipe)
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        result = pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=self.image,
            strength=self.cfg.strength,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
            width=self.cfg.width1,
            height=self.cfg.height1,
        ).images[0]
        return result

    def run_controlnet(self) -> Image.Image:
        controlnet = SD3ControlNetModel.from_pretrained(
            self.cfg.controlnet_model,
            torch_dtype=self.dtype,
        )
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.cfg.base_model,
            controlnet=controlnet,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._load_lora_checkpoint(pipe, controlnet=controlnet)

        canny_image = self._get_canny_image()
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        result = pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            control_image=canny_image,
            controlnet_conditioning_scale=self.cfg.controlnet_scale,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
        ).images[0]
        return result

    def save_output(self, output: Image.Image) -> None:
        output_dir = os.path.dirname(self.cfg.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output.save(self.cfg.output)
        print(f"Saved ({self.cfg.method}) -> {self.cfg.output}")

    def save_grid(self, output: Image.Image) -> None:
        grid_output = self.cfg.grid_output
        if grid_output is None:
            output_stem, _ = os.path.splitext(self.cfg.output)
            grid_output = f"{output_stem}_grid.png"

        grid_dir = os.path.dirname(grid_output)
        if grid_dir:
            os.makedirs(grid_dir, exist_ok=True)

        if self.cfg.method == "sdedit":
            grid = make_image_grid([self.image, output], rows=1, cols=2)
        else:
            grid = make_image_grid([self.image, self._get_canny_image(), output], rows=1, cols=3)
        grid.save(grid_output)
        print(f"Saved comparison grid -> {grid_output}")

    def run(self) -> None:
        if self.cfg.method == "sdedit":
            output = self.run_sdedit()
        elif self.cfg.method == "controlnet":
            output = self.run_controlnet()

        if self.cfg.save_grid:
            self.save_grid(output)
        else:
            self.save_output(output)
    
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _build_dataset(self, split: str = "train") -> GoProPairDataset:
        dcfg = self.cfg.train.data
        return GoProPairDataset(
            blur_dir=dcfg.blur_dir,
            sharp_dir=dcfg.sharp_dir,
            split_file=dcfg.split_file,
            split=split,
            resolution=dcfg.resolution,
            canny_low=self.cfg.canny_low,
            canny_high=self.cfg.canny_high,
        )

    def _encode_image_tensor(self, pipe, image: Image.Image) -> torch.Tensor:
        """Encode a single PIL image to VAE latents."""
        image_tensor = pipe.image_processor.preprocess(image).to(
            device=self.device, dtype=pipe.vae.dtype
        )
        latents = retrieve_latents(pipe.vae.encode(image_tensor))
        return (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    def train(self) -> None:
        if LoraConfig is None or get_peft_model_state_dict is None:
            raise ImportError("PEFT is required. Install `peft` to train LoRA.")

        train_cfg = self.cfg.train
        arch = train_cfg.tuning_architecture  # "SD3" or "controlnetSD3"

        # ── Build pipeline ──────────────────────────────────────────
        if arch == "controlnetSD3":
            controlnet = SD3ControlNetModel.from_pretrained(
                self.cfg.controlnet_model, torch_dtype=self.dtype,
            )
            pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                self.cfg.base_model, controlnet=controlnet, torch_dtype=self.dtype,
            ).to(self.device)
        else:  # SD3
            controlnet = None
            pipe = self._build_img2img_pipeline()

        # ── Freeze everything ───────────────────────────────────────
        pipe.vae.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.requires_grad_(False)
        if controlnet is not None:
            controlnet.requires_grad_(False)

        if train_cfg.gradient_checkpointing:
            pipe.transformer.enable_gradient_checkpointing()
            if controlnet is not None:
                controlnet.enable_gradient_checkpointing()
            if train_cfg.train_text_encoders:
                pipe.text_encoder.gradient_checkpointing_enable()
                pipe.text_encoder_2.gradient_checkpointing_enable()

        # ── LoRA adapters ───────────────────────────────────────────
        trainable_modules = []

        if arch == "controlnetSD3":
            # LoRA on the ControlNet
            controlnet_lora_config = LoraConfig(
                r=train_cfg.rank,
                lora_alpha=train_cfg.alpha,
                lora_dropout=train_cfg.dropout,
                target_modules=list(train_cfg.target_modules),
            )
            controlnet.add_adapter(controlnet_lora_config)
            trainable_modules.append(controlnet)
        else:
            # LoRA on the Transformer
            transformer_lora_config = LoraConfig(
                r=train_cfg.rank,
                lora_alpha=train_cfg.alpha,
                lora_dropout=train_cfg.dropout,
                target_modules=list(train_cfg.target_modules),
            )
            pipe.transformer.add_adapter(transformer_lora_config)
            trainable_modules.append(pipe.transformer)

        if train_cfg.train_text_encoders:
            text_lora_config = LoraConfig(
                r=train_cfg.text_encoder_rank,
                lora_alpha=train_cfg.text_encoder_alpha,
                lora_dropout=train_cfg.dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            pipe.text_encoder.add_adapter(text_lora_config)
            pipe.text_encoder_2.add_adapter(text_lora_config)
            trainable_modules.extend([pipe.text_encoder, pipe.text_encoder_2])

        cast_training_params(trainable_modules, dtype=torch.float32)

        optimizer = torch.optim.AdamW(
            [p for m in trainable_modules for p in m.parameters() if p.requires_grad],
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        # ── Set train / eval ────────────────────────────────────────
        if arch == "controlnetSD3":
            controlnet.train()
            pipe.transformer.eval()
        else:
            pipe.transformer.train()

        if train_cfg.train_text_encoders:
            pipe.text_encoder.train()
            pipe.text_encoder_2.train()
        else:
            pipe.text_encoder.eval()
            pipe.text_encoder_2.eval()

        # ── Cache prompt embeddings ─────────────────────────────────
        if not train_cfg.train_text_encoders:
            with torch.no_grad():
                cached_prompt_embeds, cached_pooled_prompt_embeds = self._encode_train_prompt(pipe)
        else:
            cached_prompt_embeds = None
            cached_pooled_prompt_embeds = None

        # ── Dataset & DataLoader ────────────────────────────────────
        dataset = self._build_dataset("train")
        val_dataset = self._build_dataset("test")
        def pil_collate(batch):
            """Keep PIL images as lists instead of stacking into tensors."""
            return {k: [d[k] for d in batch] for k in batch[0]}

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=train_cfg.data.num_workers,
            drop_last=True,
            collate_fn=pil_collate,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=train_cfg.data.num_workers,
            drop_last=False,
            collate_fn=pil_collate,
        )

        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        scheduler = pipe.scheduler
        num_train_timesteps = scheduler.config.num_train_timesteps

        # ── W&B ─────────────────────────────────────────────────────
        use_wandb = self.cfg.use_wandb
        if use_wandb:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.get("wandb_entity", None),
                config={
                    "tuning_architecture": arch,
                    "max_train_steps": train_cfg.max_train_steps,
                    "learning_rate": train_cfg.learning_rate,
                    "rank": train_cfg.rank,
                    "alpha": train_cfg.alpha,
                    "resolution": train_cfg.data.resolution,
                    "dataset_size": len(dataset),
                    "weighting_scheme": train_cfg.weighting_scheme,
                    "gradient_checkpointing": train_cfg.gradient_checkpointing,
                    "train_text_encoders": train_cfg.train_text_encoders,
                },
            )

        # ── Training loop ───────────────────────────────────────────
        global_step = 0
        max_steps = train_cfg.max_train_steps
        num_epochs = max(1, max_steps // len(dataset) + 1)

        # ── Smoothed loss trackers ─────────────────────────────────
        ema_loss = 0.0
        ema_loss_unweighted = 0.0
        ema_decay = 0.99
        # Timestep-binned running averages (low/mid/high thirds)
        bin_edges = [0, num_train_timesteps // 3, 2 * num_train_timesteps // 3, num_train_timesteps]
        bin_names = ["low_t", "mid_t", "high_t"]
        bin_loss_sum = [0.0, 0.0, 0.0]
        bin_loss_count = [0, 0, 0]
        # Window average
        from collections import deque
        loss_window = deque(maxlen=100)

        # ── Validation helper ───────────────────────────────────────
        val_generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        def run_validation(val_dl, val_gen):
            """Compute average loss over the full validation set."""
            was_training = pipe.transformer.training
            pipe.transformer.eval()
            if controlnet is not None:
                controlnet.eval()

            val_loss_sum = 0.0
            val_count = 0
            # Per-bin validation loss
            val_bin_sum = [0.0, 0.0, 0.0]
            val_bin_count = [0, 0, 0]

            with torch.no_grad():
                for val_batch in val_dl:
                    val_sharp = val_batch["sharp"][0]
                    val_target_latents = self._encode_image_tensor(pipe, val_sharp)

                    if train_cfg.train_text_encoders:
                        ve, vp = cached_prompt_embeds, cached_pooled_prompt_embeds
                    else:
                        ve, vp = cached_prompt_embeds, cached_pooled_prompt_embeds

                    val_noise = torch.randn(
                        val_target_latents.shape, generator=val_gen,
                        device=self.device, dtype=self.dtype,
                    )
                    val_u = compute_density_for_timestep_sampling(
                        weighting_scheme=train_cfg.weighting_scheme,
                        batch_size=1, device=self.device, generator=val_gen,
                    )
                    val_indices = (val_u * num_train_timesteps).long().clamp(max=num_train_timesteps - 1).cpu()
                    val_ts = scheduler.timesteps[val_indices].to(device=self.device)
                    val_sigmas = scheduler.sigmas[val_indices].to(device=self.device, dtype=self.dtype)
                    val_noisy = scheduler.scale_noise(val_target_latents, val_ts, val_noise)
                    val_target = val_noise - val_target_latents

                    if arch == "controlnetSD3":
                        val_canny = val_batch["canny"][0]
                        val_canny_t = pipe.image_processor.preprocess(val_canny).to(
                            device=self.device, dtype=self.dtype,
                        )
                        val_cn_out = controlnet(
                            hidden_states=val_noisy, timestep=val_ts,
                            encoder_hidden_states=ve, pooled_projections=vp,
                            controlnet_cond=val_canny_t, return_dict=False,
                        )
                        val_pred = pipe.transformer(
                            hidden_states=val_noisy, timestep=val_ts,
                            encoder_hidden_states=ve, pooled_projections=vp,
                            block_controlnet_hidden_states=val_cn_out[0],
                            return_dict=False,
                        )[0]
                    else:
                        val_pred = pipe.transformer(
                            hidden_states=val_noisy, timestep=val_ts,
                            encoder_hidden_states=ve, pooled_projections=vp,
                            return_dict=False,
                        )[0]

                    sample_loss = (val_pred.float() - val_target.float()).pow(2).mean().item()
                    val_loss_sum += sample_loss
                    val_count += 1

                    vt = val_ts[0].item()
                    for bi in range(3):
                        if bin_edges[bi] <= vt < bin_edges[bi + 1]:
                            val_bin_sum[bi] += sample_loss
                            val_bin_count[bi] += 1
                            break

            # Restore training mode
            if was_training:
                pipe.transformer.train()
            if controlnet is not None and arch == "controlnetSD3":
                controlnet.train()

            avg_loss = val_loss_sum / max(val_count, 1)
            bin_avgs = {}
            for bi, bname in enumerate(bin_names):
                if val_bin_count[bi] > 0:
                    bin_avgs[bname] = val_bin_sum[bi] / val_bin_count[bi]
            return avg_loss, bin_avgs

        # ── Checkpoint save helper ─────────────────────────────────
        def save_checkpoint(step_label):
            output_dir = Path(train_cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Replace .safetensors with _step{label}.safetensors
            base_name = train_cfg.weight_name.replace(".safetensors", "")
            ckpt_name = f"{base_name}_step{step_label}.safetensors"

            if arch == "controlnetSD3":
                controlnet_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(controlnet)
                )
                from safetensors.torch import save_file
                save_file(controlnet_lora_layers, str(output_dir / ckpt_name))
            else:
                transformer_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(pipe.transformer)
                )
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None
                if train_cfg.train_text_encoders:
                    text_encoder_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(pipe.text_encoder)
                    )
                    text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(pipe.text_encoder_2)
                    )
                pipe.save_lora_weights(
                    save_directory=str(output_dir),
                    transformer_lora_layers=transformer_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                    weight_name=ckpt_name,
                )
            print(f"[ckpt] Saved -> {output_dir / ckpt_name}")

        print(f"[train] arch={arch}  dataset={len(dataset)}  val={len(val_dataset)}  "
              f"max_steps={max_steps}  epochs≈{num_epochs}")

        pbar = tqdm(total=max_steps, desc=f"[train] {arch}", unit="step")

        for epoch in range(1, num_epochs + 1):
            for batch in dataloader:
                global_step += 1
                if global_step > max_steps:
                    break

                # Target = sharp image latents
                sharp_img = batch["sharp"][0]  # PIL
                with torch.no_grad():
                    target_latents = self._encode_image_tensor(pipe, sharp_img)

                # Prompt
                prompt_context = nullcontext() if train_cfg.train_text_encoders else torch.no_grad()
                with prompt_context:
                    if train_cfg.train_text_encoders:
                        prompt_embeds, pooled_prompt_embeds = self._encode_train_prompt(pipe)
                    else:
                        prompt_embeds = cached_prompt_embeds
                        pooled_prompt_embeds = cached_pooled_prompt_embeds

                # Noise
                noise = torch.randn(
                    target_latents.shape, generator=generator,
                    device=self.device, dtype=self.dtype,
                )
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=train_cfg.weighting_scheme,
                    batch_size=1, device=self.device, generator=generator,
                )
                indices = (u * num_train_timesteps).long().clamp(max=num_train_timesteps - 1).cpu()
                timesteps = scheduler.timesteps[indices].to(device=self.device)
                sigmas = scheduler.sigmas[indices].to(device=self.device, dtype=self.dtype)
                noisy_latents = scheduler.scale_noise(target_latents, timesteps, noise)
                target = noise - target_latents

                # Forward
                if arch == "controlnetSD3":
                    canny_img = batch["canny"][0]  # PIL
                    canny_tensor = pipe.image_processor.preprocess(canny_img).to(
                        device=self.device, dtype=self.dtype,
                    )
                    controlnet_output = controlnet(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        controlnet_cond=canny_tensor,
                        return_dict=False,
                    )
                    model_pred = pipe.transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=controlnet_output[0],
                        return_dict=False,
                    )[0]
                else:
                    model_pred = pipe.transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                weighting = compute_loss_weighting_for_sd3(
                    train_cfg.weighting_scheme, sigmas=sigmas,
                ).view(-1, 1, 1, 1)
                loss = (weighting * (model_pred.float() - target.float()).pow(2)).mean()
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for m in trainable_modules for p in m.parameters() if p.requires_grad],
                    train_cfg.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # ── Logging ────────────────────────────────────────
                loss_val = loss.item()
                loss_unweighted = (model_pred.float() - target.float()).pow(2).mean().item()

                # EMA smoothing
                if global_step == 1:
                    ema_loss = loss_val
                    ema_loss_unweighted = loss_unweighted
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss_val
                    ema_loss_unweighted = ema_decay * ema_loss_unweighted + (1 - ema_decay) * loss_unweighted

                # Window average
                loss_window.append(loss_val)
                window_avg = sum(loss_window) / len(loss_window)

                # Timestep bin accumulation
                t_val = timesteps[0].item()
                for bi in range(3):
                    if bin_edges[bi] <= t_val < bin_edges[bi + 1]:
                        bin_loss_sum[bi] += loss_unweighted
                        bin_loss_count[bi] += 1
                        break

                pbar.update(1)
                pbar.set_postfix(epoch=epoch, loss=f"{loss_val:.6f}", ema=f"{ema_loss:.6f}")

                if use_wandb:
                    log_dict = {
                        # Raw loss (per-step, noisy)
                        "loss/raw": loss_val,
                        "loss/raw_unweighted": loss_unweighted,
                        # Smoothed loss (학습 추이 확인용 — 이것을 보세요)
                        "loss/ema": ema_loss,
                        "loss/ema_unweighted": ema_loss_unweighted,
                        "loss/window_avg_100": window_avg,
                        # Schedule info
                        "schedule/epoch": epoch,
                        "schedule/timestep": t_val,
                        "schedule/sigma": sigmas[0].item(),
                        # Optimizer
                        "optim/lr": optimizer.param_groups[0]["lr"],
                        "optim/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    }

                    # Timestep-binned loss (log every log_every steps)
                    if global_step % train_cfg.log_every == 0:
                        for bi, bname in enumerate(bin_names):
                            if bin_loss_count[bi] > 0:
                                log_dict[f"loss_by_t/{bname}_avg"] = bin_loss_sum[bi] / bin_loss_count[bi]
                                log_dict[f"loss_by_t/{bname}_count"] = bin_loss_count[bi]
                        # Reset bins after logging
                        bin_loss_sum = [0.0, 0.0, 0.0]
                        bin_loss_count = [0, 0, 0]

                    # GPU memory
                    if torch.cuda.is_available() and (global_step == 1 or global_step % train_cfg.log_every == 0):
                        log_dict["system/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                        log_dict["system/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

                    wandb.log(log_dict, step=global_step)

                # ── Periodic checkpoint ───────────────────────────
                if global_step % train_cfg.save_every == 0:
                    save_checkpoint(global_step)

                # ── Periodic validation (every val_every steps) ───
                if global_step % train_cfg.val_every == 0:
                    print(f"[val] Running validation at step {global_step}...")
                    val_avg_loss, val_bin_avgs = run_validation(val_dataloader, val_generator)
                    print(f"[val] step={global_step}  val_loss={val_avg_loss:.6f}")
                    if use_wandb:
                        val_log = {
                            "val/loss": val_avg_loss,
                            "val/epoch": epoch,
                        }
                        for bname, bavg in val_bin_avgs.items():
                            val_log[f"val_by_t/{bname}_avg"] = bavg
                        wandb.log(val_log, step=global_step)

            if global_step >= max_steps:
                break

        pbar.close()
        if use_wandb:
            wandb.finish()

        # ── Save final weights ─────────────────────────────────────
        save_checkpoint("final")


@hydra.main(version_base=None, config_path=".", config_name="run")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    if cfg.mode == "train":
        workspace.train()
    else:
        workspace.run()


if __name__ == "__main__":
    main()
