"""Differentiable one-step Cosmos3-Nano I2V training adapter.

Inference still uses the normal 35-step Cosmos sampler.  During joint training
we sample one rectified-flow noise level, predict velocity once, recover
``x0_hat = x_t - sigma * v_theta``, and decode its terminal frame.  This keeps
the subgoal path differentiable while avoiding a 35-step backward graph.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from cosmos_pi05.cotrain.torch_joint import WorldModelForwardOutput


def sample_waver_sigmas(batch_size: int, device: torch.device, *, shift: float = 3.0) -> torch.Tensor:
    """Match Cosmos Nano's video SFT time distribution and 256px shift."""

    u = torch.rand(batch_size, device=device, dtype=torch.float32)
    raw = 1.0 - u - 1.29 * (torch.cos(torch.pi / 2.0 * u).square() - 1.0 + u)
    sigma = shift * raw / (1.0 + (shift - 1.0) * raw)
    return sigma.clamp_(0.001, 0.999)


def rectified_flow_interpolate(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    condition_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise only unconditioned frames and return ``(xt, velocity, sigma_eff)``."""

    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shapes differ: {tuple(clean.shape)} != {tuple(noise.shape)}")
    if clean.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W] latents, got {tuple(clean.shape)}")
    noisy_mask = (1.0 - condition_mask.to(device=clean.device, dtype=torch.float32)).view(1, 1, clean.shape[2], 1, 1)
    sigma_eff = sigma.reshape(-1, 1, 1, 1, 1) * noisy_mask
    xt = noise * sigma_eff + clean.float() * (1.0 - sigma_eff)
    velocity = noise - clean.float()
    return xt, velocity, sigma_eff


class DifferentiableCosmosI2V(nn.Module):
    """Wrap an exported ``Cosmos3OmniPipeline`` as a trainable one-step I2V module.

    The public pipeline is intentionally single-sample.  Local batch entries
    are processed sequentially; distributed FSDP supplies the global batch.
    """

    def __init__(self, pipeline: Any, *, resolution: int = 256, num_frames: int = 17, fps: float = 12.0) -> None:
        super().__init__()
        if num_frames < 5:
            raise ValueError("Cosmos I2V needs at least five pixel frames")
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        object.__setattr__(self, "_pipeline_helpers", pipeline)
        self.resolution = int(resolution)
        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self.register_buffer(
            "vae_latents_mean",
            torch.tensor(self.vae.config.latents_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "vae_latents_inv_std",
            1.0 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32),
            persistent=False,
        )
        self._token_cache: dict[str, list[int]] = {}
        self.vae.requires_grad_(requires_grad=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        resolution: int = 256,
        num_frames: int = 17,
        fps: float = 12.0,
    ) -> DifferentiableCosmosI2V:
        try:
            from diffusers import Cosmos3OmniPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Cosmos3OmniPipeline requires diffusers==0.39.0; run scripts/cotrain/setup_torch_e2e_env.sh"
            ) from exc

        pipeline = Cosmos3OmniPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            sound_tokenizer=None,
            enable_safety_checker=False,
            low_cpu_mem_usage=True,
        )
        return cls(pipeline, resolution=resolution, num_frames=num_frames, fps=fps)

    def enable_gradient_checkpointing(self) -> None:
        enable = getattr(self.transformer, "enable_gradient_checkpointing", None)
        if callable(enable):
            enable()
        elif hasattr(self.transformer, "gradient_checkpointing"):
            self.transformer.gradient_checkpointing = True

    def _prompt_ids(self, prompt: str) -> list[int]:
        cached = self._token_cache.get(prompt)
        if cached is not None:
            return cached
        ids, _ = self._pipeline_helpers.tokenize_prompt(
            prompt,
            negative_prompt="",
            num_frames=self.num_frames,
            height=self.resolution,
            width=self.resolution,
            fps=self.fps,
            use_system_prompt=True,
            add_resolution_template=True,
            add_duration_template=True,
        )
        self._token_cache[prompt] = list(ids)
        return list(ids)

    def _encode_target(self, video: torch.Tensor) -> torch.Tensor:
        # Target VAE features are labels.  Freezing their encode graph saves a
        # large amount of activation memory; the decode path below remains
        # differentiable with respect to Cosmos' predicted latent.
        with torch.no_grad():
            return self._pipeline_helpers._encode_video(video).contiguous().float()  # noqa: SLF001

    def _decode(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        dtype = self.vae.dtype
        mean = self.vae_latents_mean.to(device=normalized_latent.device, dtype=dtype)
        inv_std = self.vae_latents_inv_std.to(device=normalized_latent.device, dtype=dtype)
        raw = normalized_latent.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = self.vae.decode(raw).sample
        return decoded[:, :, -1].float().clamp(-1.0, 1.0)

    def _one(
        self,
        current_image: torch.Tensor,
        target_video: torch.Tensor,
        prompt: str,
    ) -> WorldModelForwardOutput:
        device = current_image.device
        # Wrappers such as DDP do not guarantee a public ``dtype`` property.
        transformer_dtype = getattr(self.transformer, "dtype", None)
        if transformer_dtype is None:
            transformer_dtype = next(self.transformer.parameters()).dtype
        video = target_video.unsqueeze(0).float()
        if video.shape[-2:] != (self.resolution, self.resolution):
            batch, channels, frames, height, width = video.shape
            video = (
                F.interpolate(
                    video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width),
                    size=(self.resolution, self.resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .reshape(batch, frames, channels, self.resolution, self.resolution)
                .permute(0, 2, 1, 3, 4)
            )
        current = F.interpolate(
            current_image.unsqueeze(0).float(),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        video = video.clone()
        video[:, :, 0] = current

        clean = self._encode_target(video.to(dtype=transformer_dtype))
        latent_frames = clean.shape[2]
        condition_mask = torch.zeros(latent_frames, 1, 1, device=device, dtype=torch.float32)
        condition_mask[0] = 1.0
        sigma = sample_waver_sigmas(1, device)
        noise = torch.randn_like(clean, dtype=torch.float32)
        xt, velocity_target, sigma_eff = rectified_flow_interpolate(clean, noise, sigma, condition_mask)

        text_segment = self._pipeline_helpers._prepare_text_segment(self._prompt_ids(prompt), device=device)  # noqa: SLF001
        condition_indexes = [0]
        vision_segment = self._pipeline_helpers._prepare_vision_segment(  # noqa: SLF001
            input_vision_tokens=xt,
            has_image_condition=True,
            mrope_offset=text_segment["vision_start_temporal_offset"],
            vision_fps=self.fps,
            curr=text_segment["und_len"],
            device=device,
            condition_frame_indexes=condition_indexes,
        )
        packed = {
            **text_segment,
            **vision_segment,
            "position_ids": torch.cat([text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]], dim=1),
            "sequence_length": text_segment["und_len"] + vision_segment["num_vision_tokens"],
        }
        timestep = float(sigma.item() * 1000.0)
        vision_timesteps = torch.full(
            (vision_segment["num_noisy_vision_tokens"],), timestep, device=device, dtype=torch.float32
        )
        predictions, _, _ = self.transformer(
            input_ids=packed["input_ids"],
            text_indexes=packed["text_indexes"],
            position_ids=packed["position_ids"],
            und_len=packed["und_len"],
            sequence_length=packed["sequence_length"],
            vision_tokens=[xt.to(dtype=transformer_dtype)],
            vision_token_shapes=packed["vision_token_shapes"],
            vision_sequence_indexes=packed["vision_sequence_indexes"],
            vision_mse_loss_indexes=packed["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=packed["vision_noisy_frame_indexes"],
        )
        predicted_velocity = predictions[0].float()
        noisy_mask = (1.0 - condition_mask).view(1, 1, latent_frames, 1, 1)
        active = (
            noisy_mask.sum() * predicted_velocity.shape[1] * predicted_velocity.shape[3] * predicted_velocity.shape[4]
        )
        flow_loss = ((predicted_velocity - velocity_target).square() * noisy_mask).sum() / active.clamp(min=1)

        predicted_clean = xt - sigma_eff * predicted_velocity
        predicted_clean = condition_mask.view(1, 1, latent_frames, 1, 1) * clean + noisy_mask * predicted_clean
        subgoal = self._decode(predicted_clean)
        oracle = video[:, :, -1].float()
        if oracle.shape[-2:] != subgoal.shape[-2:]:
            oracle = F.interpolate(
                oracle, size=subgoal.shape[-2:], mode="bilinear", align_corners=False, antialias=True
            )
        reconstruction_loss = F.l1_loss(subgoal, oracle)
        return WorldModelForwardOutput(subgoal, flow_loss, reconstruction_loss, sigma)

    def forward(
        self,
        current_image: torch.Tensor,
        target_video: torch.Tensor,
        prompts: list[str],
    ) -> WorldModelForwardOutput:
        batch = current_image.shape[0]
        if target_video.shape[0] != batch or len(prompts) != batch:
            raise ValueError("current_image, target_video and prompts must have equal batch size")
        outputs = [self._one(current_image[i], target_video[i], prompts[i]) for i in range(batch)]
        return WorldModelForwardOutput(
            subgoal_image=torch.cat([item.subgoal_image for item in outputs], dim=0),
            flow_matching_loss=torch.stack([item.flow_matching_loss for item in outputs]).mean(),
            reconstruction_loss=torch.stack([item.reconstruction_loss for item in outputs]).mean(),
            sigma=torch.cat([item.sigma for item in outputs], dim=0),
        )
