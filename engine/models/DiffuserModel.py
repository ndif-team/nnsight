from __future__ import annotations

from typing import Any, List, Union

import diffusers
import torch
from diffusers import AutoencoderKL, SchedulerMixin, UNet2DConditionModel
from PIL import Image
from torch.utils.hooks import RemovableHandle
from transformers import CLIPTextModel, CLIPTokenizer

from .AbstractModel import AbstractModel


class Diffuser(torch.nn.Module):
    def __init__(
        self, repoid_or_path, tokenizer: CLIPTokenizer, *args, **kwargs
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            repoid_or_path, *args, **kwargs, subfolder="vae"
        )
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            repoid_or_path, *args, **kwargs, subfolder="unet"
        )
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            repoid_or_path, *args, **kwargs, subfolder="text_encoder"
        )

    def get_text_embeddings(self, text_tokens, n_imgs) -> torch.Tensor:
        text_ids = text_tokens.input_ids.to(self.text_encoder.device)

        text_embeddings = self.text_encoder(text_ids)[0]

        unconditional_tokens = self.text_tokenize([""] * len(text_ids))

        unconditional_ids = unconditional_tokens.input_ids.to(self.text_encoder.device)

        unconditional_embeddings = self.text_encoder(unconditional_ids)[0]

        text_embeddings = torch.repeat_interleave(
            torch.cat([unconditional_embeddings, text_embeddings]), n_imgs, dim=0
        )

        return text_embeddings

    def text_tokenize(self, prompts):
        return self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def text_detokenize(self, tokens):
        return [
            self.tokenizer.decode(token)
            for token in tokens
            if token != self.tokenizer.vocab_size - 1
        ]

    def get_noise(self, batch_size, img_size) -> torch.Tensor:
        return torch.randn(
            (batch_size, self.unet.config.in_channels, img_size // 8, img_size // 8)
        )

    def get_initial_latents(self, n_imgs, img_size, n_prompts) -> torch.Tensor:
        latents = self.get_noise(n_imgs, img_size).repeat(n_prompts, 1, 1, 1)

        return latents

    def decode(self, latents):
        return self.vae.decode(1 / 0.18215 * latents).sample

    def encode(self, tensors):
        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def predict_noise(
        self, scheduler, iteration, latents, text_embeddings, guidance_scale=7.5
    ):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = scheduler.scale_model_input(latents, scheduler.timesteps[iteration])

        # predict the noise residual
        noise_prediction = self.unet(
            latents,
            scheduler.timesteps[iteration],
            encoder_hidden_states=text_embeddings,
        ).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * (
            noise_prediction_text - noise_prediction_uncond
        )

        return noise_prediction

    def diffusion(
        self,
        scheduler,
        latents,
        text_embeddings,
        end_iteration=1000,
        start_iteration=0,
        **kwargs,
    ):
        for iteration in range(start_iteration, end_iteration):
            noise_pred = self.predict_noise(
                scheduler, iteration, latents, text_embeddings, **kwargs
            )

            # compute the previous noisy sample x_t -> x_t-1
            output = scheduler.step(noise_pred, scheduler.timesteps[iteration], latents)

            latents = output.prev_sample

        return latents


class DiffuserModel(AbstractModel):
    def __init__(self, *args, **kwargs) -> None:
        self.local_model: Diffuser = None
        self.meta_model: Diffuser = None
        self.tokenizer: CLIPTokenizer = None

        super().__init__(*args, **kwargs)

    def _register_increment_hook(self, hook) -> RemovableHandle:
        return self.local_model.unet.register_forward_hook(hook)

    def _load_meta(
        self, repoid_or_path, *args, device="cpu", **kwargs
    ) -> torch.nn.Module:
        self.tokenizer = CLIPTokenizer.from_pretrained(
            repoid_or_path, *args, **kwargs, subfolder="tokenizer"
        )

        return Diffuser(repoid_or_path, self.tokenizer, *args, **kwargs)

    def _load_local(
        self, repoid_or_path, *args, device="cpu", **kwargs
    ) -> torch.nn.Module:
        return Diffuser(repoid_or_path, self.tokenizer, *args, **kwargs).to(device)

    def _prepare_inputs(
        self,
        inputs,
        n_imgs=1,
        img_size=512,
    ) -> Any:
        if not isinstance(inputs, list):
            inputs = [inputs]

        latents = self.meta_model.get_initial_latents(n_imgs, img_size, len(inputs))

        text_tokens = self.meta_model.text_tokenize(inputs)

        return text_tokens, latents

    def _run_meta(self, inputs, *args, n_imgs=1, img_size=512, **kwargs) -> None:
        text_tokens, latents = self._prepare_inputs(
            inputs, n_imgs=n_imgs, img_size=img_size
        )

        text_embeddings = self.meta_model.get_text_embeddings(text_tokens, n_imgs)

        latents = torch.cat([latents] * 2).to("meta")

        self.meta_model.unet(
            latents,
            torch.zeros((1,), device="meta"),
            encoder_hidden_states=text_embeddings,
        ).sample

        self.meta_model.vae.decode(latents)

        return text_tokens.input_ids

    def _run_local(self, inputs, *args, n_imgs=1, img_size=512, **kwargs) -> None:
        text_tokens, latents = self._prepare_inputs(
            inputs, n_imgs=n_imgs, img_size=img_size
        )

        text_embeddings = self.meta_model.get_text_embeddings(text_tokens, n_imgs)

        latents = torch.cat([latents] * 2).to("meta")

        return self.meta_model.unet(
            latents,
            torch.zeros((1,), device="meta"),
            encoder_hidden_states=text_embeddings,
        ).sample

    def _generation(
        self,
        inputs,
        *args,
        n_steps=20,
        scheduler="LMSDiscreteScheduler",
        n_imgs=1,
        img_size=512,
        **kwargs,
    ) -> None:
        text_tokens, latents = self._prepare_inputs(
            inputs, n_imgs=n_imgs, img_size=img_size
        )

        text_embeddings = self.local_model.get_text_embeddings(text_tokens, n_imgs)

        if isinstance(scheduler, str):
            scheduler: SchedulerMixin = getattr(diffusers, scheduler).from_pretrained(
                self.repoid_or_path, subfolder="scheduler"
            )
        scheduler.set_timesteps(n_steps)

        latents = latents * scheduler.init_noise_sigma

        latents = self.local_model.diffusion(
            scheduler,
            latents.to(self.local_model.unet.device),
            text_embeddings.to(self.local_model.unet.device),
            *args,
            **kwargs,
            end_iteration=n_steps,
        )

        latents = (1 / 0.18215) * latents

        return self.local_model.vae.decode(latents).sample

    def to_image(self, latents) -> List[Image.Image]:
        """
        Function to convert latents to images
        """

        image = (latents / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images
