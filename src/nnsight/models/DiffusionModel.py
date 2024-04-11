from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import DiffusionPipeline, SchedulerMixin
from PIL import Image
from transformers import BatchEncoding, CLIPTextModel, CLIPTokenizer

from .mixins import GenerationMixin
from .NNsightModel import NNsight
from torch._guards import detect_fake_mode
from .. import util

class Diffuser(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.pipeline = DiffusionPipeline.from_pretrained(*args, **kwargs)

        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module):
                setattr(self, key, value)

        self.tokenizer = self.pipeline.tokenizer
                
    @torch.no_grad()
    def scan(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):

        # 0. Default height and width to unet
        height = (
            height
            or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        )
        width = (
            width
            or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        )

        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            "meta",
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps = self.pipeline.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            "meta",
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        
        # predict the noise residual
        noise_pred = self.pipeline.unet(
            latent_model_input,
            0,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if not output_type == "latent":
            image = self.pipeline.vae.decode(
                latents / self.pipeline.vae.config.scaling_factor, return_dict=False
            )[0]
        else:
            image = latents
            has_nsfw_concept = None


class DiffusionModel(GenerationMixin, NNsight):
    def __init__(self, *args, **kwargs) -> None:
        
        self._model: Diffuser = None

        super().__init__(*args, **kwargs)

    def _load(self, repo_id: str, device_map=None, **kwargs) -> Diffuser:

        if self._model is None:

            model = Diffuser(
                repo_id,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=False,
                **kwargs,
            )

            return model

        model = Diffuser(repo_id, device_map=device_map, **kwargs)

        return model

    def _prepare_inputs(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        return (inputs,), len(inputs)

    # def _forward(self, inputs, *args, n_imgs=1, img_size=512, **kwargs) -> None:
    #     text_tokens, latents = inputs

    #     text_embeddings = self.meta_model.get_text_embeddings(text_tokens, n_imgs)

    #     latents = torch.cat([latents] * 2).to("meta")

    #     return self.meta_model.unet(
    #         latents,
    #         torch.zeros((1,), device="meta"),
    #         encoder_hidden_states=text_embeddings,
    #     ).sample

    def _batch_inputs(self,         batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,) -> torch.Tensor:
        
        if batched_inputs is None:
            
            return prepared_inputs
        
        return batched_inputs + prepared_inputs
    
    def _execute_forward(self, prepared_inputs: Any, *args, **kwargs):

        device = next(self._model.parameters()).device

        return self._model.unet(
            prepared_inputs,
            *args
            **kwargs,
        )

    def _execute_generate(
        self, prepared_inputs: Any, *args, **kwargs
    ):
        device = next(self._model.parameters()).device
        
        if detect_fake_mode(prepared_inputs):
            
            output = self._model.scan(*prepared_inputs)
            
        else:

            output = self._model.pipeline(prepared_inputs, *args, **kwargs)
            
        output = self._model(output)

        return output
