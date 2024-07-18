import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import LCMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from einops import rearrange

from live2diff.image_filter import SimilarImageFilter

from .animatediff.pipeline import AnimationDepthPipeline


WARMUP_FRAMES = 8
WINDOW_SIZE = 16


class StreamAnimateDiffusionDepth:
    def __init__(
        self,
        pipe: AnimationDepthPipeline,
        num_inference_steps: int,
        t_index_list: Optional[List[int]] = None,
        strength: Optional[float] = None,
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        clip_skip: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "none",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.pipe = pipe

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.clip_skip = clip_skip

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        if strength is not None:
            t_index_list, timesteps = self.get_timesteps(num_inference_steps, strength, self.device)
            print(
                f"Generate t_index_list: {t_index_list} via "
                f"num_inference_steps: {num_inference_steps}, strength: {strength}"
            )
            self.timesteps = timesteps
        else:
            print(
                f"t_index_list is passed: {t_index_list}. "
                f"Number Inference Steps: {num_inference_steps}, "
                f"equivalents to strength {1 - t_index_list[0] / num_inference_steps}."
            )
            self.timesteps = self.scheduler.timesteps.to(self.device)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)
        self.strength = strength

        assert cfg_type == "none", f'cfg_type must be "none" for now, but got {cfg_type}.'
        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.depth_detector = pipe.depth_model
        self.inference_time_ema = 0
        self.depth_time_ema = 0
        self.inference_time_list = []
        self.depth_time_list = []
        self.mask_shift = 1

        self.is_tensorrt = False

    def prepare_cache(self, height, width, denoising_steps_num):
        kv_cache_list = self.pipe.prepare_cache(
            height=height,
            width=width,
            denoising_steps_num=denoising_steps_num,
        )
        self.pipe.prepare_warmup_unet(height=height, width=width, unet=self.unet_warmup)
        self.kv_cache_list = kv_cache_list

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(device)
        t_index = list(range(len(timesteps)))

        return t_index, timesteps

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict,
            adapter_name,
            **kwargs,
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(
        self,
        threshold: float = 0.98,
        max_skip_frame: float = 10,
    ) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        warmup_frames: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: int = 2,
    ) -> None:
        """
        Forward warm-up frames and fill the buffer
        images: [warmup_size, 3, h, w] in [0, 1]
        """

        if generator is None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)
        else:
            self.generator = generator
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    1,  # for video
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )

            self.depth_latent_buffer = torch.zeros_like(self.x_t_latent_buffer)
        else:
            self.x_t_latent_buffer = None
            self.depth_latent_buffer = None

        self.attn_bias, self.pe_idx, self.update_idx = self.initialize_attn_bias_pe_and_update_idx()

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            clip_skip=self.clip_skip,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
            self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, WARMUP_FRAMES, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1, 1).to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1, 1).to(dtype=self.dtype, device=self.device)
        )
        # print(self.c_skip)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        # do warmup
        # 1. encode images
        warmup_x_list = []
        for f in warmup_frames:
            x = self.image_processor.preprocess(f, self.height, self.width)
            warmup_x_list.append(x.to(device=self.device, dtype=self.dtype))
        warmup_x = torch.cat(warmup_x_list, dim=0)  # [warmup_size, c, h, w]
        warmup_x_t = self.encode_image(warmup_x)
        x_t_latent = rearrange(warmup_x_t, "f c h w -> c f h w")[None, ...]
        depth_latent = self.encode_depth(warmup_x)
        depth_latent = rearrange(depth_latent, "f c h w -> c f h w")[None, ...]

        # 2. run warmup denoising
        self.unet_warmup = self.unet_warmup.to(device="cuda", dtype=self.dtype)
        warmup_prompt = self.prompt_embeds[0:1]
        for idx, t in enumerate(self.sub_timesteps_tensor):
            t = t.view(1).repeat(x_t_latent.shape[0])

            output_t = self.unet_warmup(
                x_t_latent,
                t,
                temporal_attention_mask=None,
                depth_sample=depth_latent,
                encoder_hidden_states=warmup_prompt,
                kv_cache=[cache[idx] for cache in self.kv_cache_list],
                return_dict=True,
            )
            model_pred = output_t["sample"]
            x_0_pred = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if idx < len(self.sub_timesteps_tensor) - 1:
                # x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred

                x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred + self.beta_prod_t_sqrt[
                    idx + 1
                ] * torch.randn_like(x_0_pred, device=self.device, dtype=self.dtype)

        self.unet_warmup = self.unet_warmup.to(device="cpu")
        x_0_pred = rearrange(x_0_pred, "b c f h w -> b f c h w")[0]  # [f, c, h, w]
        denoisied_frame = self.decode_image(x_0_pred)

        self.warmup_engine()

        return denoisied_frame

    def warmup_engine(self):
        """Warmup tensorrt engine."""

        if not self.is_tensorrt:
            return

        print("Warmup TensorRT engine.")
        pseudo_latent = self.init_noise[:, :, 0:1, ...]
        for _ in range(self.batch_size):
            self.unet(
                pseudo_latent,
                self.sub_timesteps_tensor,
                depth_sample=pseudo_latent,
                encoder_hidden_states=self.prompt_embeds,
                temporal_attention_mask=self.attn_bias,
                kv_cache=self.kv_cache_list,
                pe_idx=self.pe_idx,
                update_idx=self.update_idx,
                return_dict=True,
            )
        print("Warmup TensorRT engine finished.")

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch

        return denoised_batch

    def initialize_attn_bias_pe_and_update_idx(self):
        attn_mask = torch.zeros((self.denoising_steps_num, WINDOW_SIZE), dtype=torch.bool, device=self.device)
        attn_mask[:, :WARMUP_FRAMES] = True
        attn_mask[0, WARMUP_FRAMES] = True
        attn_bias = torch.zeros_like(attn_mask, dtype=self.dtype)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        pe_idx = torch.arange(WINDOW_SIZE).unsqueeze(0).repeat(self.denoising_steps_num, 1).cuda()
        update_idx = torch.ones(self.denoising_steps_num, dtype=torch.int64, device=self.device) * WARMUP_FRAMES
        update_idx[1] = WARMUP_FRAMES + 1

        return attn_bias, pe_idx, update_idx

    def update_attn_bias(self, attn_bias, pe_idx, update_idx):
        """
        attn_bias: (timesteps, prev_len), init value: [[0, 0, 0, inf], [0, 0, inf, inf]]
        pe_idx: (timesteps, prev_len), init value: [[0, 1, 2, 3], [0, 1, 2, 3]]
        update_idx: (timesteps, ), init value: [2, 1]
        """

        for idx in range(self.denoising_steps_num):
            # update pe_idx and update_idx based on attn_bias from last iteration
            if torch.isinf(attn_bias[idx]).any():
                # some position not filled, do not change pe
                # some position not filled, fill the last position
                update_idx[idx] = (attn_bias[idx] == 0).sum()
            else:
                # all position are filled, roll pe
                pe_idx[idx, WARMUP_FRAMES:] = pe_idx[idx, WARMUP_FRAMES:].roll(shifts=1, dims=0)
                # all position are filled, fill the position with largest PE
                update_idx[idx] = pe_idx[idx].argmax()

            num_unmask = (attn_bias[idx] == 0).sum()
            attn_bias[idx, : min(num_unmask + 1, WINDOW_SIZE)] = 0

        return attn_bias, pe_idx, update_idx

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        output = self.unet(
            x_t_latent_plus_uc,
            t_list,
            depth_sample=depth_latent,
            encoder_hidden_states=self.prompt_embeds,
            temporal_attention_mask=self.attn_bias,
            kv_cache=self.kv_cache_list,
            pe_idx=self.pe_idx,
            update_idx=self.update_idx,
            return_dict=True,
        )
        model_pred = output["sample"]
        kv_cache_list = output["kv_cache"]
        self.kv_cache_list = kv_cache_list

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
                self.stock_noise = init_noise + delta_x

        else:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        image_tensors: [f, c, h, w]
        """
        # num_frames = image_tensors.shape[2]
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        noise = torch.randn(
            img_latent.shape,
            device=img_latent.device,
            dtype=img_latent.dtype,
            generator=self.generator,
        )
        x_t_latent = self.add_noise(img_latent, noise, 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        """
        x_0_pred: [f, c, h, w]
        """
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]
        return output_latent.clip(-1, 1)

    def encode_depth(self, image_tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        image_tensor: [f, c, h, w], [-1, 1]
        """
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.depth_detector.dtype,
        )
        # depth_map = self.depth_detector(image_tensors)
        # depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map_norm = depth_map_norm[:, None].repeat(1, 3, 1, 1) * 2 - 1
        # depth_latent = retrieve_latents(self.vae.encode(depth_map_norm.to(dtype=self.vae.dtype)), self.generator)
        # depth_latent = depth_latent * self.vae.config.scaling_factor
        # return depth_latent

        # preprocess
        h, w = image_tensors.shape[2], image_tensors.shape[3]
        images_input = F.interpolate(image_tensors, (384, 384), mode="bilinear", align_corners=False)
        # forward
        depth_map = self.depth_detector(images_input)
        # postprocess
        depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map_norm = depth_map_norm[:, None].repeat(1, 3, 1, 1) * 2 - 1
        depth_map_norm = F.interpolate(depth_map_norm, (h, w), mode="bilinear", align_corners=False)
        # encode
        depth_latent = retrieve_latents(self.vae.encode(depth_map_norm.to(dtype=self.vae.dtype)), self.generator)
        depth_latent = depth_latent * self.vae.config.scaling_factor
        return depth_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer
        prev_depth_latent_batch = self.depth_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                depth_latent = torch.cat((depth_latent, prev_depth_latent_batch), dim=0)

                self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, depth_latent, t_list)
            self.attn_bias, self.pe_idx, self.update_idx = self.update_attn_bias(
                self.attn_bias, self.pe_idx, self.update_idx
            )

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    # self.x_t_latent_buffer = (
                    #     self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    #     + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    # )
                    self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + self.beta_prod_t_sqrt[
                        1:
                    ] * torch.randn_like(x_0_pred_batch[:-1])
                else:
                    self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                self.depth_latent_buffer = depth_latent[:-1]
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(x_t_latent, depth_latent, t, idx)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(x_0_pred, device=self.device, dtype=self.dtype)
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.image_processor.preprocess(x, self.height, self.width).to(device=self.device, dtype=self.dtype)
        if self.similar_image_filter:
            x = self.similar_filter(x)
            if x is None:
                time.sleep(self.inference_time_ema)
                return self.prev_image_result
        x_t_latent = self.encode_image(x)

        start_depth = torch.cuda.Event(enable_timing=True)
        end_depth = torch.cuda.Event(enable_timing=True)
        start_depth.record()
        depth_latent = self.encode_depth(x)
        end_depth.record()
        torch.cuda.synchronize()
        depth_time = start_depth.elapsed_time(end_depth) / 1000

        x_t_latent = x_t_latent.unsqueeze(2)
        depth_latent = depth_latent.unsqueeze(2)
        x_0_pred_out = self.predict_x0_batch(x_t_latent, depth_latent)  # [1, c, 1, h, w]
        x_0_pred_out = rearrange(x_0_pred_out, "b c f h w -> (b f) c h w")
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        self.depth_time_ema = 0.9 * self.depth_time_ema + 0.1 * depth_time
        self.inference_time_list.append(inference_time)
        self.depth_time_list.append(depth_time)
        return x_output

    def load_warmup_unet(self, config):
        unet_warmup = self.pipe.build_warmup_unet(config)
        self.unet_warmup = unet_warmup
        self.pipe.unet_warmup = unet_warmup
        print("Load Warmup UNet.")
