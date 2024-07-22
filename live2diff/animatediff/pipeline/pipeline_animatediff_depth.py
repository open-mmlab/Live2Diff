# Adapted from https://github.com/open-mmlab/PIA/blob/main/animatediff/pipelines/i2v_pipeline.py

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.depth_utils import MidasDetector
from ..models.unet_depth_streaming import UNet3DConditionStreamingModel
from .loader import LoraLoaderWithWarmup


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    input_images: Optional[Union[torch.Tensor, np.ndarray]] = None


class AnimationDepthPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderWithWarmup):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionStreamingModel,
        depth_model: MidasDetector,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            depth_model=depth_model,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.log_denoising_mean = False

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, clip_skip=None
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            text_embeddings = text_embeddings[0]
        else:
            # support ckip skip here, suitable for model based on NAI~
            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            text_embeddings = text_embeddings[-1][-(clip_skip + 1)]
            text_embeddings = self.text_encoder.text_model.final_layer_norm(text_embeddings)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @classmethod
    def build_pipeline(cls, config_path: str, dreambooth: Optional[str] = None):
        """We build pipeline from config path"""
        from omegaconf import OmegaConf

        from ...utils.config import load_config
        from ..converter import load_third_party_checkpoints
        from ..models.unet_depth_streaming import UNet3DConditionStreamingModel

        cfg = load_config(config_path)
        pretrained_model_path = cfg.pretrained_model_path
        unet_additional_kwargs = cfg.get("unet_additional_kwargs", {})
        noise_scheduler_kwargs = cfg.noise_scheduler_kwargs
        third_party_dict = cfg.get("third_party_dict", {})

        noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        vae = vae.to(device="cuda", dtype=torch.bfloat16)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        text_encoder = text_encoder.to(device="cuda", dtype=torch.float16)

        unet = UNet3DConditionStreamingModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs) if unet_additional_kwargs else {},
        )

        motion_module_path = cfg.motion_module_path
        # load motion module to unet
        mm_checkpoint = torch.load(motion_module_path, map_location="cuda")
        if "global_step" in mm_checkpoint:
            print(f"global_step: {mm_checkpoint['global_step']}")
        state_dict = mm_checkpoint["state_dict"] if "state_dict" in mm_checkpoint else mm_checkpoint
        # NOTE: hard code here: remove `grid` from state_dict
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "grid" not in k}

        m, u = unet.load_state_dict(state_dict, strict=False)
        assert len(u) == 0, f"Find unexpected keys ({len(u)}): {u}"
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        unet = unet.to(dtype=torch.float16)
        depth_model = MidasDetector(cfg.depth_model_path).to(device="cuda", dtype=torch.float16)

        pipeline = cls(
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            depth_model=depth_model,
            scheduler=noise_scheduler,
        )
        pipeline = load_third_party_checkpoints(pipeline, third_party_dict, dreambooth)

        return pipeline

    @classmethod
    def build_warmup_unet(cls, config_path: str, dreambooth: Optional[str] = None):
        from omegaconf import OmegaConf

        from ...utils.config import load_config
        from ..converter import load_third_party_unet
        from ..models.unet_depth_warmup import UNet3DConditionWarmupModel

        cfg = load_config(config_path)
        pretrained_model_path = cfg.pretrained_model_path
        unet_additional_kwargs = cfg.get("unet_additional_kwargs", {})
        third_party_dict = cfg.get("third_party_dict", {})

        unet = UNet3DConditionWarmupModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs) if unet_additional_kwargs else {},
        )
        motion_module_path = cfg.motion_module_path
        # load motion module to unet
        mm_checkpoint = torch.load(motion_module_path, map_location="cpu")
        if "global_step" in mm_checkpoint:
            print(f"global_step: {mm_checkpoint['global_step']}")
        state_dict = mm_checkpoint["state_dict"] if "state_dict" in mm_checkpoint else mm_checkpoint
        # NOTE: hard code here: remove `grid` from state_dict
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "grid" not in k}

        m, u = unet.load_state_dict(state_dict, strict=False)
        assert len(u) == 0, f"Find unexpected keys ({len(u)}): {u}"
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        unet = load_third_party_unet(unet, third_party_dict, dreambooth)
        return unet

    def prepare_cache(self, height: int, width: int, denoising_steps_num: int):
        vae = self.vae
        scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.unet.set_info_for_attn(height // scale_factor, width // scale_factor)
        kv_cache_list = self.unet.prepare_cache(denoising_steps_num)
        return kv_cache_list

    def prepare_warmup_unet(self, height: int, width: int, unet):
        vae = self.vae
        scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        unet.set_info_for_attn(height // scale_factor, width // scale_factor)
