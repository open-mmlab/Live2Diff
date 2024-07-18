from typing import Optional

import torch
from diffusers.pipelines import StableDiffusionPipeline
from safetensors import safe_open

from .convert_from_ckpt import convert_ldm_clip_checkpoint, convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from .convert_lora_safetensor_to_diffusers import convert_lora_model_level


def load_third_party_checkpoints(
    pipeline: StableDiffusionPipeline,
    third_party_dict: dict,
    dreambooth_path: Optional[str] = None,
):
    """
    Modified from https://github.com/open-mmlab/PIA/blob/4b1ee136542e807a13c1adfe52f4e8e5fcc65cdb/animatediff/pipelines/i2v_pipeline.py#L165
    """
    vae = third_party_dict.get("vae", None)
    lora_list = third_party_dict.get("lora_list", [])

    dreambooth = dreambooth_path or third_party_dict.get("dreambooth", None)

    text_embedding_dict = third_party_dict.get("text_embedding_dict", {})

    if dreambooth is not None:
        dreambooth_state_dict = {}
        if dreambooth.endswith(".safetensors"):
            with safe_open(dreambooth, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        else:
            dreambooth_state_dict = torch.load(dreambooth, map_location="cpu")
            if "state_dict" in dreambooth_state_dict:
                dreambooth_state_dict = dreambooth_state_dict["state_dict"]
        # load unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
        pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        # load vae from dreambooth (if need)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
        # add prefix for compiled model
        if "_orig_mod" in list(pipeline.vae.state_dict().keys())[0]:
            converted_vae_checkpoint = {f"_orig_mod.{k}": v for k, v in converted_vae_checkpoint.items()}
        pipeline.vae.load_state_dict(converted_vae_checkpoint, strict=True)

        # load text encoder (if need)
        text_encoder_checkpoint = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        if text_encoder_checkpoint:
            pipeline.text_encoder.load_state_dict(text_encoder_checkpoint, strict=False)

    if vae is not None:
        vae_state_dict = {}
        if vae.endswith("safetensors"):
            with safe_open(vae, framework="pt", device="cpu") as f:
                for key in f.keys():
                    vae_state_dict[key] = f.get_tensor(key)
        elif vae.endswith("ckpt") or vae.endswith("pt"):
            vae_state_dict = torch.load(vae, map_location="cpu")
        if "state_dict" in vae_state_dict:
            vae_state_dict = vae_state_dict["state_dict"]

        vae_state_dict = {f"first_stage_model.{k}": v for k, v in vae_state_dict.items()}

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_state_dict, pipeline.vae.config)
        # add prefix for compiled model
        if "_orig_mod" in list(pipeline.vae.state_dict().keys())[0]:
            converted_vae_checkpoint = {f"_orig_mod.{k}": v for k, v in converted_vae_checkpoint.items()}
        pipeline.vae.load_state_dict(converted_vae_checkpoint, strict=True)

    if lora_list:
        for lora_dict in lora_list:
            lora, lora_alpha = lora_dict["lora"], lora_dict["lora_alpha"]
            lora_state_dict = {}
            with safe_open(lora, framework="pt", device="cpu") as file:
                for k in file.keys():
                    lora_state_dict[k] = file.get_tensor(k)
            pipeline.unet, pipeline.text_encoder = convert_lora_model_level(
                lora_state_dict,
                pipeline.unet,
                pipeline.text_encoder,
                alpha=lora_alpha,
            )
            print(f'Add LoRA "{lora}":{lora_alpha} to pipeline.')

    if text_embedding_dict is not None:
        from diffusers.loaders import TextualInversionLoaderMixin

        assert isinstance(
            pipeline, TextualInversionLoaderMixin
        ), "Pipeline must inherit from TextualInversionLoaderMixin."

        for token, embedding_path in text_embedding_dict.items():
            pipeline.load_textual_inversion(embedding_path, token)

    return pipeline


def load_third_party_unet(unet, third_party_dict: dict, dreambooth_path: Optional[str] = None):
    lora_list = third_party_dict.get("lora_list", [])
    dreambooth = dreambooth_path or third_party_dict.get("dreambooth", None)

    if dreambooth is not None:
        dreambooth_state_dict = {}
        if dreambooth.endswith(".safetensors"):
            with safe_open(dreambooth, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        else:
            dreambooth_state_dict = torch.load(dreambooth, map_location="cpu")
            if "state_dict" in dreambooth_state_dict:
                dreambooth_state_dict = dreambooth_state_dict["state_dict"]
        # load unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, unet.config)
        unet.load_state_dict(converted_unet_checkpoint, strict=False)

    if lora_list:
        for lora_dict in lora_list:
            lora, lora_alpha = lora_dict["lora"], lora_dict["lora_alpha"]
            lora_state_dict = {}

            with safe_open(lora, framework="pt", device="cpu") as file:
                for k in file.keys():
                    if "text" not in k:
                        lora_state_dict[k] = file.get_tensor(k)
            unet, _ = convert_lora_model_level(
                lora_state_dict,
                unet,
                None,
                alpha=lora_alpha,
            )
            print(f'Add LoRA "{lora}":{lora_alpha} to Warmup UNet.')

    return unet
