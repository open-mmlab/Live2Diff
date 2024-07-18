import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from .builder import EngineBuilder
from .models import BaseModel


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


def compile_engine(
    torch_model: nn.Module,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    builder = EngineBuilder(
        model_data,
        torch_model,
        device=torch.device("cuda"),
    )
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
