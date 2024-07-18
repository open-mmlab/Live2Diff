from typing import *

import torch
from polygraphy import cuda

from live2diff.animatediff.models.unet_depth_streaming import UNet3DConditionStreamingOutput

from .utilities import Engine


try:
    from diffusers.models.autoencoder_tiny import AutoencoderTinyOutput
except ImportError:
    from dataclasses import dataclass

    from diffusers.utils import BaseOutput

    @dataclass
    class AutoencoderTinyOutput(BaseOutput):
        """
        Output of AutoencoderTiny encoding method.

        Args:
            latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

        """

        latents: torch.Tensor


try:
    from diffusers.models.vae import DecoderOutput
except ImportError:
    from dataclasses import dataclass

    from diffusers.utils import BaseOutput

    @dataclass
    class DecoderOutput(BaseOutput):
        r"""
        Output of decoding method.

        Args:
            sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                The decoded output sample from the last layer of the model.
        """

        sample: torch.FloatTensor


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def encode(self, images: torch.Tensor, **kwargs):
        self.encoder.allocate_buffers(
            shape_dict={
                "images": images.shape,
                "latent": (
                    images.shape[0],
                    4,
                    images.shape[2] // self.vae_scale_factor,
                    images.shape[3] // self.vae_scale_factor,
                ),
            },
            device=images.device,
        )
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        self.decoder.allocate_buffers(
            shape_dict={
                "latent": latent.shape,
                "images": (
                    latent.shape[0],
                    3,
                    latent.shape[2] * self.vae_scale_factor,
                    latent.shape[3] * self.vae_scale_factor,
                ),
            },
            device=latent.device,
        )
        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class UNet2DConditionModelDepthEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.init_profiler()

        self.engine.load()
        self.engine.activate(profiler=self.profiler)
        self.has_allocated = False

    def init_profiler(self):
        import tensorrt

        class Profiler(tensorrt.IProfiler):
            def __init__(self):
                tensorrt.IProfiler.__init__(self)

            def report_layer_time(self, layer_name, ms):
                print(f"{layer_name}: {ms} ms")

        self.profiler = Profiler()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temporal_attention_mask: torch.Tensor,
        depth_sample: torch.Tensor,
        kv_cache: List[torch.Tensor],
        pe_idx: torch.Tensor,
        update_idx: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        feed_dict = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "temporal_attention_mask": temporal_attention_mask,
            "depth_sample": depth_sample,
            "pe_idx": pe_idx,
            "update_idx": update_idx,
        }
        for idx, cache in enumerate(kv_cache):
            feed_dict[f"kv_cache_{idx}"] = cache
        shape_dict = {k: v.shape for k, v in feed_dict.items()}

        if not self.has_allocated:
            self.engine.allocate_buffers(
                shape_dict=shape_dict,
                device=latent_model_input.device,
            )
            self.has_allocated = True

        output = self.engine.infer(
            feed_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )

        noise_pred = output["latent"]
        kv_cache = [output[f"kv_cache_out_{idx}"] for idx in range(len(kv_cache))]
        return UNet3DConditionStreamingOutput(sample=noise_pred, kv_cache=kv_cache)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class MidasEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()
        self.has_allocated = False
        self.default_batch_size = 1

    def __call__(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Any:
        if not self.has_allocated or images.shape[0] != self.default_batch_size:
            bz = images.shape[0]
            self.engine.allocate_buffers(
                shape_dict={
                    "images": (bz, 3, 384, 384),
                    "depth_map": (bz, 384, 384),
                },
                device=images.device,
            )
            self.has_allocated = True
            self.default_batch_size = bz

        depth_map = self.engine.infer(
            {
                "images": images,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["depth_map"]  #  (1, 384, 384)

        return depth_map

    def norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
