import os
import sys


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

import torch
from config import Args
from PIL import Image
from pydantic import BaseModel, Field

from live2diff.utils.config import load_config
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper


default_prompt = "masterpiece, best quality, felted, 1man with glasses, glasses, play with his pen"

page_content = """<h1 class="text-3xl font-bold">Live2Diff: </h1>
<h2 class="text-xl font-bold">Live Stream Translation via Uni-directional Attention in Video Diffusion Models</h2>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/open-mmlab/Live2Diff"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Live2Diff
</a>
pipeline using
    <a
    href="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">LCM-LoRA</a
    > with a MJPEG stream server.
</p>
"""


WARMUP_FRAMES = 8
WINDOW_SIZE = 16


class Pipeline:
    class Info(BaseModel):
        name: str = "Live2Diff"
        input_mode: str = "image"
        page_content: str = page_content

    def build_input_params(self, default_prompt: str = default_prompt, width=512, height=512):
        class InputParams(BaseModel):
            prompt: str = Field(
                default_prompt,
                title="Prompt",
                field="textarea",
                id="prompt",
            )
            width: int = Field(
                512,
                min=2,
                max=15,
                title="Width",
                disabled=True,
                hide=True,
                id="width",
            )
            height: int = Field(
                512,
                min=2,
                max=15,
                title="Height",
                disabled=True,
                hide=True,
                id="height",
            )

        return InputParams

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        config_path = args.config

        cfg = load_config(config_path)
        prompt = args.prompt or cfg.prompt or default_prompt

        self.InputParams = self.build_input_params(default_prompt=prompt)
        params = self.InputParams()

        num_inference_steps = args.num_inference_steps or cfg.get("num_inference_steps", None)
        strength = args.strength or cfg.get("strength", None)
        t_index_list = args.t_index_list or cfg.get("t_index_list", None)

        self.stream = StreamAnimateDiffusionDepthWrapper(
            few_step_model_type="lcm",
            config_path=config_path,
            cfg_type="none",
            strength=strength,
            num_inference_steps=num_inference_steps,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            acceleration=args.acceleration,
            do_add_noise=True,
            output_type="pil",
            enable_similar_image_filter=True,
            similar_image_filter_threshold=0.98,
            use_denoising_batch=True,
            use_tiny_vae=True,
            seed=args.seed,
            engine_dir=args.engine_dir,
        )

        self.last_prompt = prompt

        self.warmup_frame_list = []
        self.has_prepared = False

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        prompt = params.prompt
        if prompt != self.last_prompt:
            self.last_prompt = prompt
            self.warmup_frame_list.clear()

        if len(self.warmup_frame_list) < WARMUP_FRAMES:
            # from PIL import Image
            self.warmup_frame_list.append(self.stream.preprocess_image(params.image))

        elif len(self.warmup_frame_list) == WARMUP_FRAMES and not self.has_prepared:
            warmup_frames = torch.stack(self.warmup_frame_list)
            self.stream.prepare(
                warmup_frames=warmup_frames,
                prompt=prompt,
                guidance_scale=1,
            )
            self.has_prepared = True

        if self.has_prepared:
            image_tensor = self.stream.preprocess_image(params.image)
            output_image = self.stream(image=image_tensor)
            return output_image
        else:
            return Image.new("RGB", (params.width, params.height))
