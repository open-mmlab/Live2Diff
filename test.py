import os
from typing import Dict, List, Literal, Optional

import fire
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from live2diff.utils.config import load_config
from live2diff.utils.io import read_video_frames, save_videos_grid
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str,
    config_path: str,
    prompt: Optional[str] = None,
    prompt_template: Optional[str] = None,
    output: str = os.path.join("outputs", "output.mp4"),
    dreambooth_path: Optional[str] = None,
    lora_dict: Optional[Dict[str, float]] = None,
    height: int = 512,
    width: int = 512,
    max_frames: int = -1,
    num_inference_steps: Optional[int] = None,
    t_index_list: Optional[List[int]] = None,
    strength: Optional[float] = None,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    enable_similar_image_filter: bool = False,
    few_step_model_type: str = "lcm",
    enable_tiny_vae: bool = True,
    fps: int = 16,
    save_input: bool = True,
    seed: int = 42,
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    input : str
        The input video name or name of video frames to load images from.
    config_path: str, optional
        The path to config file.
    prompt : str
        The prompt to generate images from.
    prompt_template: str, optional
        The template for specific dreambooth / LoRA. If not None, `{}` must be contained,
        and the prompt used for inference will be `prompt_template.format(prompt)`.
    output : str, optional
        The output video name to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: `python main.py --lora_dict='{"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}'`
    height: int, optional
        The height of the image, by default 512.
    width: int, optional
        The width of the image, by default 512.
    max_frames : int, optional
        The maximum number of frames to process, by default -1.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default True.
    fps: int
        The fps of the output video, by default 16.
    save_input: bool, optional
        Whether to save the input video or not, by default True.
        If true, the input video will be saved as `output` + "_inp.mp4".
    seed : int, optional
        The seed, by default 42. if -1, use random seed.
    """

    if os.path.isdir(input):
        video = read_video_frames(input) / 255
    elif input.endswith(".mp4"):
        reader = VideoReader(input)
        total_frames = len(reader)
        frame_indices = np.arange(total_frames)
        video = reader.get_batch(frame_indices).asnumpy() / 255
        video = torch.from_numpy(video)
    elif input.endswith(".gif"):
        video_frames = []
        image = Image.open(input)
        for frames in range(image.n_frames):
            image.seek(frames)
            video_frames.append(np.array(image.convert("RGB")))
        video = torch.from_numpy(np.array(video_frames)) / 255

    video = video[2:]

    height = int(height // 8 * 8)
    width = int(width // 8 * 8)

    trans = transforms.Compose(
        [
            transforms.Resize(min(height, width), antialias=True),
            transforms.CenterCrop((height, width)),
        ]
    )
    video = trans(video.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    if max_frames > 0:
        video = video[: min(max_frames, len(video))]
        print(f"Clipping video to {len(video)} frames.")

    cfg = load_config(config_path)
    print("Inference Config:")
    print(cfg)

    # handle prompt
    cfg_prompt = cfg.get("prompt", None)
    prompt = prompt or cfg_prompt

    prompt_template = prompt_template or cfg.get("prompt_template", None)
    if prompt_template is not None:
        assert "{}" in prompt_template, '"{}" must be contained in "prompt_template".'
        prompt = prompt_template.format(prompt)

        print(f'Convert input prompt to "{prompt}".')

    # handle timesteps
    num_inference_steps = num_inference_steps or cfg.get("num_inference_steps", None)
    strength = strength or cfg.get("strength", None)
    t_index_list = t_index_list or cfg.get("t_index_list", None)

    stream = StreamAnimateDiffusionDepthWrapper(
        few_step_model_type=few_step_model_type,
        config_path=config_path,
        cfg_type="none",
        dreambooth_path=dreambooth_path,
        lora_dict=lora_dict,
        strength=strength,
        num_inference_steps=num_inference_steps,
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=width,
        height=height,
        acceleration=acceleration,
        do_add_noise=True,
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=True,
        use_tiny_vae=enable_tiny_vae,
        seed=seed,
    )
    warmup_frames = video[:8].permute(0, 3, 1, 2)
    warmup_results = stream.prepare(
        warmup_frames=warmup_frames,
        prompt=prompt,
        guidance_scale=1,
    )
    video_result = torch.zeros(video.shape[0], height, width, 3)
    warmup_results = warmup_results.cpu().float()
    video_result[:8] = warmup_results

    skip_frames = stream.batch_size - 1
    for i in tqdm(range(8, video.shape[0])):
        output_image = stream(video[i].permute(2, 0, 1))
        if i - 8 >= skip_frames:
            video_result[i - skip_frames] = output_image.permute(1, 2, 0)
    video_result = video_result[:-skip_frames]
    # video_result = video_result[:8]

    save_root = os.path.dirname(output)
    if save_root != "":
        os.makedirs(save_root, exist_ok=True)
    if output.endswith(".mp4"):
        video_result = video_result * 255
        write_video(output, video_result, fps=fps)
        if save_input:
            write_video(output.replace(".mp4", "_inp.mp4"), video * 255, fps=fps)
    elif output.endswith(".gif"):
        save_videos_grid(
            video_result.permute(3, 0, 1, 2)[None, ...],
            output,
            rescale=False,
            fps=fps,
        )
        if save_input:
            save_videos_grid(
                video.permute(3, 0, 1, 2)[None, ...],
                output.replace(".gif", "_inp.gif"),
                rescale=False,
                fps=fps,
            )
    else:
        raise TypeError(f"Unsupported output format: {output}")
    print("Inference time ema: ", stream.stream.inference_time_ema)
    inference_time_list = np.array(stream.stream.inference_time_list)
    print(f"Inference time mean & std: {inference_time_list.mean()} +/- {inference_time_list.std()}")
    if hasattr(stream.stream, "depth_time_ema"):
        print("Depth time ema: ", stream.stream.depth_time_ema)

    print(f'Video saved to "{output}".')


if __name__ == "__main__":
    fire.Fire(main)
