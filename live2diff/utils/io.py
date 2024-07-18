import os
import os.path as osp

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def read_video_frames(folder: str, height=None, width=None):
    """
    Read video frames from the given folder.

    Output:
        frames, in [0, 255], uint8, THWC
    """
    _SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

    frames = [f for f in os.listdir(folder) if osp.splitext(f)[1] in _SUPPORTED_EXTENSIONS]
    # sort frames
    sorted_frames = sorted(frames, key=lambda x: int(osp.splitext(x)[0]))
    sorted_frames = [osp.join(folder, f) for f in sorted_frames]

    if height is not None and width is not None:
        sorted_frames = [np.array(Image.open(f).resize((width, height))) for f in sorted_frames]
    else:
        sorted_frames = [np.array(Image.open(f)) for f in sorted_frames]
    sorted_frames = torch.stack([torch.from_numpy(f) for f in sorted_frames], dim=0)
    return sorted_frames


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    parent_dir = os.path.dirname(path)
    if parent_dir != "":
        os.makedirs(parent_dir, exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, loop=0)
