import torch
import torch.nn as nn


try:
    from ...MiDaS.midas.dpt_depth import DPTDepthModel
except ImportError:
    print('Please pull the MiDaS submodule via "git submodule update --init --recursive"!')


class MidasDetector(nn.Module):
    def __init__(self, model_path="./models/dpt_hybrid_384"):
        super().__init__()

        self.model = DPTDepthModel(path=model_path, backbone="vitb_rn50_384", non_negative=True)
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        Input: [b, c, h, w]
        """
        return self.model(images)
