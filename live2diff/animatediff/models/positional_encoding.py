import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, roll: Optional[int] = None, full_video_length: Optional[int] = None):
        """
        Support roll for positional encoding.
        We select the first `full_video_length` elements and roll it by `roll`.
        And then select the first `x.size(1)` elements and add them to `x`.

        Take full_video_length = 4, roll = 2, and x.size(1) = 1 as example.

        If the original positional encoding is:
            [1, 2, 3, 4, 5, 6, 7, 8]
        The rolled encoding is:
            [3, 4, 1, 2]
        And the selected encoding added to input is:
            [3, 4]

        """
        if roll is None:
            pe = self.pe[:, : x.size(1)]
        else:
            assert full_video_length is not None, "full_video_length must be passed when roll is not None."
            pe = self.pe[:, :full_video_length].roll(shifts=roll, dims=1)[:, : x.size(1)]
        x = x + pe
        return self.dropout(x)
