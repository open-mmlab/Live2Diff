# Adapted from https://github.com/guoyww/AnimateDiff
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torch import nn

from .attention import CrossAttention
from .positional_encoding import PositionalEncoding
from .resnet import zero_module
from .stream_motion_module import StreamTemporalAttention


def attn_mask_to_bias(attn_mask: torch.Tensor):
    """
    Convert bool attention mask to float attention bias tensor.
    """
    if attn_mask.dtype in [torch.float, torch.half]:
        return attn_mask
    elif attn_mask.dtype == torch.bool:
        attn_bias = torch.zeros_like(attn_mask).float().masked_fill(attn_mask.logical_not(), float("-inf"))
        return attn_bias
    else:
        raise TypeError("Only support float or bool tensor for attn_mask input. " f"But receive {type(attn_mask)}.")


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str,
    motion_module_kwargs: dict,
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    elif motion_module_type == "Streaming":
        return VanillaTemporalModule(
            in_channels=in_channels,
            enable_streaming=True,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=32,
        temporal_attention_dim_div=1,
        # parameters for 3d conv
        num_3d_conv_layers=0,
        kernel_size=3,
        down_up_sample=False,
        zero_initialize=True,
        attention_class_name="versatile",
        attention_kwargs={},
        enable_streaming=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            attention_class_name=attention_class_name,
            attention_kwargs=attention_kwargs,
            enable_streaming=enable_streaming,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

        self.enable_streaming = enable_streaming

    def forward(self, *args, **kwargs):
        fwd_fn = self.forward_streaming if self.enable_streaming else self.forward_orig
        return fwd_fn(*args, **kwargs)

    def forward_orig(
        self,
        input_tensor,
        temb,
        encoder_hidden_states,
        attention_mask=None,
        temporal_attention_mask=None,
        kv_cache=None,
    ):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, temporal_attention_mask, kv_cache=kv_cache
        )

        output = hidden_states
        return output

    def forward_streaming(
        self,
        input_tensor,
        temb,
        encoder_hidden_states,
        attention_mask=None,
        temporal_attention_mask=None,
        kv_cache=None,
        pe_idx=None,
        update_idx=None,
    ):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temporal_attention_mask,
            kv_cache=kv_cache,
            pe_idx=pe_idx,
            update_idx=update_idx,
        )

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=1280,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=32,
        attention_class_name="versatile",
        attention_kwargs={},
        enable_streaming=False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    attention_class_name=attention_class_name,
                    attention_extra_args=attention_kwargs,
                    enable_streaming=enable_streaming,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.enable_streaming = enable_streaming

    def forward(self, *args, **kwargs):
        fwd_fn = self.forward_streaming if self.enable_streaming else self.forward_orig
        return fwd_fn(*args, **kwargs)

    def forward_orig(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temporal_attention_mask=None,
        kv_cache=None,
    ):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                height=height,
                width=width,
                temporal_attention_mask=temporal_attention_mask,
                kv_cache=kv_cache,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output

    def forward_streaming(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temporal_attention_mask=None,
        kv_cache=None,
        pe_idx=None,
        update_idx=None,
    ):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                height=height,
                width=width,
                temporal_attention_mask=temporal_attention_mask,
                kv_cache=kv_cache,
                pe_idx=pe_idx,
                update_idx=update_idx,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=32,
        attention_class_name: str = "versatile",
        attention_extra_args={},
        enable_streaming=False,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        if attention_class_name == "versatile":
            attention_cls = VersatileAttention
        elif attention_class_name == "stream":
            attention_cls = StreamTemporalAttention
            assert enable_streaming, "StreamTemporalAttention can only used under streaming mode"
        else:
            raise ValueError(f"Do not support attention_cls: {attention_class_name}.")

        for block_name in attention_block_types:
            attention_blocks.append(
                attention_cls(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    **attention_extra_args,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

        self.enable_streaming = enable_streaming

    def forward(self, *args, **kwargs):
        fwd_func = self.forward_streaming if self.enable_streaming else self.forward_orig
        return fwd_func(*args, **kwargs)

    def forward_orig(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        height=None,
        width=None,
        temporal_attention_mask=None,
        kv_cache=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            kv_cache_ = kv_cache[attention_block.motion_module_idx]
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                    height=height,
                    width=width,
                    temporal_attention_mask=temporal_attention_mask,
                    kv_cache=kv_cache_,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output

    def forward_streaming(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        height=None,
        width=None,
        temporal_attention_mask=None,
        kv_cache=None,
        pe_idx=None,
        update_idx=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            kv_cache_ = kv_cache[attention_block.motion_module_idx]
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                    height=height,
                    width=width,
                    temporal_attention_mask=temporal_attention_mask,
                    kv_cache=kv_cache_,
                    pe_idx=pe_idx,
                    update_idx=update_idx,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class VersatileAttention(CrossAttention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=32,
        stream_cache_mode=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.stream_cache_mode = stream_cache_mode
        self.timestep = None

        assert attention_mode in ["Temporal"]

        self.attention_mode = self._orig_attention_mode = attention_mode
        self.is_cross_attention = kwargs.get("cross_attention_dim", None) is not None

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"], dropout=0.0, max_len=temporal_position_encoding_max_len
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_index(self, idx):
        self.motion_module_idx = idx

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        kv_cache=None,
        *args,
        **kwargs,
    ):
        batch_size_frame, sequence_length, _ = hidden_states.shape

        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        kv_cache[0, :, :video_length, :] = key.clone()
        kv_cache[1, :, :video_length, :] = value.clone()

        pe = self.pos_encoder.pe[:, :video_length]

        pe_q = self.to_q(pe)
        pe_k = self.to_k(pe)
        pe_v = self.to_v(pe)

        query = query + pe_q
        key = key + pe_k
        value = value + pe_v

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            attention_bias = attn_mask_to_bias(attention_mask)
            if attention_bias.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_bias = F.pad(attention_mask, (0, target_length), value=float("-inf"))
                attention_bias = attention_bias.repeat_interleave(self.heads, dim=0)
            attention_bias = attention_bias.to(query)
        else:
            attention_bias = None

        hidden_states = self._memory_efficient_attention_pt20(query, key, value, attention_bias)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
