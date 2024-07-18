import torch
import torch.nn.functional as F
from einops import rearrange

from .attention import CrossAttention
from .positional_encoding import PositionalEncoding


class StreamTemporalAttention(CrossAttention):
    """

    * window_size: The max length of attention window.
    * sink_size: The number sink token.
    * positional_rule: absolute, relative

    Therefore, the seq length of temporal self-attention will be:
        sink_length + cache_size

    """

    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=32,
        window_size=8,
        sink_size=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.attention_mode = self._orig_attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0.0,
            max_len=temporal_position_encoding_max_len,
        )

        self.window_size = window_size
        self.sink_size = sink_size
        self.cache_size = self.window_size - self.sink_size
        assert self.cache_size >= 0, (
            "cache_size must be greater or equal to 0. Please check your configuration. "
            f"window_size: {window_size}, sink_size: {sink_size}, "
            f"cache_size: {self.cache_size}"
        )

        self.motion_module_idx = None

    def set_index(self, idx):
        self.motion_module_idx = idx

    @torch.no_grad()
    def set_cache(self, denoising_steps_num: int):
        """
        larger buffer index means cleaner latent
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # [t, 2, hw, L, c], 2 means k and v
        kv_cache = torch.zeros(
            denoising_steps_num,
            2,
            self.h * self.w,
            self.window_size,
            self.kv_channels,
            device=device,
            dtype=dtype,
        )
        self.denoising_steps_num = denoising_steps_num

        return kv_cache

    @torch.no_grad()
    def prepare_pe_buffer(self):
        """In AnimateDiff, Temporal Self-attention use absolute positional encoding:
        q = w_q * (x + pe) + bias
        k = w_k * (x + pe) + bias
        v = w_v * (x + pe) + bias

        If we want to conduct relative positional encoding with kv-cache, we should pre-calcute
        `w_q/k/v * pe` and then cache `w_q/k/v * x + bias`
        """

        pe_list = self.pos_encoder.pe[:, : self.window_size]  # [1, window_size, ch]
        q_pe = F.linear(pe_list, self.to_q.weight)
        k_pe = F.linear(pe_list, self.to_k.weight)
        v_pe = F.linear(pe_list, self.to_v.weight)

        self.register_buffer("q_pe", q_pe)
        self.register_buffer("k_pe", k_pe)
        self.register_buffer("v_pe", v_pe)

    def prepare_qkv_full_and_cache(self, hidden_states, kv_cache, pe_idx, update_idx):
        """
        hidden_states: [(N * bhw), F, c],
        kv_cache: [2, N, hw, L, c]

        * for warmup case: `N` should be 1 and `F` should be warmup_size (`sink_size`)
        * for streaming case: `N` should be `denoising_steps_num` and `F` should be `chunk_size`

        """
        q_layer = self.to_q(hidden_states)
        k_layer = self.to_k(hidden_states)
        v_layer = self.to_v(hidden_states)

        q_layer = rearrange(q_layer, "(n bhw) f c -> n bhw f c", n=self.denoising_steps_num)
        k_layer = rearrange(k_layer, "(n bhw) f c -> n bhw f c", n=self.denoising_steps_num)
        v_layer = rearrange(v_layer, "(n bhw) f c -> n bhw f c", n=self.denoising_steps_num)

        # onnx & trt friendly indexing
        for idx in range(self.denoising_steps_num):
            kv_cache[idx, 0, :, update_idx[idx]] = k_layer[idx, :, 0]
            kv_cache[idx, 1, :, update_idx[idx]] = v_layer[idx, :, 0]

        k_full = kv_cache[:, 0]
        v_full = kv_cache[:, 1]

        kv_idx = pe_idx
        q_idx = torch.stack([kv_idx[idx, update_idx[idx]] for idx in range(self.denoising_steps_num)]).unsqueeze_(
            1
        )  # [timesteps, 1]

        pe_k = torch.cat(
            [self.k_pe.index_select(1, kv_idx[idx]) for idx in range(self.denoising_steps_num)], dim=0
        )  # [n, window_size, c]
        pe_v = torch.cat(
            [self.v_pe.index_select(1, kv_idx[idx]) for idx in range(self.denoising_steps_num)], dim=0
        )  # [n, window_size, c]
        pe_q = torch.cat(
            [self.q_pe.index_select(1, q_idx[idx]) for idx in range(self.denoising_steps_num)], dim=0
        )  # [n, window_size, c]

        q_layer = q_layer + pe_q.unsqueeze(1)
        k_full = k_full + pe_k.unsqueeze(1)
        v_full = v_full + pe_v.unsqueeze(1)

        q_layer = rearrange(q_layer, "n bhw f c -> (n bhw) f c")
        k_full = rearrange(k_full, "n bhw f c -> (n bhw) f c")
        v_full = rearrange(v_full, "n bhw f c -> (n bhw) f c")

        return q_layer, k_full, v_full

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        temporal_attention_mask=None,
        kv_cache=None,
        pe_idx=None,
        update_idx=None,
        *args,
        **kwargs,
    ):
        """
        temporal_attention_mask: attention mask specific for the temporal self-attention.
        """

        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query_layer, key_full, value_full = self.prepare_qkv_full_and_cache(
            hidden_states, kv_cache, pe_idx, update_idx
        )

        # [(n * hw * b), f, c] -> [(n * hw * b * head), f, c // head]
        query_layer = self.reshape_heads_to_batch_dim(query_layer)
        key_full = self.reshape_heads_to_batch_dim(key_full)
        value_full = self.reshape_heads_to_batch_dim(value_full)

        if temporal_attention_mask is not None:
            q_size = query_layer.shape[1]
            # [n, self.window_size] -> [n, hw, q_size, window_size]
            temporal_attention_mask_ = temporal_attention_mask[:, None, None, :].repeat(1, self.h * self.w, q_size, 1)
            temporal_attention_mask_ = rearrange(temporal_attention_mask_, "n hw Q KV -> (n hw) Q KV")
            temporal_attention_mask_ = temporal_attention_mask_.repeat_interleave(self.heads, dim=0)
        else:
            temporal_attention_mask_ = None

        # attention, what we cannot get enough of
        if hasattr(F, "scaled_dot_product_attention"):
            hidden_states = self._memory_efficient_attention_pt20(
                query_layer, key_full, value_full, attention_mask=temporal_attention_mask_
            )

        elif self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(
                query_layer, key_full, value_full, attention_mask=temporal_attention_mask_
            )
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query_layer.dtype)
        else:
            hidden_states = self._attention(query_layer, key_full, value_full, temporal_attention_mask_)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
