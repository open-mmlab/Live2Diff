# Modified from https://github.com/open-mmlab/PIA/blob/main/animatediff/utils/convert_lora_safetensor_to_diffusers.py and
# https://github.com/guoyww/AnimateDiff/blob/main/animatediff/utils/convert_lora_safetensor_to_diffusers.py
# Copyright 2023, Haofan Wang, Qixun Wang, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Conversion script for the LoRA's safetensors checkpoints."""

import torch


def convert_lora_model_level(
    state_dict, unet, text_encoder=None, LORA_PREFIX_UNET="lora_unet", LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=0.6
):
    """convert lora in model level instead of pipeline leval"""

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            assert text_encoder is not None, "text_encoder must be passed since lora contains text encoder layers"
            curr_layer = text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        # NOTE: load lycon, maybe have bugs :(
        if "conv_in" in pair_keys[0]:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            weight_up = weight_up.view(weight_up.size(0), -1)
            weight_down = weight_down.view(weight_down.size(0), -1)
            shape = list(curr_layer.weight.data.shape)
            shape[1] = 4
            curr_layer.weight.data[:, :4, ...] += alpha * (weight_up @ weight_down).view(*shape)
        elif "conv" in pair_keys[0]:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            weight_up = weight_up.view(weight_up.size(0), -1)
            weight_down = weight_down.view(weight_down.size(0), -1)
            shape = list(curr_layer.weight.data.shape)
            curr_layer.weight.data += alpha * (weight_up @ weight_down).view(*shape)
        elif len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3).to(
                curr_layer.weight.data.device
            )
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return unet, text_encoder
