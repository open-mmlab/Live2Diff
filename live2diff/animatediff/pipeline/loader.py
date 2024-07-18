from typing import Dict, List, Optional, Union

import torch
from diffusers.loaders.lora import LoraLoaderMixin
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.utils import USE_PEFT_BACKEND


class LoraLoaderWithWarmup(LoraLoaderMixin):
    unet_warmup_name = "unet_warmup"

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name=None,
        **kwargs,
    ):
        # load lora for text encoder and unet-streaming
        super().load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name=adapter_name, **kwargs)

        # load lora for unet-warmup
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=getattr(self, self.unet_warmup_name) if not hasattr(self, "unet_warmup") else self.unet_warmup,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=adapter_name,
            _pipeline=self,
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):
        # fuse lora for text encoder and unet-streaming
        super().fuse_lora(fuse_unet, fuse_text_encoder, lora_scale, safe_fusing, adapter_names)

        # fuse lora for unet-warmup
        if fuse_unet:
            unet_warmup = (
                getattr(self, self.unet_warmup_name) if not hasattr(self, "unet_warmup") else self.unet_warmup
            )
            unet_warmup.fuse_lora(lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names)

    def unfuse_lora(self, unfuse_unet: bool = True, unfuse_text_encoder: bool = True):
        # unfuse lora for text encoder and unet-streaming
        super().unfuse_lora(unfuse_unet, unfuse_text_encoder)

        # unfuse lora for unet-warmup
        if unfuse_unet:
            unet_warmup = (
                getattr(self, self.unet_warmup_name) if not hasattr(self, "unet_warmup") else self.unet_warmup
            )
            if not USE_PEFT_BACKEND:
                unet_warmup.unfuse_lora()
            else:
                from peft.tuners.tuners_utils import BaseTunerLayer

                for module in unet_warmup.modules():
                    if isinstance(module, BaseTunerLayer):
                        module.unmerge()
