import os
import os.path as osp

from omegaconf import OmegaConf


config_suffix = [".yaml"]


def load_config(config: str) -> OmegaConf:
    config = OmegaConf.load(config)
    base_config = config.pop("base", None)

    if base_config:
        config = OmegaConf.merge(OmegaConf.load(base_config), config)

    return config


def dump_config(config: OmegaConf, save_path: str = None):
    from omegaconf import Container

    if isinstance(config, Container):
        if not save_path.endswith(".yaml"):
            save_dir = save_path
            save_path = osp.join(save_dir, "config.yaml")
        else:
            save_dir = osp.basename(config)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config, save_path)

    else:
        raise TypeError("Only support saving `Config` from `OmegaConf`.")

    print(f"Dump Config to {save_path}.")
