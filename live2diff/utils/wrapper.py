import gc
import os
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderTiny
from PIL import Image

from live2diff import StreamAnimateDiffusionDepth
from live2diff.image_utils import postprocess_image
from live2diff.pipeline_stream_animation_depth import WARMUP_FRAMES


class StreamAnimateDiffusionDepthWrapper:
    def __init__(
        self,
        config_path: str,
        few_step_model_type: str,
        num_inference_steps: int,
        t_index_list: Optional[List[int]] = None,
        strength: Optional[float] = None,
        dreambooth_path: Optional[str] = None,
        lora_dict: Optional[Dict[str, float]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 42,
        engine_dir: Optional[Union[str, Path]] = "engines",
        opt_unet: bool = False,
    ):
        """
        Initializes the StreamAnimateDiffusionWrapper.

        Parameters
        ----------
        config_path : str
            The model id or path to load.
        few_step_model_type : str
            The few step model type to use.
        num_inference_steps : int
            The number of inference steps to perform. If `t_index_list`
            is passed, `num_infernce_steps` will parsed as the number
            of denoising steps before apply few-step lora. Otherwise,
            `num_inference_steps` will be parsed as the number of
            steps after applying few-step lora.
        t_index_list : List[int]
            The t_index_list to use for inference.
        strength : Optional[float]
            The strength to use for inference.
        dreambooth_path : Optional[str]
            The dreambooth path to use for inference. If not passed,
            will use dreambooth from config.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 42.
        engine_dir : Optional[Union[str, Path]], optional
            The directory to save TensorRT engines, by default "engines".
        opt_unet : bool, optional
            Whether to optimize UNet or not, by default False.
        """
        self.sd_turbo = False

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size

        self.use_denoising_batch = use_denoising_batch

        self.stream: StreamAnimateDiffusionDepth = self._load_model(
            config_path=config_path,
            lora_dict=lora_dict,
            dreambooth_path=dreambooth_path,
            few_step_model_type=few_step_model_type,
            vae_id=vae_id,
            num_inference_steps=num_inference_steps,
            t_index_list=t_index_list,
            strength=strength,
            height=height,
            width=width,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            opt_unet=opt_unet,
        )
        self.batch_size = len(self.stream.t_list) * frame_buffer_size if use_denoising_batch else frame_buffer_size

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        warmup_frames: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> torch.Tensor:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.

        Returns
        ----------
        warmup_frames : torch.Tensor
            generated warmup-frames.

        """
        warmup_frames = self.stream.prepare(
            warmup_frames=warmup_frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            delta=delta,
        )

        warmup_frames = warmup_frames.permute(0, 2, 3, 1)
        warmup_frames = (warmup_frames.clip(-1, 1) + 1) / 2
        return warmup_frames

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        return self.img2img(image, prompt)

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(image, self.height, self.width).to(
            device=self.device, dtype=self.dtype
        )

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            output = postprocess_image(image_tensor, output_type=output_type)
        else:
            output = postprocess_image(image_tensor, output_type=output_type)[0]

        if output_type not in ["pil", "np"]:
            return output.cpu()
        else:
            return output

    @staticmethod
    def get_model_prefix(
        config_path: str,
        few_step_model_type: str,
        use_tiny_vae: bool,
        num_denoising_steps: int,
        height: int,
        width: int,
        dreambooth: Optional[str] = None,
        lora_dict: Optional[dict] = None,
    ) -> str:
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        third_party = config.third_party_dict
        dreambooth_path = dreambooth or third_party.dreambooth
        if dreambooth_path is None:
            dreambooth_name = "sd15"
        else:
            dreambooth_name = Path(dreambooth_path).stem

        base_lora_list = third_party.get("lora_list", [])
        lora_dict = lora_dict or {}
        for lora_alpha in base_lora_list:
            lora_name = lora_alpha["lora"]
            alpha = lora_alpha["lora_alpha"]
            if lora_name not in lora_dict:
                lora_dict[lora_name] = alpha

        prefix = f"{dreambooth_name}--{few_step_model_type}--step{num_denoising_steps}--"
        for k, v in lora_dict.items():
            prefix += f"{Path(k).stem}-{v}--"
        prefix += f"tiny_vae-{use_tiny_vae}--h-{height}--w-{width}"
        return prefix

    def _load_model(
        self,
        config_path: str,
        num_inference_steps: int,
        height: int,
        width: int,
        t_index_list: Optional[List[int]] = None,
        strength: Optional[float] = None,
        dreambooth_path: Optional[str] = None,
        lora_dict: Optional[Dict[str, float]] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        few_step_model_type: Optional[str] = None,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        opt_unet: bool = False,
    ) -> StreamAnimateDiffusionDepth:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        6. Load the safety checker if needed.

        Parameters
        ----------
        config_path : str
            The path to config, all needed checkpoints are list in config file.
        t_index_list : List[int]
            The t_index_list to use for inference.
        dreambooth_path : Optional[str]
            The dreambooth path to use for inference. If not passed,
            will use dreambooth from config.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        opt_unet : bool, optional
            Whether to optimize UNet or not, by default False.

        Returns
        -------
        AnimatePipeline
            The loaded pipeline.
        """
        supported_few_step_model = ["LCM"]
        assert (
            few_step_model_type.upper() in supported_few_step_model
        ), f"Only support few_step_model: {supported_few_step_model}, but receive {few_step_model_type}."

        # NOTE: build animatediff pipeline
        from live2diff.animatediff.pipeline import AnimationDepthPipeline

        try:
            pipe = AnimationDepthPipeline.build_pipeline(
                config_path,
            ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        if few_step_model_type.upper() == "LCM":
            few_step_lora = "latent-consistency/lcm-lora-sdv1-5"
            stream_pipeline_cls = StreamAnimateDiffusionDepth

        print(f"Pipeline class: {stream_pipeline_cls}")
        print(f"Few-step LoRA: {few_step_lora}")

        # parse clip skip from config
        from .config import load_config

        cfg = load_config(config_path)
        third_party_dict = cfg.third_party_dict
        clip_skip = third_party_dict.get("clip_skip", 1)

        stream = stream_pipeline_cls(
            pipe=pipe,
            num_inference_steps=num_inference_steps,
            t_index_list=t_index_list,
            strength=strength,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
            clip_skip=clip_skip,
        )

        stream.load_warmup_unet(config_path)
        stream.load_lora(few_step_lora)
        stream.fuse_lora()

        denoising_steps_num = len(stream.t_list)
        stream.prepare_cache(
            height=height,
            width=width,
            denoising_steps_num=denoising_steps_num,
        )
        kv_cache_list = stream.kv_cache_list

        if lora_dict is not None:
            for lora_name, lora_scale in lora_dict.items():
                stream.load_lora(lora_name)
                stream.fuse_lora(lora_scale=lora_scale)
                print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            vae_id = "madebyollin/taesd" if vae_id is None else vae_id
            stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=pipe.device, dtype=pipe.dtype)

        try:
            if acceleration == "none":
                stream.pipe.unet = torch.compile(stream.pipe.unet, options={"triton.cudagraphs": True}, fullgraph=True)
                stream.vae = torch.compile(stream.vae, options={"triton.cudagraphs": True}, fullgraph=True)
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda

                from live2diff.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_engine,
                )
                from live2diff.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    MidasEngine,
                    UNet2DConditionModelDepthEngine,
                )
                from live2diff.acceleration.tensorrt.models import (
                    VAE,
                    InflatedUNetDepth,
                    Midas,
                    VAEEncoder,
                )

                prefix = self.get_model_prefix(
                    config_path=config_path,
                    few_step_model_type=few_step_model_type,
                    use_tiny_vae=use_tiny_vae,
                    num_denoising_steps=denoising_steps_num,
                    height=height,
                    width=width,
                    dreambooth=dreambooth_path,
                    lora_dict=lora_dict,
                )

                engine_dir = os.path.join(Path(engine_dir), prefix)
                unet_path = os.path.join(engine_dir, "unet", "unet.engine")
                unet_opt_path = os.path.join(engine_dir, "unet-opt", "unet.engine.opt")
                midas_path = os.path.join(engine_dir, "depth", "midas.engine")
                vae_encoder_path = os.path.join(engine_dir, "vae", "vae_encoder.engine")
                vae_decoder_path = os.path.join(engine_dir, "vae", "vae_decoder.engine")

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    os.makedirs(os.path.dirname(unet_opt_path), exist_ok=True)
                    unet_model = InflatedUNetDepth(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                        kv_cache_list=kv_cache_list,
                    )
                    compile_engine(
                        torch_model=stream.unet,
                        model_data=unet_model,
                        onnx_path=unet_path + ".onnx",
                        onnx_opt_path=unet_opt_path,  # use specific folder for external data
                        engine_path=unet_path,
                        opt_image_height=height,
                        opt_image_width=width,
                        opt_batch_size=stream.trt_unet_batch_size,
                        engine_build_options={"ignore_onnx_optimize": not opt_unet},
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    max_bz = WARMUP_FRAMES
                    opt_bz = min_bz = 1
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=max_bz,
                        min_batch_size=min_bz,
                    )
                    compile_engine(
                        torch_model=stream.vae,
                        model_data=vae_decoder_model,
                        onnx_path=vae_decoder_path + ".onnx",
                        onnx_opt_path=vae_decoder_path + ".opt.onnx",
                        engine_path=vae_decoder_path,
                        opt_image_height=height,
                        opt_image_width=width,
                        opt_batch_size=opt_bz,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(midas_path):
                    os.makedirs(os.path.dirname(midas_path), exist_ok=True)
                    max_bz = WARMUP_FRAMES
                    opt_bz = min_bz = 1
                    midas = Midas(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=max_bz,
                        min_batch_size=min_bz,
                    )
                    compile_engine(
                        torch_model=stream.depth_detector.half(),
                        model_data=midas,
                        onnx_path=midas_path + ".onnx",
                        onnx_opt_path=midas_path + ".opt.onnx",
                        engine_path=midas_path,
                        opt_batch_size=opt_bz,
                        opt_image_height=384,
                        opt_image_width=384,
                        engine_build_options={
                            "auto_cast": False,
                            "handle_batch_norm": True,
                            "ignore_onnx_optimize": True,
                        },
                    )

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    max_bz = WARMUP_FRAMES
                    opt_bz = min_bz = 1
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=max_bz,
                        min_batch_size=min_bz,
                    )
                    compile_engine(
                        torch_model=vae_encoder,
                        model_data=vae_encoder_model,
                        onnx_path=vae_encoder_path + ".onnx",
                        onnx_opt_path=vae_encoder_path + ".opt.onnx",
                        engine_path=vae_encoder_path,
                        opt_batch_size=opt_bz,
                        opt_image_height=height,
                        opt_image_width=width,
                    )
                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype
                midas_dtype = stream.depth_detector.dtype

                stream.unet = UNet2DConditionModelDepthEngine(unet_path, cuda_stream, use_cuda_graph=False)
                stream.depth_detector = MidasEngine(midas_path, cuda_stream, use_cuda_graph=False)
                setattr(stream.depth_detector, "dtype", midas_dtype)
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                stream.is_tensorrt = True

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")

        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        return stream
