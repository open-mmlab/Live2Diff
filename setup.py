from setuptools import find_packages, setup


deps = [
    "diffusers==0.25.0",
    "transformers",
    "accelerate",
    "fire",
    "einops",
    "omegaconf",
    "imageio",
    "timm==0.6.7",
    "lightning",
    "peft",
    "av",
    "decord",
    "pillow",
    "pywin32;sys_platform == 'win32'",
]

deps_tensorrt = [
    "onnx==1.16.0",
    "onnxruntime==1.16.3",
    "protobuf==5.27.0",
    "polygraphy",
    "onnx-graphsurgeon",
    "cuda-python",
    "tensorrt==10.0.1",
    "colored",
]
deps_tensorrt_cu11 = [
    "tensorrt_cu11_libs==10.0.1",
    "tensorrt_cu11_bindings==10.0.1",
]
deps_tensorrt_cu12 = [
    "tensorrt_cu12_libs==10.0.1",
    "tensorrt_cu12_bindings==10.0.1",
]
extras = {
    "tensorrt_cu11": deps_tensorrt + deps_tensorrt_cu11,
    "tensorrt_cu12": deps_tensorrt + deps_tensorrt_cu12,
}


if __name__ == "__main__":
    setup(
        name="Live2Diff",
        version="0.1",
        description="real-time interactive video translation pipeline",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="deep learning diffusion pytorch stable diffusion streamdiffusion real-time next-frame prediction",
        license="Apache 2.0 License",
        author="leo",
        author_email="xingzhening@pjlab.org.cn",
        url="https://github.com/open-mmlab/Live2Diff",
        package_dir={"": "live2diff"},
        packages=find_packages("live2diff"),
        python_requires=">=3.10.0",
        install_requires=deps,
        extras_require=extras,
    )
