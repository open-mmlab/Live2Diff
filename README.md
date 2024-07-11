# Live2Diff: **Live** Stream Translation via Uni-directional Attention in Video **Diffusion** Models

<p align="center">
  <img src="./assets/attn-mask.png" width=100%>
</p>

**Authors:** [Zhening Xing](https://github.com/LeoXing1996), [Gereon Fox](https://people.mpi-inf.mpg.de/~gfox/), [Yanhong Zeng](https://zengyh1900.github.io/), [Xingang Pan](https://xingangpan.github.io/), [Mohamed Elgharib](https://people.mpi-inf.mpg.de/~elgharib/), [Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/), [Kai Chen †](https://chenkai.site/) (†: corresponding author)


[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](TODO)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://live2diff.github.io/)

## Code will be release in **one week**, stay tuned!

## Key Features

<p align="center">
  <img src="./assets/framework.jpg" width=100%>
</p>

* **Uni-directional** Temporal Attention with **Warmup** Mechanism
* **Multitimestep KV-Cache** for Temporal Attention during Inference
* **Depth Prior** for Better Structure Consistency
* Compatible with **DreamBooth and LoRA** for Various Styles
* **TensorRT** Supported

The speed evaluation is conducted on **Ubuntu 20.04.6 LTS** and **Pytorch 2.2.2** with **RTX 4090 GPU** and **Intel(R) Xeon(R) Platinum 8352V CPU**. Denoising steps are set as 2.

| Resolution | TensorRT |    FPS    |
| :--------: | :------: | :-------: |
| 512 x 512  |  **On**  | **16.43** |
| 512 x 512  |   Off    |   6.91    |
| 768 x 512  |  **On**  | **12.15** |
| 768 x 512  |   Off    |   6.29    |

## Real-Time Video2Video Demo

<div align="center">
    <table align="center">
        <tbody>
        <tr align="center">
            <td>
                <p> Human Face (Web Camera Input) </p>
            </td>
            <td>
                <p> Anime Character (Screen Video Input) </p>
            </td>
        </tr>
        <tr align="center">
            <td>
                <video controls autoplay src="https://github-production-user-asset-6210df.s3.amazonaws.com/28132635/346031564-a56bea4f-9aed-497c-9cb9-868db72599fb.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240705%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240705T064755Z&X-Amz-Expires=300&X-Amz-Signature=3d0890fa6af8b86d10c69de9c31b8f97ff1d35cdbcdd4b1a9e465e49dd8d8194&X-Amz-SignedHeaders=host&actor_id=28132635&key_id=0&repo_id=819189683" width="100%">
            </td>
            <td>
                <video controls autoplay src="https://github-production-user-asset-6210df.s3.amazonaws.com/28132635/346055688-00f873dc-7348-458d-a814-7a7e8d158db5.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240705%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240705T081427Z&X-Amz-Expires=300&X-Amz-Signature=609199998d9f10a814d7a65e6dbff980a84bba77f25f43f99a1607bd0cee2f01&X-Amz-SignedHeaders=host&actor_id=28132635&key_id=0&repo_id=819189683" width="80%">
            </td>
        </tr>
        </tbody>
    </table>

</div>

## Acknowledgements

The video and image demos in this GitHub repository were generated using [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5). Stream batch in [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) is used for model acceleration. The design of Video Diffusion Model is adopted from [AnimateDiff](https://github.com/guoyww/AnimateDiff). We use a third-party implementation of [MiDaS](https://github.com/lewiji/MiDaS) implementation which support onnx export. Our online demo is modified from [Real-Time-Latent-Consistency-Model](https://github.com/radames/Real-Time-Latent-Consistency-Model/).
