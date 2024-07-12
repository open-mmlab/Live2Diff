# Live2Diff: **Live** Stream Translation via Uni-directional Attention in Video **Diffusion** Models

<p align="center">
  <img src="./assets/attn-mask.png" width=100%>
</p>

**Authors:** [Zhening Xing](https://github.com/LeoXing1996), [Gereon Fox](https://people.mpi-inf.mpg.de/~gfox/), [Yanhong Zeng](https://zengyh1900.github.io/), [Xingang Pan](https://xingangpan.github.io/), [Mohamed Elgharib](https://people.mpi-inf.mpg.de/~elgharib/), [Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/), [Kai Chen †](https://chenkai.site/) (†: corresponding author)


[![arXiv](https://img.shields.io/badge/arXiv-2407.08701-b31b1b.svg)](https://arxiv.org/abs/2407.08701)
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
                <video controls autoplay src="https://github.com/user-attachments/assets/c39e4b1f-e336-479a-af72-d07b1e3c6e30" width="100%">
            </td>
            <td>
                <video controls autoplay src="https://github.com/user-attachments/assets/42727f46-b3cf-48ea-971c-9f653bf5a264" width="80%">
            </td>
        </tr>
        </tbody>
    </table>

</div>

## Acknowledgements

The video and image demos in this GitHub repository were generated using [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5). Stream batch in [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) is used for model acceleration. The design of Video Diffusion Model is adopted from [AnimateDiff](https://github.com/guoyww/AnimateDiff). We use a third-party implementation of [MiDaS](https://github.com/lewiji/MiDaS) implementation which support onnx export. Our online demo is modified from [Real-Time-Latent-Consistency-Model](https://github.com/radames/Real-Time-Latent-Consistency-Model/).

## BibTex

If you find it helpful, please consider citing our work:

```bibtex
@article{xing2024live2diff,
  title={Live2Diff: Live Stream Translation via Uni-directional Attention in Video Diffusion Models},
  author={Zhening Xing and Gereon Fox and Yanhong Zeng and Xingang Pan and Mohamed Elgharib and Christian Theobalt and Kai Chen},
  booktitle={arXiv preprint arxiv:2407.08701},
  year={2024}
}
```
