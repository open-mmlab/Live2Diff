# Video2Video Example

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

This example, based on this [MJPEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Usage

### 1. Prepare Dependencies

You need Node.js 18+ and Python 3.10 to run this example. Please make sure you've installed all dependencies according to the [installation instructions](../README.md#installation).

```bash
cd frontend
npm i
npm run build
cd ..
pip install -r requirements.txt
```

If you face some difficulties in install `npm`, you can try to install it via `conda`:

```bash
conda install -c conda-forge nodejs
```

### 2. Run Demo

If you run the demo with default [setting](./demo_cfg.yaml), you should download the model for style `felted`.

```bash
bash ../scripts/download_model.sh felted
```

Then, you can run the demo with the following command, and open `http://127.0.0.1:7860` in your browser:

```bash
# with TensorRT acceleration, please pay patience for the first time, may take more than 20 minutes
python main.py --port 7860 --host 127.0.0.1 --acceleration tensorrt
# if you don't have TensorRT, you can run it with `none` acceleration
python main.py --port 7860 --host 127.0.0.1 --acceleration none
```

If you want to run this demo on a remote server, you can set host to `0.0.0.0`, e.g.

```bash
python main.py --port 7860 --host 0.0.0.0 --acceleration tensorrt
```
