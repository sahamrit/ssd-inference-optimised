# SSD INFERENCE OPTIMIZED

This repository optimises inference of [NVIDIA's SSD300](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/). 
This is purely for learning purposes and reimplements most ideas from [Paul Bridger's](https://paulbridger.com/posts/about/) blog on optimising SSD inference from 9FPS to 2530 FPS.

## Overview

This [branch](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-inference-v2) contains  implementation of **SSD300** inference with **Deepstream** library for video processing. This is improvement over [version1](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-inference-v1) (**80FPS**) and runs at **196FPS** on a **V100 machine**.

**Changes introduced** 
1. Concurrency with two python threads. Each python thread submits kernels to its own CUDA stream.
2. Half precision inference.

## Code Walk

[gst_frame_probe.py](./gst_frame_probe.py) - Contains basic Gstreamer pipeline with probe on sink pad. The probe is of type buffer and hence the callback is at frame level. The callback converts buffer to np array. **This is to test working on basic Gstreamer pipeline.**

[ssd_inference_pytorch](./ssd_inference_pytorch.py) - Contains SSD inference code. This version introduces concurrency and half precision inference.

[utils](./utils/) - Contains utilities related to SSD pre & post processing and some Gst utilities.

[examples](./examples/) - Contains a basic Gstreamer pipeline and SSD inference working and decoupled from each other.

[input](./media/) - Contains input MP4 video for inference.
## Analysis

[profiling_analysis](./profiling_analysis/) - Contains profiling analysis of current code version.

[logs](./logs/) - contains logs related to object detection sample, gst pipeline diagrams etc.

## Environment Setup


Build image corresponding to [Dockerfile](./Dockerfile). 
```
docker build -t <image_name>:<tag> <path_to_Dockerfile>
``` 
Run container from the image built.
```
sudo nvidia-docker run -it --privileged --network=host --pid=host --gpus all -v /home/azureuser/cloudfiles:/mnt <image_id> /bin/bash
```
Add following commands to **.bashrc**
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-6.1/lib/:$LD_LIBRARY_PATH
```
Commands
```
python ssd_inference_pytorch.py 

## Run it from within docker
## Try changing input media/in.mp4
```
**Additional Tip** - You can access Vscode features like Intellisense, if you open the project from within container. Checkout [Developing within Vscode tutorial](https://code.visualstudio.com/docs/devcontainers/containers) from their official documentation. 





