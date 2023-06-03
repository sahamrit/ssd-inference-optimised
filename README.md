# SSD INFERENCE OPTIMIZED

This repository optimises inference of [NVIDIA's SSD300](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/). 
This is purely for learning purposes and reimplements most ideas from [Paul Bridger's](https://paulbridger.com/posts/about/) blog on optimising SSD inference from 9FPS to 2530 FPS.

## Overview

This repository contains implementation of **SSD300** inference with **Deepstream** library for video processing. It runs at **336FPS** on a **V100 machine**. 

**Changes introduced in V3** 
1. Post processing on GPU is done on entire batch at once. Earlier we had to do it per image on GPU. We encode the class labels such that classes across images can be distinguished. Refer to **postprocess** function.

2. Keep objects end-to-end on GPU. [nvds.py](utils\nvds.py) helps to parse serialised buffers we receive from **nvvideoconvert** element. Refer to [deepstream pipeline](logs\ssd_inference_pytorch_ds_pipeline.png). The implementation is taken from [Paul Bridger's repo](https://github.com/pbridger/pytorch-video-pipeline/blob/master/ghetto_nvds.py).

Refer to other branches to see gradual improvements, profiling analysis for better learning.
1. [Baseline](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-baseline-pipeline) runs at 6FPS.
2. [V1](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-inference-v1) runs at 80 FPS. Changes - batching and pre/post processing on GPU.
3. [V2](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-inference-v2) runs at 196 FPS. Changes - concurrency and fp16.
4. [V3](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-inference-v3) runs at 336 FPS. Changes - batch postprocessing and end-to-end GPU pipeline.

## Code Walk

[ssd_inference_pytorch](./ssd_inference_pytorch.py) - Contains SSD inference code. This runs at 336 FPS on V100 GPU.

[utils](./utils/) - Contains utilities related to SSD pre & post processing and Deepstream utilities.

[gst_frame_probe.py](./gst_frame_probe.py) - Contains basic Gstreamer pipeline with probe on sink pad. The probe is of type buffer and hence the callback is at frame level. The callback converts buffer to np array. **This is to test working of basic Gstreamer pipeline.**

[examples](./examples/) - contains a basic Gstreamer pipeline and SSD inference working and decoupled from each other.


[input](./media/) - Contains input MP4 video for inference.
## Analysis


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









