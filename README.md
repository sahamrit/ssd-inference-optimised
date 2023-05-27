# SSD INFERENCE OPTIMIZED

This repository optimises inference of [NVIDIA's SSD300](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/). 
This is purely for learning purposes and reimplements most ideas from [Paul Bridger's](https://paulbridger.com/posts/about/) blog on optimising SSD inference from 9FPS to 2530 FPS.

## Overview

This [branch](https://github.com/sahamrit/ssd-inference-optimised/tree/pytorch-baseline-pipeline) contains basic implementation of **SSD300** inference with **Gstreamer** library for video processing. Baseline speed is **4.17 FPS (K80 GPU)** without profiling overhead.

## Code Walk

[gst_frame_probe.py](./gst_frame_probe.py) - Contains basic Gstreamer pipeline with probe on sink pad. The probe is of type buffer and hence the callback is at frame level. The callback converts buffer to np array. This is to test working on basic Gstreamer pipeline.

[ssd_inference_pytorch](./ssd_inference_pytorch.py) - Contains SSD inference code in the frame callback and hence does SSD inference per frame.

[examples](./examples/) - contains a basic Gstreamer pipeline and SSD inference working and decoupled from each other.

[input](./media/) - Contains input MP4 video for inference.
## Analysis

[profiling_analysis](./profiling_analysis/) - Contains Nsys report of the current code version. [report1](./report1.nsys-rep) contains the complete report which can be viewed from NVIDIA Nsight.

[logs](./logs/) - contains logs related to object detection sample, gst pipeline diagrams etc.

## Environment Setup

TODO



