import sys
import os
import logging
import gi
import numpy as np
import torch
import time

torch.backends.cudnn.enabled = False
gi.require_version("Gst", "1.0")

from gi.repository import Gst

from utils.gst_utils import buffer_to_numpy
from utils.util import plt_results

logging.basicConfig(
    level=logging.INFO, format="[%(name)s] [%(levelname)8s] - %(message)s"
)
logger = logging.getLogger(__name__)

# global setting
videoformat = "RGBA"
log_dir = "/home/azureuser/localfiles/Repo/ssd-inference-optimised/logs"
ssd_threshold = 0.4
frames_processed = 0

# setup ssd eval
device = "cuda" if torch.cuda.is_available() else "cpu"
ssd_model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd")
ssd_utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)

ssd_model.to(device)
ssd_model.eval()


def preprocess(img):
    img = ssd_utils.rescale(img, 300, 300)
    img = ssd_utils.crop_center(img, 300, 300)
    img = ssd_utils.normalize(img)

    return img


def probe_callback_per_frame(pad, info):
    global frames_processed

    img = buffer_to_numpy(pad, info)
    img = preprocess(img)
    input_tensor = torch.tensor(img, device=device, dtype=torch.float32)
    logger.debug(
        f"Input tensor max : {input_tensor.max()}, min : {input_tensor.min()} and shape : {input_tensor.shape}"
    )
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        detections = ssd_model(input_tensor)

    logger.debug(
        f"Detections bbox : {detections[0].shape}, class : {detections[1].shape}"
    )
    result = ssd_utils.decode_results(detections)
    best_result = ssd_utils.pick_best(result[0], 0.40)

    if (frames_processed + 1) % 50 == 0:
        plt_results(
            [best_result],
            [img],
            os.path.join(log_dir, f"ssd_infer_pytorch_baseline_{frames_processed}.png"),
            ssd_utils,
        )
    frames_processed += 1
    return Gst.PadProbeReturn.OK


# initialize GStreamer
Gst.init(sys.argv[1:])

# build the pipeline
pipeline = Gst.parse_launch(
    f"filesrc location=media/in.mp4 num-buffers=256 ! \
     decodebin ! \
     nvvideoconvert ! \
     video/x-raw, format = {videoformat} ! \
     fakesink name=fs"
)

# start playing
pipeline.set_state(Gst.State.PLAYING)

# add probe to sink pad
fs = pipeline.get_by_name("fs")
sink_pad = fs.get_static_pad("sink")
sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback_per_frame)

# wait until EOS or error
bus = pipeline.get_bus()
start_time = time.time()
msg = bus.timed_pop_filtered(
    Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
)
end_time = time.time()
# Parse message
if msg:
    if msg.type == Gst.MessageType.ERROR:
        err, debug_info = msg.parse_error()
        logger.error(f"Error received from element {msg.src.get_name()}: {err.message}")
        logger.error(f"Debugging information: {debug_info if debug_info else 'none'}")
    elif msg.type == Gst.MessageType.EOS:
        logger.info("End-Of-Stream reached.")
        logger.info(f"FPS - {frames_processed / (end_time - start_time):.2f}")
    else:
        # This should not happen as we only asked for ERRORs and EOS
        logger.error("Unexpected message received.")

open(os.path.join(log_dir, "ssd_inference_pytorch_gst_pipeline.dot"), "w").write(
    Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
)

# free resources
pipeline.set_state(Gst.State.NULL)
