"Baseline inference of NVIDIA SSD300 in Pytorch"
import sys
import os
import logging
import time
import torch
from gi.repository import Gst  # pylint: disable=no-name-in-module
import gi
import numpy as np
import nvtx

from utils.gst_utils import buffer_to_numpy
from utils.util import plt_results

torch.backends.cudnn.enabled = False
gi.require_version("Gst", "1.0")

logging.basicConfig(
    level=logging.INFO, format="[%(name)s] [%(levelname)8s] - %(message)s"
)
logger = logging.getLogger(__name__)

# global setting
NUM_BUFFERS = 256
VIDEOFORMAT = "RGBA"
LOG_DIR = "/home/azureuser/localfiles/Repo/ssd-inference-optimised/logs"
SSD_THRESHOLD = 0.4

# setup ssd eval
frames_processed = 0  # pylint: disable=invalid-name
device = "cuda" if torch.cuda.is_available() else "cpu"  # pylint: disable=invalid-name
ssd_model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd")
ssd_utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)

ssd_model.to(device)
ssd_model.eval()


def preprocess(img: np.array) -> np.array:
    """Preprocess image according to NVIDIA SSD -
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/dle/inference.py
    """
    img = ssd_utils.rescale(img, 300, 300)
    img = ssd_utils.crop_center(img, 300, 300)
    img = ssd_utils.normalize(img)

    return img


@nvtx.annotate("frame_probe", color="pink")
def probe_callback_per_frame(pad: Gst.Pad, info: Gst.PadProbeInfo):
    """Callback for sink pad. It detects objects per frame."""
    global frames_processed  # pylint: disable=global-statement,invalid-name

    with nvtx.annotate("preprocess buffer", color="green"):
        img = buffer_to_numpy(pad, info)
        img = preprocess(img)
        # pylint: disable=no-member
        input_tensor = torch.tensor(img, device=device, dtype=torch.float32)
        # pylint: disable=logging-fstring-interpolation
        logger.debug(
            f"""Input tensor max : {input_tensor.max()}, min : {input_tensor.min()} 
            and shape : {input_tensor.shape}"""
        )
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        with nvtx.annotate("ssd forward", color="yellow"):
            detections = ssd_model(input_tensor)

    # pylint: disable=logging-fstring-interpolation
    logger.debug(
        f"Detections bbox : {detections[0].shape}, class : {detections[1].shape}"
    )
    with nvtx.annotate("post processing", color="purple"):
        result = ssd_utils.decode_results(detections)
        best_result = ssd_utils.pick_best(result[0], SSD_THRESHOLD)

    if (frames_processed + 1) % 50 == 0:
        with nvtx.annotate("plot", color="blue"):
            plt_results(
                [best_result],
                [img],
                os.path.join(
                    LOG_DIR, f"ssd_infer_pytorch_baseline_{frames_processed}.png"
                ),
                ssd_utils,
            )
    frames_processed += 1
    return Gst.PadProbeReturn.OK


# initialize GStreamer
Gst.init(sys.argv[1:])

# build the pipeline
pipeline = Gst.parse_launch(
    f"filesrc location=media/in.mp4 num-buffers={NUM_BUFFERS} ! \
     decodebin ! \
     nvvideoconvert ! \
     video/x-raw, format = {VIDEOFORMAT} ! \
     fakesink name=fs"
)

# start playing
pipeline.set_state(Gst.State.PLAYING)

# add probe to sink pad
fs = pipeline.get_by_name("fs")
sink_pad = fs.get_static_pad("sink")
sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback_per_frame)

start_time = time.time()
# wait until EOS or error
with nvtx.annotate("video processing", color="red"):
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
    )
end_time = time.time()
# Parse message
if msg:
    if msg.type == Gst.MessageType.ERROR:
        err, debug_info = msg.parse_error()
        logger.error(
            "Error received from element {%s}: {%s}", msg.src.get_name(), err.message
        )
        logger.error(
            "Debugging information: {%s}", debug_info if debug_info else "none"
        )
    elif msg.type == Gst.MessageType.EOS:
        logger.info("End-Of-Stream reached.")
        logger.info("FPS - %.2f", frames_processed / (end_time - start_time))
    else:
        # This should not happen as we only asked for ERRORs and EOS
        logger.error("Unexpected message received.")
with open(
    os.path.join(LOG_DIR, "ssd_inference_pytorch_gst_pipeline.dot"),
    "w",
    encoding="utf-8",
) as f:
    f.write(Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL))

# free resources
pipeline.set_state(Gst.State.NULL)
