# pylint: disable-all
"Baseline inference of NVIDIA SSD300 in Pytorch"
import sys
import os
import logging
import time
import threading
import queue
import torch
import gi
import numpy as np
import nvtx
import torchvision
from gi.repository import Gst  # pylint: disable=no-name-in-module
from torchvision import transforms

from utils.gst_utils import buffer_to_numpy
from utils.util import plt_results
from utils.ssd import Encoder, dboxes300_coco

# torch.backends.cudnn.enabled = False
gi.require_version("Gst", "1.0")

logging.basicConfig(
    level=logging.INFO, format="[%(name)s] [%(levelname)8s] - %(message)s"
)
logger = logging.getLogger(__name__)

# global setting
NUM_BUFFERS = 1000
DISABLE_PLOTTING = True
BATCH_SIZE = 64
NUM_THREADS = 2

VIDEOFORMAT = "RGBA"
MODEL_PRECISION = "fp32"
MODEL_DTYPE = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32
LOG_DIR = "/home/azureuser/localfiles/Repo/ssd-inference-optimised/logs"
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45

# setup ssd eval
frames_processed = 0  # pylint: disable=invalid-name
start_time = time.time()
thread_queues = [queue.Queue(2 * BATCH_SIZE) for _ in range(NUM_THREADS)]
threads = []
device = (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # pylint: disable=invalid-name
ssd_model = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=MODEL_PRECISION
)
ssd_utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)

ssd_model.to(device)
ssd_model.eval()

transform = transforms.Compose(
    [
        transforms.CenterCrop(300),
        transforms.Normalize(127.5, 127.5),
    ]
)


def postprocess(detections_batch):
    """Input = Tuple [ ploc , pconf ].
    ploc shape - [ bsz, 4, num_boxes ]
    pconf shape - [bsz, 81, num_boxes ]

    There are 81 classes and 8732 num_boxes
    """
    encoder = Encoder(dboxes300_coco())
    bboxes, probs = encoder.scale_back_batch(detections_batch[0], detections_batch[1])
    output = []
    for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
        bbox = bbox.squeeze(0)
        prob = prob.squeeze(0)

        label = torch.max(prob, dim=-1)  # pylint: disable=no-member
        lbl = label.indices
        score = label.values
        mask = (score > CONF_THRESHOLD) & (lbl > 0)
        bbox, lbl, score = bbox[mask, :], lbl[mask], score[mask]
        max_ids = torchvision.ops.batched_nms(bbox, score, lbl, NMS_THRESHOLD)
        output.append(
            [bbox[max_ids, :].cpu(), lbl[max_ids].cpu(), score[max_ids].cpu()]
        )
    return output


def preprocess(img: torch.Tensor) -> torch.Tensor:
    """Preprocess image according to NVIDIA SSD -
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/dle/inference.py
    """
    return transform(img)


def preprocess_img(img: np.array) -> np.array:
    """Preprocess image according to NVIDIA SSD -
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/dle/inference.py
    """
    img = ssd_utils.rescale(img, 300, 300)
    img = ssd_utils.crop_center(img, 300, 300)
    img = ssd_utils.normalize(img)

    return img


def inference_per_thread(thread_idx, image_queue: queue.Queue):
    stream = torch.cuda.Stream(device)
    input_batch_buffer = []
    exit_flag = False
    while True:
        tensor, img, frame_id = image_queue.get()

        if tensor is None:
            exit_flag = True

        if not exit_flag:
            input_batch_buffer.append(tensor)
            if len(input_batch_buffer) < BATCH_SIZE:
                continue

        if len(input_batch_buffer) == 0:
            return
        with torch.cuda.stream(stream):
            with nvtx.annotate("preprocess", color="green"):
                # pylint: disable=no-member
                input_tensor = torch.stack(input_batch_buffer).to(device)
                input_tensor = preprocess(input_tensor)
                input_batch_buffer = []

            # # pylint: disable=logging-fstring-interpolation
            logger.debug(
                f"""Input tensor max : {input_tensor.max()}, min : {input_tensor.min()}
                and shape : {input_tensor.shape} and device: {input_tensor.device}"""
            )

            with torch.no_grad():
                with nvtx.annotate("ssd forward", color="yellow"):
                    detections = ssd_model(input_tensor)

            # pylint: disable=logging-fstring-interpolation
            logger.debug(
                f"Detections bbox : {detections[0].shape}, class : {detections[1].shape}"
            )
            with nvtx.annotate("post processing", color="purple"):
                best_results_per_input = postprocess(detections)
                best_result = best_results_per_input[-1]

        if frame_id and (not DISABLE_PLOTTING):
            with nvtx.annotate("plot", color="blue"):
                plt_results(
                    [best_result],
                    [preprocess_img(img)],
                    os.path.join(LOG_DIR, f"ssd_infer_pytorch_{frame_id}.png"),
                    ssd_utils,
                )
        if exit_flag:
            return


@nvtx.annotate("frame_probe", color="pink")
def probe_callback_per_frame(pad: Gst.Pad, info: Gst.PadProbeInfo):
    """Callback for sink pad. It detects objects per frame."""
    global frames_processed, thread_queues, start_time  # pylint: disable=global-statement,invalid-name
    frames_processed += 1
    with nvtx.annotate("buffer to numpy", color="green"):
        img = buffer_to_numpy(pad, info)

        # pylint: disable=no-member
        thread_queues[frames_processed % NUM_THREADS].put(
            (
                torch.tensor(img, dtype=MODEL_DTYPE).permute(2, 0, 1),
                img,  # for plotting
                frames_processed,
            )
        )
        logger.debug(
            f"Thread Queue len - {thread_queues[frames_processed % NUM_THREADS].qsize()}. Frames - {frames_processed}"
        )
        start_time = time.time() if frames_processed == 1 else start_time
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

for thread_idx in range(NUM_THREADS):
    curr_thread = threading.Thread(
        target=inference_per_thread, args=(thread_idx, thread_queues[thread_idx])
    )
    curr_thread.start()
    threads.append(curr_thread)

# wait until EOS or error

bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(
    Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
)

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

        for tidx, t in enumerate(threads):
            thread_queues[tidx].put((None, None, None))
            t.join()

        end_time = time.time()
        logger.info(
            "FPS - %.2f, Total frames - {%d}",
            frames_processed / (end_time - start_time),
            frames_processed,
        )
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
