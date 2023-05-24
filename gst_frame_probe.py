import sys
import logging
import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst


logging.basicConfig(
    level=logging.DEBUG, format="[%(name)s] [%(levelname)8s] - %(message)s"
)
logger = logging.getLogger(__name__)

# global setting
videoformat = "RGBA"


def probe_callback_per_frame(pad, info):
    logger.debug(f"probe id - {info.id}")
    buf = info.get_buffer()
    logger.debug(f"buffer pts - [{buf.pts / Gst.SECOND:6.2f} sec]")
    return Gst.PadProbeReturn.OK


# initialize GStreamer
Gst.init(sys.argv[1:])

# build the pipeline
pipeline = Gst.parse_launch(
    f"filesrc location=media/in.mp4 ! \
     decodebin ! \
     nvvideoconvert ! \
     video/x-raw(memory:NVMM), format = {videoformat} ! \
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
msg = bus.timed_pop_filtered(
    Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
)

# Parse message
if msg:
    if msg.type == Gst.MessageType.ERROR:
        err, debug_info = msg.parse_error()
        logger.error(f"Error received from element {msg.src.get_name()}: {err.message}")
        logger.error(f"Debugging information: {debug_info if debug_info else 'none'}")
    elif msg.type == Gst.MessageType.EOS:
        logger.info("End-Of-Stream reached.")
    else:
        # This should not happen as we only asked for ERRORs and EOS
        logger.error("Unexpected message received.")

open(f"logs/gst_baseline.dot", "w").write(
    Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
)

# free resources
pipeline.set_state(Gst.State.NULL)
