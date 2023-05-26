import gi
import logging
import numpy as np

gi.require_version("Gst", "1.0")

from gi.repository import Gst

logging.basicConfig(
    level=logging.DEBUG, format="[%(name)s] [%(levelname)8s] - %(message)s"
)
logger = logging.getLogger(__name__)


def get_buffer_size(caps):
    caps_struct = caps.get_structure(0)

    (success, width) = caps_struct.get_int("width")
    if not success:
        return False, (0, 0)

    (success, height) = caps_struct.get_int("height")
    if not success:
        return False, (0, 0)

    return True, (width, height)


def probe_callback_per_frame(pad, info):
    logger.debug(f"probe id - {info.id}")
    buf = info.get_buffer()
    logger.debug(f"buffer pts - [{buf.pts / Gst.SECOND:6.2f} sec]")
    success, map_info = buf.map(Gst.MapFlags.READ)

    if not success:
        raise RuntimeError("Could not map buffer data!")

    success, (width, height) = get_buffer_size(pad.get_current_caps())

    if not success:
        raise RuntimeError("Could not extract widht and height from pad caps!")

    logger.info(f"Extracted buffer of shape (H / W) ({height} / {width})")
    numpy_frame = np.ndarray(
        shape=(height, width, 4), dtype=np.uint8, buffer=map_info.data
    )
    logger.info(f"Converted to numpy array of shape - {numpy_frame.shape}")

    return Gst.PadProbeReturn.OK
