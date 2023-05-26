"Utilities for Gstreamer"
import logging

from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from gi.repository import Gst  # pylint: disable=E0611

import gi
import numpy as np

gi.require_version("Gst", "1.0")
logger = logging.getLogger(__name__)


def get_buffer_size(caps: Gst.Caps) -> Tuple[bool, Tuple[int, int]]:
    """Get size of buffer via current capability of pad"""
    caps_struct = caps.get_structure(0)

    (success, width) = caps_struct.get_int("width")
    if not success:
        return False, (0, 0)

    (success, height) = caps_struct.get_int("height")
    if not success:
        return False, (0, 0)

    return True, (width, height)


def buffer_to_numpy(pad: Gst.Pad, info: Gst.PadProbeInfo) -> np.array:
    """Convert Gst.Buf to Numpy Array"""

    buf = info.get_buffer()
    logger.debug("buffer pts - [%6.2f sec]", buf.pts / Gst.SECOND)

    success, map_info = buf.map(Gst.MapFlags.READ)

    if not success:
        raise RuntimeError("Could not map buffer data!")
    success, (width, height) = get_buffer_size(pad.get_current_caps())

    if not success:
        raise RuntimeError("Could not extract widht and height from pad caps!")
    logger.debug("Extracted buffer of shape (H / W) (%d / %d)", height, width)

    numpy_frame = np.ndarray(
        shape=(height, width, 4), dtype=np.uint8, buffer=map_info.data
    )
    result = numpy_frame[:, :, :3].copy()

    # pylint: disable=W1203
    logger.debug(f"Converted to numpy array of shape - {result.shape}")

    buf.unmap(map_info)
    return result
