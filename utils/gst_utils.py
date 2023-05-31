"Utilities for Gstreamer"
import logging

from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from gi.repository import Gst  # pylint: disable=E0611

import gi
import torch
from utils import nvds

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


def buffer_to_image_tensor(pad, info, device):
    """Converts buffer to tensor. The map_info of Gst.Buffer when
    using deepstream follows NvBufSurface structure. Hence to
    deserialise the string we need python-c++ binding (see nvds)"""
    buf = info.get_buffer()

    _, (width, height) = get_buffer_size(pad.get_current_caps())
    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if not is_mapped:
        raise RuntimeError("Could not map buffer data!")
    if is_mapped:
        try:
            source_surface = nvds.NvBufSurface(map_info)
            torch_surface = nvds.NvBufSurface(map_info)
            # pylint: disable=no-member
            dest_tensor = torch.zeros(
                (
                    height,
                    width,
                    4,
                ),
                dtype=torch.uint8,
                device=device,
            )

            torch_surface.struct_copy_from(source_surface)
            assert source_surface.numFilled == 1
            assert source_surface.surfaceList[0].colorFormat == 19  # RGBA

            # make torch_surface map to dest_tensor memory
            torch_surface.surfaceList[0].dataPtr = dest_tensor.data_ptr()

            # copy decoded GPU buffer (source_surface) into
            # Pytorch tensor (torch_surface -> dest_tensor)
            torch_surface.mem_copy_from(source_surface)
        finally:
            buf.unmap(map_info)

        return dest_tensor[:, :, :3]
