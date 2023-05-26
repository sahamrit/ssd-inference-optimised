"Generic utilities"
import os
from typing import *  # pylint: disable=W0401,W0614

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt


def plt_results(
    results: List[Tuple], inputs: np.array, output_path: os.path, ssd_utils
) -> None:
    """Plot results of object detection

    Parameters
    ----------
        results: List of tuple - (bboxes, class prob, conf)
        inputs: Image numpy array
        output_path: os.path
        ssd_utils: SSD utils from torch hub
    """
    classes_to_labels = ssd_utils.get_coco_object_dictionary()

    for image_idx, _ in enumerate(results):
        _, axes = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        axes.imshow(image)
        # ...with detections
        bboxes, classes, confidences = results[image_idx]
        for idx, bbox in enumerate(bboxes):
            left, bot, right, top = bbox

            # pylint: disable=invalid-name
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
            )
            axes.add_patch(rect)
            axes.text(
                x,
                y,
                f"{classes_to_labels[classes[idx] - 1]} {confidences[idx] * 100:.0f}",
                bbox=dict(facecolor="white", alpha=0.5),
            )
    plt.savefig(output_path)
