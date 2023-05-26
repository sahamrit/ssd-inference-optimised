import os

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def plt_results(results, inputs, output_path, ssd_utils):
    classes_to_labels = ssd_utils.get_coco_object_dictionary()

    for image_idx in range(len(results)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = results[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                "{} {:.0f}%".format(
                    classes_to_labels[classes[idx] - 1], confidences[idx] * 100
                ),
                bbox=dict(facecolor="white", alpha=0.5),
            )
    plt.savefig(output_path)
