from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = np.array(Image.open(img_path))
        if i == 0:
            output = img
        else:
            output = concat_images(output, img)
    return output


if __name__ == "__main__":
    images_folder = Path("C:/Users/abhay/code/src/f1-cv/scripts/archive/" +
                         "coco/triplets/images")
    baselines_folder = Path(
        "C:/Users/abhay/code/src/f1-cv/scripts/archive/coco/triplets/" +
        "baselines_colored")
    outputs_folder = Path(
        "C:/Users/abhay/code/src/f1-cv/scripts/archive/coco/triplets/" +
        "outputs_colored")
    gt_folder = Path(
        "C:/Users/abhay/code/src/f1-cv/scripts/archive/coco/triplets/" +
        "gt_colored")
    merged_folder = Path(
        "C:/Users/abhay/code/src/f1-cv/scripts/archive/coco/triplets/" +
        "merged")

    image_names = [
        f.replace(".jpg", "") for f in os.listdir(images_folder)
        if os.path.isfile(Path(images_folder, f))
    ]

    for i, image_name in enumerate(image_names):
        merged = concat_n_images([
            Path(images_folder, image_name + ".jpg"),
            Path(baselines_folder, image_name + "_colored.png"),
            Path(outputs_folder, image_name + "_colored.png"),
            Path(gt_folder, image_name + "_colored.png"),
        ])
        Image.fromarray(np.uint8(merged)).save(Path(merged_folder, str(i) + ".png"))
