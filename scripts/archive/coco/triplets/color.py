from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil


class Colorer:
    MIN_COLOR = 10
    MAX_COLOR = 220
    N_CLASSES = 7
    NO_CLASS = 0
    N_CLASSES = 92
    LAST_CLASS_LABEL = N_CLASSES - 1
    MAX_PIXEL_VAL = 255

    def __init__(self):
        self.colors = self._set_colors()

    def _set_colors(self):
        palette = sns.diverging_palette(
            self.MIN_COLOR,
            self.MAX_COLOR,
            sep=round(self.MAX_COLOR / self.N_CLASSES),
            n=self.N_CLASSES)
        sns.palplot(palette)
        plt.savefig("colors.png")
        plt.close()

        colors = []
        for color in palette:
            r, g, b, _ = color
            colors.append((int(r * self.MAX_PIXEL_VAL),
                           int(g * self.MAX_PIXEL_VAL),
                           int(b * self.MAX_PIXEL_VAL)))
        return colors

    def color_gt(self, img_file_path):
        gt = np.array(Image.open(img_file_path))
        for i in np.unique(gt):
            if i < self.N_CLASSES:
                gt[gt == i] = self.NO_CLASS

        red_layer = np.zeros(gt.shape)
        blue_layer = np.zeros(gt.shape)
        green_layer = np.zeros(gt.shape)
        available_colors = self.colors
        used_colors = {}
        for i in np.unique(gt):
            if i == self.NO_CLASS or i == self.MAX_PIXEL_VAL:
                pass
            else:
                r, g, b = available_colors.pop()
                red_layer[gt == i] = r
                green_layer[gt == i] = g
                blue_layer[gt == i] = b
                used_colors[i] = (r, g, b)
        gt_colored = np.dstack([red_layer, green_layer, blue_layer])
        Image.fromarray(
            np.uint8(gt_colored)).save(img_file_path.stem + "_colored.png")
        return used_colors, available_colors

    def color_output(self, img_file_path, used_colors, available_colors):
        output = np.array(Image.open(img_file_path))
        output += self.LAST_CLASS_LABEL
        output[output == self.LAST_CLASS_LABEL] = self.NO_CLASS

        red_layer = np.zeros(output.shape)
        blue_layer = np.zeros(output.shape)
        green_layer = np.zeros(output.shape)

        for i in np.unique(output):
            if i == self.NO_CLASS:
                continue
            if i in used_colors.keys():
                r, g, b = used_colors[i]
            else:
                r, g, b = available_colors.pop()
                used_colors[i] = (r, g, b)
            red_layer[output == i] = r
            green_layer[output == i] = g
            blue_layer[output == i] = b
        output_colored = np.dstack([red_layer, green_layer, blue_layer])
        Image.fromarray(
            np.uint8(output_colored)).save(img_file_path.stem + "_colored.png")
        return used_colors, available_colors


def color():
    output_names = [
        f for f in os.listdir(Path("outputs"))
        if os.path.isfile(Path("outputs", f))
    ]
    for output_name in output_names:
        colorer = Colorer()
        used_colors, available_colors = colorer.color_gt(
            Path(output_name.replace(".png", "") + "_gt.png"))
        used_colors, available_colors = colorer.color_output(
            Path("outputs", output_name), used_colors, available_colors)
        colorer.color_output(
            Path("baselines",
                 output_name.replace(".png", "") + "_baseline.png"))

        img_folder = Path("D:/code/data/cocostuff/dataset/images/val2017")
        shutil.copy2(
            Path(img_folder, output_name.replace(".png", ".jpg")),
            Path("images"))


if __name__ == "__main__":
    color()
