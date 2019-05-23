from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLORS = [(217, 58, 70), (233, 142, 149), (250, 230, 231), (242, 242, 242),
          (233, 242, 245), (147, 184, 195), (63, 127, 147)]


class Colorer:
    def __init__(self):
        self.colors = self._set_colors()

    def _set_colors(self):
        palette = sns.diverging_palette(10, 220, sep=20, n=11)
        sns.palplot(palette)
        plt.savefig("colors.png")
        plt.close()

        colors = []
        for color in palette:
            r, g, b, _ = color
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def color_gt(self, img_file_path):
        gt = np.array(Image.open(img_file_path))
        for i in np.unique(gt):
            if i < 92:
                gt[gt == i] = 0

        red_layer = np.zeros(gt.shape)
        blue_layer = np.zeros(gt.shape)
        green_layer = np.zeros(gt.shape)
        available_colors = self.colors
        used_colors = {}
        for i in np.unique(gt):
            if i == 0 or i == 104 or i == 255:
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
        output += 91
        output[output == 91] = 0

        red_layer = np.zeros(output.shape)
        blue_layer = np.zeros(output.shape)
        green_layer = np.zeros(output.shape)

        for i in np.unique(output):
            if i == 0:
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


if __name__ == "__main__":
    colorer = Colorer()
    used_colors, available_colors = colorer.color_gt(
        Path("000000359781_gt.png"))
    used_colors, available_colors = colorer.color_output(
        Path("000000359781.png"), used_colors, available_colors)
    colorer.color_output(
        Path("000000359781_baseline.png"), used_colors, available_colors)
