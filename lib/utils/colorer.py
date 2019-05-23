from PIL import Image
import numpy as np
import seaborn as sns


class Colorer:
    def __init__(self):
        self.colors = self._set_colors()

    def _set_colors(self):
        palette = sns.diverging_palette(220, 10, sep=20, n=11)

        colors = []
        for color in palette:
            r, g, b, _ = color
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def color(self, gt):
        red_layer = np.zeros(gt.shape)
        blue_layer = np.zeros(gt.shape)
        green_layer = np.zeros(gt.shape)
        available_colors = self.colors
        used_colors = {}
        for i in np.unique(gt):
            if i != 0:
                r, g, b = available_colors.pop()
                red_layer[gt == i] = r
                green_layer[gt == i] = g
                blue_layer[gt == i] = b
                used_colors[i] = (r, g, b)
        gt_colored = np.dstack([red_layer, green_layer, blue_layer])
        return Image.fromarray(np.uint8(gt_colored)), used_colors, \
            available_colors

    def color_output(self, output, used_colors, available_colors):
        red_layer = np.zeros(output.shape)
        blue_layer = np.zeros(output.shape)
        green_layer = np.zeros(output.shape)

        for i in np.unique(output):
            if i != 0:
                if i in used_colors.keys():
                    r, g, b = used_colors[i]
                else:
                    r, g, b = available_colors.pop()
                red_layer[output == i] = r
                green_layer[output == i] = g
                blue_layer[output == i] = b
        output_colored = np.dstack([red_layer, green_layer, blue_layer])
        return Image.fromarray(np.uint8(output_colored))
