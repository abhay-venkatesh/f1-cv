from mnistf1 import MNISTF1
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = sns.diverging_palette(220, 10, n=7)
FONT_SCALE = 1.2
STYLE = "ticks"
CONTEXT = "paper"


def set_styles():
    sns.set(
        style=STYLE,
        context=CONTEXT,
        font_scale=FONT_SCALE,
    )
    sns.set_palette(PALETTE)


def plot_histogram(xs, i):
    plotobj = sns.kdeplot(xs, shade=True)
    # Aesthetics
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()

    fig.savefig(Path("hists", "hist-" + str(i) + "-.png"))
    plt.clf()


if __name__ == "__main__":
    dataset = MNISTF1(Path.cwd(), download=True)
    sizes = {}
    for img, target in dataset:
        target = target.numpy()
        class_ = target[0]
        img = img.numpy()
        img[img > 0] = 1

        if class_ not in sizes.keys():
            sizes[class_] = []
        sizes[class_].append(img.sum())

    for i in sizes.keys():
        plot_histogram(sizes[i], i)