from mnistf1 import MNISTF1
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

PALETTE = sns.diverging_palette(220, 10, n=7)
FONT_SCALE = 4
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
    set_styles()
    plotobj = sns.kdeplot(xs, shade=True)
    plotobj.set_xlabel("Size of " + str(i))
    plotobj.set_ylabel("Frequency")
    # Aesthetics
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()

    fig.savefig(Path("hists", "hist-" + str(i) + ".png"))
    plt.clf()


if __name__ == "__main__":
    cache_file = "histogram.cache"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            sizes = pickle.load(fp)
    else:
        dataset = MNISTF1(Path.cwd(), download=True)
        sizes = {}
        for img, target in dataset:
            target = target.numpy()
            class_ = target[0]
            img = img.numpy()
            img[img > 0] = 1

            if class_ not in sizes.keys():
                sizes[class_] = []
            sizes[class_].append(img.sum() / (img.shape[0] * img.shape[1]))

        with open(cache_file, 'wb') as fp:
            pickle.dump(sizes, fp)

    for i in sizes.keys():
        plot_histogram(sizes[i], i)