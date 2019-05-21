import pandas as pd
from pathlib import Path
import os
import seaborn as sns

PALETTE = sns.diverging_palette(220, 10, n=7)
FONT_SCALE = 1.5
STYLE = "ticks"
CONTEXT = "paper"


def set_styles():
    sns.set_context(CONTEXT)
    sns.set(style=STYLE, font_scale=FONT_SCALE)
    sns.set_palette(PALETTE)


def get_data():
    # F1 Data
    df = pd.read_csv(
        Path("data", "f1", "beta_0001_mu_003"), sep=',', header=None)
    df.columns = ["Epoch", "Mean IoU"]
    experiment = ["F1-Regularized"] * len(df)
    df["Experiment"] = experiment

    file_names = [
        f for f in os.listdir(Path("data", "f1"))
        if os.path.isfile(Path("data", "f1", f))
    ]
    for file_name in file_names:
        df_ = pd.read_csv(Path("data", "f1", file_name), sep=',', header=None)
        df_.columns = ["Epoch", "Mean IoU"]
        experiment = ["F1-Regularized"] * len(df_)
        df_["Experiment"] = experiment
        df = pd.concat([df, df_], axis=0)

    # Baseline Data
    df_ = pd.read_csv(Path("data", "baseline"), sep=',', header=None)
    df_.columns = ["Epoch", "Mean IoU"]
    df_.Epoch += 1
    experiment = ["Baseline"] * len(df_)
    df_["Experiment"] = experiment
    df = pd.concat([df, df_], axis=0)

    return df


def plot():
    data = get_data()
    plotobj = sns.lineplot(
        x="Epoch",
        y="Mean IoU",
        hue="Experiment",
        data=data,
        palette=sns.diverging_palette(220, 10, n=2))

    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()
    fig.savefig("coco_stuff_small_confidences.png")
    fig.savefig("coco_stuff_small_confidences.pdf")


def main():
    plot()


if __name__ == "__main__":
    set_styles()
    main()
