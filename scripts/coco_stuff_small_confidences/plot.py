import pandas as pd
from pathlib import Path
import os
import seaborn as sns


def set_styles():
    # Seaborn
    sns.set_context("paper")
    sns.set_style("ticks")
    # sns.set_palette(sns.color_palette("RdBu_r", 7))
    sns.set_palette(sns.diverging_palette(220, 20, n=7))


def get_data():
    df = pd.read_csv(
        Path("data", "f1", "beta_0001_mu_003"), sep=',', header=None)
    df.columns = ["Epoch", "Mean IoU"]

    file_names = [
        f for f in os.listdir(Path("data", "f1"))
        if os.path.isfile(Path("data", "f1", f))
    ]
    for file_name in file_names:
        df_ = pd.read_csv(Path("data", "f1", file_name), sep=',', header=None)
        df_.columns = ["Epoch", "Mean IoU"]
        df = pd.concat([df, df_], axis=0)

    return df


def plot(df):
    plotobj = sns.lineplot(x="Epoch", y="Mean IoU", data=df)
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()
    fig.savefig("test.png")


def main():
    df = get_data()
    plot(df)


if __name__ == "__main__":
    set_styles()
    main()
