import pandas as pd
from pathlib import Path
import os


def get_data():
    df = pd.read_csv(
        Path("data", "f1", "beta_0001_mu_003"), sep=',', header=None)
    df = df.drop(0, 1)
    df.columns = ["b0001m003"]

    file_names = [
        f for f in os.listdir(Path("data", "f1"))
        if os.path.isfile(Path("data", "f1", f))
    ]
    for file_name in file_names:
        df_ = pd.read_csv(Path("data", "f1", file_name), sep=',', header=None)
        df_ = df_.drop(0, 1)
        df_.columns = [file_name]
        df = pd.concat([df, df_], axis=1)

    return df


if __name__ == "__main__":
    df = get_data()
    print(df)