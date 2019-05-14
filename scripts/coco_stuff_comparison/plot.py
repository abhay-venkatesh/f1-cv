import csv
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pylab as pylab
params = {
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}
pylab.rcParams.update(params)

if __name__ == "__main__":
    baseline_ys = []
    with open(Path("./baseline")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            baseline_ys.append(float(row[1]))

    f1_ys = []
    with open(Path("./f1")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            f1_ys.append(float(row[1]))

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(baseline_ys, label="baseline")
    ax.plot(f1_ys, label="f1 normalized")
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Mean IoU", fontsize=14)
    ax.legend()
    plt.savefig("coco_stuff_sky.png")