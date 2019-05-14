from pathlib import Path
import csv
import matplotlib.pyplot as plt

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
    with open(Path("./epoch_accuracy_baseline")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            baseline_ys.append(float(row[1]))

    f1_ys = []
    with open(Path("./epochs_accuracy_f1")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            f1_ys.append(float(row[1]))

    f1_fp_ys = []
    with open(Path("./epochs_accuracy_f1_fp")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            f1_fp_ys.append(float(row[1]))

    f1_ap_ys = []
    with open(Path("./epochs_accuracy_f1_ap")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            f1_ap_ys.append(float(row[1]))

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(baseline_ys, label="baseline")
    ax.plot(f1_ys, label="f1 normalized")
    ax.plot(f1_fp_ys, label="f1 normalized (full projections)")
    ax.plot(f1_ap_ys, label="f1 normalized (approx. projections)")
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.legend()
    plt.savefig("mnist_big_small.png")