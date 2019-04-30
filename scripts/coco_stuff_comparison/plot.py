import csv
import matplotlib.pyplot as plt
from pathlib import Path

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
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean IoU")
    ax.legend()
    plt.savefig("epochs_accuracy_comparison.png")