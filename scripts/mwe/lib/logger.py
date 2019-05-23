from pathlib import Path
import csv
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, folder):
        self.folder = folder
        self.filenames = set()

    def log(self, keyname, key, valname, val):
        filename = keyname + "_" + valname
        if filename not in self.filenames:
            self.filenames.add(filename)

        with open(
                Path(self.folder, filename), mode='a', newline='') as logfile:
            writer = csv.writer(
                logfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([key, val])

    def graph(self):
        for filename in self.filenames:
            self._plot(filename)

    def _plot(self, filename):
        xs, ys = self._load(filename)
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.set_xlabel(Path(filename).stem.split("_")[0])
        ax.set_ylabel(Path(filename).stem.split("_")[1])
        plt.savefig(Path(self.folder, Path(filename).stem + ".png"))
        plt.close(fig)

    def _load(self, filename):
        xs, ys = [], []
        with open(Path(self.folder, filename)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
        return xs, ys