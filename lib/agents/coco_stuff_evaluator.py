from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuffEval
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import importlib
import torch
from math import floor
from torchvision import transforms

# from pycocotools.cocostuffhelper import segmentationToCocoResult


class COCOStuffEvaluator(Agent):
    N_CLASSES = 92
    WINDOW_SIZE = 320

    def run(self):
        testset = COCOStuffEval(self.config["dataset path"])

        net_module = importlib.import_module(
            ("lib.models.{}".format(self.config["model"])))
        net = getattr(net_module, "build_" + self.config["model"])

        model = net(
            n_classes=self.N_CLASSES,
            size=(self.config["img width"],
                  self.config["img height"])).to(self.device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(Path(self.config["checkpoint path"])))

        model.eval()
        with torch.no_grad():
            for img in testset:
                _, h, w = img.shape
                windows, h_overlaps, w_overlaps = self._get_windows(img)

                X = torch.stack(windows).to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)
                seg = self._construct_mask(predicted, h, w)

                X = torch.stack(h_overlaps).to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)
                seg = self._apply_h_overlaps(predicted, seg, w)

                X = torch.stack(w_overlaps).to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)
                seg = self._apply_w_overlaps(predicted, seg, h)

                seg_img = transforms.ToPILImage()(seg)
                seg_img.save(Path(self.config["outputs folder"]))
                raise RuntimeError

    def _construct_mask(self, predicted, h, w):
        seg = torch.zeros((h, w))
        num_h_fits = h / self.WINDOW_SIZE
        num_w_fits = w / self.WINDOW_SIZE
        k = 0
        for i in range(0, floor(num_h_fits)):
            for j in range(0, floor(num_w_fits)):
                h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
                w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
                seg[h1:h2, w1:w2] = predicted[k, :, :]
                k += 1
        return seg

    def _apply_h_overlaps(self, predicted, seg, w):
        pass

    def _apply_w_overlaps(self, predicted, seg, h):
        pass

    def _get_windows(self, img):
        _, h, w = img.shape
        num_h_fits = h / self.WINDOW_SIZE
        num_w_fits = w / self.WINDOW_SIZE
        windows = []
        for i in range(0, floor(num_h_fits)):
            for j in range(0, floor(num_w_fits)):
                h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
                w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
                windows.append(img[:, h1:h2, w1:w2])

        h_overlaps = []
        if not num_h_fits.is_integer():
            for j in range(0, floor(num_w_fits)):
                h1, h2 = h - self.WINDOW_SIZE, h
                w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
                windows.append(img[:, h1:h2, w1:w2])

        w_overlaps = []
        if not num_w_fits.is_integer():
            for i in range(0, floor(num_h_fits)):
                h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
                w1, w2 = w - self.WINDOW_SIZE, w
                windows.append(img[:, h1:h2, w1:w2])

        return windows, h_overlaps, w_overlaps
