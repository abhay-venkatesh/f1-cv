from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuff
from pathlib import Path
import importlib
import torch

# from pycocotools.cocostuffhelper import segmentationToCocoResult


class COCOStuffTestDev2017Evaluator(Agent):
    def run(self):
        raise NotImplementedError
        
        testset = COCOStuff(
            Path(self.config["dataset path"], "test"),
            is_cropped=self.config["is cropped"],
            crop_size=(self.config["img width"], self.config["img height"]),
            in_memory=self.config["in memory"])

        net_module = importlib.import_module(
            ("lib.models.{}".format(self.config["model"])))
        net = getattr(net_module, "build_" + self.config["model"])

        model = net(n_classes=self.N_CLASSES).to(self.device)

        model.eval()
        with torch.no_grad():
            for X, Y in testset:
                X, Y = X.to(self.device), Y.long().to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)
