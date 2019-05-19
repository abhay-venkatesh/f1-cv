from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuffEval
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import importlib
import torch

# from pycocotools.cocostuffhelper import segmentationToCocoResult


class COCOStuffEvaluator(Agent):
    N_CLASSES = 92

    def run(self):
        testset = COCOStuffEval(self.config["dataset path"])
        test_loader = DataLoader(dataset=testset, batch_size=2)

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
            for X in test_loader:
                X = X.to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)
                print(predicted)
                raise RuntimeError
