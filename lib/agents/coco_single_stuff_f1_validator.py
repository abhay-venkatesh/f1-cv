from lib.agents.agent import Agent
from lib.datasets.coco_stuff_f1 import COCOSingleStuffF1
from lib.models.seg_net_f1 import SegNetF1
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class COCOSingleStuffF1Validator(Agent):
    N_CLASSES = 2

    def run(self):
        valset = COCOSingleStuffF1(
            Path(self.config["dataset path"], "val"),
            threshold=self.config["threshold"])
        val_loader = DataLoader(
            dataset=valset, batch_size=self.config["batch size"])

        model = SegNetF1(n_classes=self.N_CLASSES).to(self.device)

        for img, mask, _, _ in tqdm(val_loader):
            img, mask = img.to(self.device), mask.long().to(self.device)
            mask_, _ = model(img)
            _, predicted = torch.max(mask_, 1)
            print(mask)
            raise RuntimeError
