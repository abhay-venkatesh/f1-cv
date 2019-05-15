from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuff
from lib.utils.functional import get_iou, cross_entropy2d
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import torch


class COCOStuffBaselineTrainer(Agent):
    N_CLASSES = 92

    def run(self):
        trainset = COCOStuff(Path(self.config["dataset path"], "train"))
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.config["batch size"],
            shuffle=True)


        net_module = importlib.import_module(
            ("lib.models.{}".format(self.config["model"])))
        net = getattr(net_module, "build_" + self.config["model"])

        model = net(n_classes=self.N_CLASSES).to(self.device)
        start_epochs = self._load_checkpoint(model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning rate"])

        for epoch in tqdm(range(start_epochs, self.config["epochs"])):

            model.train()
            total_loss = 0
            for X, Y in tqdm(train_loader):
                X, Y = X.to(self.device), Y.long().to(self.device)
                Y_ = model(X)
                loss = cross_entropy2d(Y_, Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(train_loader)
            self.logger.log("epoch", epoch, "loss", avg_loss)


            self.logger.graph()

            self._save_checkpoint(epoch, model)
