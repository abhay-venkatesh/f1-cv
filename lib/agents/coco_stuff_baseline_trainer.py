from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuff
from lib.models.deep_lab import build_deep_lab
from lib.utils.functional import cross_entropy2d, get_iou
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from lib.utils.modular import ModularizedFunction, CriterionParallel


class COCOStuffBaselineTrainer(Agent):
    N_CLASSES = 92

    def run(self):
        trainset = COCOStuff(Path(self.config["dataset path"], "train"))
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.config["batch size"],
            shuffle=True)

        valset = COCOStuff(Path(self.config["dataset path"], "val"))
        val_loader = DataLoader(
            dataset=valset, batch_size=self.config["batch size"])

        model = build_deep_lab(n_classes=self.N_CLASSES).to(self.device)
        start_epochs = self._load_checkpoint(model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning rate"])

        # Parallelize loss computation
        loss_fn = ModularizedFunction(cross_entropy2d)
        if torch.cuda.device_count() > 1:
            loss_fn = CriterionParallel(loss_fn)

        for epoch in tqdm(range(start_epochs, self.config["epochs"])):

            model.train()
            total_loss = 0
            for X, Y in tqdm(train_loader):
                X, Y = X.to(self.device), Y.long().to(self.device)
                Y_ = model(X)
                loss = loss_fn(Y_, Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(train_loader)
            self.logger.log("epoch", epoch, "loss", avg_loss)

            model.eval()
            ious = []
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(self.device), Y.long().to(self.device)
                    Y_ = model(X)
                    _, predicted = torch.max(Y_.data, 1)
                    iou = get_iou(predicted, Y)
                    ious.append(iou.item())

            mean_iou = mean(ious)
            self.logger.log("epoch", epoch, "iou", mean_iou)

            self.logger.graph()

            self._save_checkpoint(epoch, model)
