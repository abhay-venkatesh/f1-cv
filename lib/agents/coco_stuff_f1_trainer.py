from lib.datasets.coco_stuff_f1 import COCOStuffF1
from lib.models.segnet import get_model
from lib.trainers.functional import cross_entropy2d, get_iou, lagrange
from lib.trainers.agent import Agent
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class COCOStuffF1Trainer(Agent):
    def run(self):
        # Training dataset
        trainset = COCOStuffF1(
            Path(self.config["dataset path"], "train"))
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.config["batch size"],
            shuffle=True)

        # Validation dataset
        valset = COCOStuffF1(
            Path(self.config["dataset path"], "val"))
        val_loader = DataLoader(
            dataset=valset, batch_size=self.config["batch size"])

        # Constants
        num_positives = self.train_loader.dataset.num_positives

        # Primal variables
        tau = torch.rand(
            len(self.train_loader.dataset),
            device=self.device,
            requires_grad=True)
        eps = torch.rand(1, device=self.device, requires_grad=True)
        w = torch.rand(1, device=self.device, requires_grad=True)

        # Dual variables
        lamb = torch.zeros(len(self.train_loader.dataset), device=self.device)
        lamb.fill_(0.001)
        mu = torch.zeros(1, device=self.device)
        mu.fill_(0.001)
        gamma = torch.zeros(1, device=self.device)
        gamma.fill_(0.001)

        # Temporary variables for dual updates
        tau_1 = 0.0
        tau_eps = 0.0
        tau_w_y = torch.zeros(len(self.train_loader.dataset)).to(self.device)

        # Primal Optimization
        var_list = [{
            "params": self.model.parameters(),
            "lr": self.config["lr"]
        }, {
            "params": tau,
            "lr": self.config["eta_tau"]
        }, {
            "params": eps,
            "lr": self.config["eta_eps"]
        }, {
            "params": w,
            "lr": self.config["eta_w"]
        }]

        # Model and optimizer
        model = get_model(n_classes=trainset.N_CLASSES).to(self.device)
        optimizer = torch.optim.Adam(
            var_list, lr=self.config["learning rate"])

        # Dataset iterator
        train_iter = iter(train_loader)
        for epoch in tqdm(range(self.start_epochs, self.config["epochs"])):
            total_loss = 0
            model.train()
            for _ in tqdm(range(self.config["n_inner"])):
                # Sample
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X, Y = next(train_iter)

                # Forward computation
                X, Y = X.to(self.device), Y.to(self.device)
                Y_, y_ = self.model(X)
                y = Y[:, 1]
                i = Y[:, 2]

                # Loss business
                lagrangian = lagrange(num_positives, y_, y, w, eps, tau[i],
                                      lamb[i], mu, gamma, self.device)
                loss = cross_entropy2d(Y_,
                                       Y) + (self.config["beta"] * lagrangian)
                total_loss += loss.item()

                # Backpropagate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Project eps to ensure non-negativity
                eps.data = torch.max(
                    torch.zeros(1, dtype=torch.float, device=self.device),
                    eps.data)

                # Cache for Dual updates
                y = y.float()
                y_ = y_.view(-1)
                tau_1 += ((y * tau[i]).sum() - 1)
                tau_eps += ((y * tau[i]).sum() - eps)
                tau_w_y[i] += (y * (tau[i] - (w * y_))).sum()

            # Dual updates
            lamb.data = lamb.data + (self.config["eta_lamb"] * tau_w_y)
            mu.data = mu.data + (self.config["eta_mu"] * tau_1)
            gamma.data = gamma.data + (self.config["eta_gamma"] * tau_eps)

            # Log loss
            avg_loss = total_loss / len(self.train_loader)
            self.logger.log("epoch", epoch, "avg_loss", avg_loss)

            # Validate
            model.eval()
            ious = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels[0].long()
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    iou = get_iou(predicted, labels)
                    ious.append(iou.item())

            # Log mean IOU
            mean_iou = mean(ious)
            self.logger.log("epoch", epoch, "mean_iou", mean_iou)

            # Graph
            self.logger.graph()

            # Checkpoint
            torch.save(
                model.state_dict(),
                Path(self.checkpoints_folder,
                     str(epoch + 1) + ".ckpt"))
