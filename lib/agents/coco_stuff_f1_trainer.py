from lib.datasets.coco_stuff_f1 import COCOStuffF1
from lib.models.segnet import get_model
from lib.trainers.agent import Agent
from lib.trainers.functional import cross_entropy2d, get_iou, lagrange
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class COCOStuffF1Trainer(Agent):
    def run(self):
        # Training dataset
        trainset = COCOStuffF1(Path(self.config["dataset path"], "train"))
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.config["batch size"],
            shuffle=True)

        # Validation dataset
        valset = COCOStuffF1(Path(self.config["dataset path"], "val"))
        val_loader = DataLoader(
            dataset=valset, batch_size=self.config["batch size"])

        # Constants
        num_positives = train_loader.dataset.num_positives

        # Primal variables
        tau = torch.rand(
            len(train_loader.dataset), device=self.device, requires_grad=True)
        eps = torch.rand(1, device=self.device, requires_grad=True)
        w = torch.rand(1, device=self.device, requires_grad=True)

        # Dual variables
        lamb = torch.zeros(len(train_loader.dataset), device=self.device)
        lamb.fill_(0.001)
        mu = torch.zeros(1, device=self.device)
        mu.fill_(0.001)
        gamma = torch.zeros(len(train_loader.dataset), device=self.device)
        gamma.fill_(0.001)

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
        optimizer = torch.optim.Adam(var_list, lr=self.config["learning rate"])

        # Load checkpoint if exists
        start_epochs = self._load_checkpoint(model)

        # Dataset iterator
        train_iter = iter(train_loader)

        # Count epochs and steps
        epochs = 0
        step = 0

        # Cache losses
        total_loss = 0
        total_t1_loss = 0
        total_t2_loss = 0

        for outer in tqdm(range(start_epochs, self.config["n_outer"])):
            model.train()

            for inner in tqdm(range(self.config["n_inner"])):
                step += 1

                # Sample
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X, Y = next(train_iter)

                # Forward computation
                X, Y = X.to(self.device), Y.to(self.device)
                y0_, y1_ = model(X)
                y0 = Y[:, 0]
                y1 = Y[:, 1]
                i = Y[:, 2]

                # Compute loss
                t1_loss = cross_entropy2d(y0_, y0)
                t2_loss = lagrange(num_positives, y1_, y1, w, eps, tau[i],
                                   lamb[i], mu, gamma[i])
                loss = t1_loss + (self.config["beta"] * t2_loss)

                # Store losses for logging
                total_loss += loss.item()
                total_t1_loss += t1_loss.item()
                total_t2_loss += t2_loss.item()

                # Backpropagate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Project eps to ensure non-negativity
                eps.data = torch.max(
                    torch.zeros(1, dtype=torch.float, device=self.device),
                    eps.data)

                # Log and validate per epoch
                if (step + 1) % len(train_loader) == 0:
                    epochs += 1

                    # Log loss
                    avg_loss = total_loss / len(train_loader)
                    avg_t1_loss = total_t1_loss / len(train_loader)
                    avg_t2_loss = total_t2_loss / len(train_loader)
                    total_loss = 0
                    total_t1_loss = 0
                    total_t2_loss = 0
                    self.logger.log("epochs", epochs, "loss", avg_loss)
                    self.logger.log("epochs", epochs, "t1loss", avg_t1_loss)
                    self.logger.log("epochs", epochs, "t2loss", avg_t2_loss)

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
                    self.logger.log("epochs", epochs, "mean_iou", mean_iou)

                    # Graph
                    self.logger.graph()

                    # Checkpoint
                    self._save_checkpoint(epochs, model)

            # Dual Updates
            with torch.no_grad():
                mu_cache = 0
                lamb_cache = torch.zeros_like(lamb)
                gamma_cache = torch.zeros_like(gamma)
                for X, Y in tqdm(train_loader):
                    # Forward computation
                    X, Y = X.to(self.device), Y.to(self.device)
                    _, y1_ = model(X)
                    y1 = Y[:, 1]
                    i = Y[:, 2]

                    # Cache for mu update
                    mu_cache += tau[i].sum()

                    # Lambda and gamma updates
                    y1 = y1.float()
                    y1_ = y1_.view(-1)

                    lamb_cache[i] += (
                        self.config["eta_lamb"] * (y1 * (tau[i] - (w * y1_))))
                    gamma_cache[i] += (
                        self.config["eta_gamma"] * (y1 * (tau[i] - eps)))

                # Update data
                mu.data += self.config["eta_mu"] * (mu_cache - 1)
                lamb.data += lamb_cache
                gamma.data += gamma_cache
