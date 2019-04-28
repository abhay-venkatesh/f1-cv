from lib.agents.trainer import Trainer
from lib.datasets.mnistf1 import MNISTF1
from lib.models.basicnet import BasicNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F


class MNISTF1Trainer(Trainer):
    def run(self):
        # Training set
        trainset = MNISTF1(
            self.config["dataset path"], train=True, download=True)
        train_loader = DataLoader(
            trainset, shuffle=True, batch_size=self.config["batch size"])

        # Validation set
        valset = MNISTF1(
            self.config["dataset path"], train=False, download=True)
        val_loader = DataLoader(valset, batch_size=self.config["batch size"])

        # Constants
        num_positives = self.train_loader.dataset.num_positives

        # Primal variables
        tau = torch.rand(
            len(train_loader.dataset),
            device=self.device,
            requires_grad=True)
        eps = torch.rand(1, device=self.device, requires_grad=True)
        w = torch.rand(1, device=self.device, requires_grad=True)

        # Dual variables
        lamb = torch.zeros(len(train_loader.dataset), device=self.device)
        lamb.fill_(0.001)
        mu = torch.zeros(1, device=self.device)
        mu.fill_(0.001)
        gamma = torch.zeros(1, device=self.device)
        gamma.fill_(0.001)

        # Temporary variables for dual updates
        tau_1 = 0.0
        tau_eps = 0.0
        tau_w_y = torch.zeros(len(train_loader.dataset)).to(self.device)

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
        optimizer = optim.SGD(var_list)

        # Dataset iterator
        train_iter = iter(train_loader)

        for outer in tqdm(range(self.config["n_outer"])):
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
                Y_ = model(X)
                loss = F.cross_entropy(Y_, Y)
                total_loss += loss.item()

                # Backpropagate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Log loss
            avg_loss = total_loss / len(train_loader)
            self.logger.log("epoch", outer, "loss", avg_loss)

            # Validate
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(self.device), Y.to(self.device)
                    Y_ = model(X)
                    _, predicted = torch.max(Y_.data, 1)
                    total += Y.size(0)
                    correct += (predicted == Y).sum().item()
            accuracy = 100. * correct / total
            self.logger.log("outer", outer, "accuracy", accuracy)

            self.logger.graph()
