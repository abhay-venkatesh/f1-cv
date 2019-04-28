from lib.agents.trainer import Trainer
from lib.datasets.mnistf1 import MNISTF1
from lib.models.basicnet import BasicNetF1
from lib.utils.functional import lagrange
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

        # Model
        model = BasicNetF1().to(self.device)

        # Constants
        num_positives = self.train_loader.dataset.num_positives

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
        gamma = torch.zeros(1, device=self.device)
        gamma.fill_(0.001)

        # Temporary variables for dual updates
        tau_1 = 0.0
        tau_eps = 0.0
        tau_w_y = torch.zeros(len(train_loader.dataset)).to(self.device)

        # Primal Optimization
        var_list = [{
            "params": model.parameters(),
            "lr": self.config["learning rate"]
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
        optimizer = torch.optim.SGD(var_list)

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
                Y_, y_ = self.model(X)
                y = Y[:, 1]
                i = Y[:, 2]

                # Loss business
                lagrangian = lagrange(num_positives, y_, y, w, eps, tau[i],
                                      lamb[i], mu, gamma, self.device)
                loss = F.cross_entropy(Y_,
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
