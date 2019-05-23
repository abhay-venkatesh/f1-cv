import yaml
from lib.logger import Logger
from lib.mnistf1 import MNISTF1
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import inflection
import os
import torch
import torch.nn.functional as F


def load_config():
    with open("config.yml", 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print(exc)


def lagrange(num_pos, y1_, y1, w, eps, tau, lamb, mu, gamma):
    # Reshape
    y1 = y1.float()
    y1_ = y1_.squeeze()

    # Term that is not associated with either positive or negative examples
    neutral = (num_pos * eps)

    # Negative example terms
    neg = torch.max(torch.zeros_like(y1_), eps + (w * y1_))
    neg = (abs(1 - y1) * neg).sum()

    # Positive example terms
    pos = mu * ((y1 * tau).sum() - 1)
    pos += (y1 * lamb * (tau - (w * y1_))).sum()
    pos += (y1 * gamma * (tau - eps)).sum()

    return neutral + neg + pos


class MNISTF1Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.logger = Logger(config["stats folder"])

    def _load_checkpoint(self, model):
        start_epochs = 0
        if self.config["checkpoint path"]:
            start_epochs = int(Path(self.config["checkpoint path"]).stem)
            model.load_state_dict(
                torch.load(Path(self.config["checkpoint path"])))
        return start_epochs

    def _save_checkpoint(self, epoch, model, retain=False):
        checkpoint_filename = str(epoch + 1) + ".ckpt"
        checkpoint_path = Path(self.config["checkpoints folder"],
                               checkpoint_filename)

        torch.save(model.state_dict(), checkpoint_path)

        if not retain:
            prev_checkpoint_filename = str(epoch) + ".ckpt"
            prev_checkpoint_path = Path(self.config["checkpoints folder"],
                                        prev_checkpoint_filename)
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)

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
        model_module = importlib.import_module(("lib.models.{}").format(
            inflection.underscore(self.config["model"])))
        Model = getattr(model_module, self.config["model"])
        model = Model().to(self.device)

        # Load checkpoint if exists
        start_epochs = self._load_checkpoint(model)

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

        # Count epochs and steps
        epochs = 0
        step = 0

        # Cache losses
        total_loss = 0
        total_t1_loss = 0
        total_t2_loss = 0

        # Train
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
                t1_loss = F.cross_entropy(y0_, y0)
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
                    total = 0
                    correct = 0
                    with torch.no_grad():
                        for X, Y in val_loader:
                            X, Y = X.to(self.device), Y.to(self.device)
                            y0_, y1_ = model(X)
                            y0 = Y[:, 0]
                            y1 = Y[:, 1]
                            _, predicted = torch.max(y0_.data, 1)
                            total += y0.size(0)
                            correct += (predicted == y0).sum().item()
                    accuracy = 100. * correct / total
                    self.logger.log("epochs", epochs, "accuracy", accuracy)

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


if __name__ == "__main__":
    config = load_config()
    trainer = MNISTF1Trainer(config)
