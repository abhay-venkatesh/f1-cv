from lib.agents.trainer import Trainer
from lib.datasets.mnistbs import MNISTBS
from lib.models.basicnet import BasicNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F


class MNISTBSTrainer(Trainer):
    def run(self):
        # Training set
        trainset = MNISTBS(
            self.config["dataset path"], train=True, download=True)
        train_loader = DataLoader(
            trainset, shuffle=True, batch_size=self.config["batch size"])

        # Validation set
        valset = MNISTBS(
            self.config["dataset path"], train=False, download=True)
        val_loader = DataLoader(valset, batch_size=self.config["batch size"])

        # Model and optimizer
        model = BasicNet().to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.config["learning rate"])

        # Dataset
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
