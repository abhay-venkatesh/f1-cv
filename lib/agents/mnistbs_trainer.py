from lib.agents.agent import Agent
from lib.datasets.mnistbs import MNISTBS
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import inflection
import torch
import torch.nn.functional as F


class MNISTBSTrainer(Agent):
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
        model_module = importlib.import_module(("lib.models.{}").format(
            inflection.underscore(self.config["model"])))
        Model = getattr(model_module, self.config["model"])
        model = Model().to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.config["learning rate"])

        # Load checkpoint if exists
        start_epochs = self._load_checkpoint(model)

        # Dataset
        for epoch in tqdm(range(start_epochs, self.config["epochs"])):
            total_loss = 0
            model.train()
            for X, Y in tqdm(train_loader):

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
            self.logger.log("epoch", epoch, "loss", avg_loss)

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
            self.logger.log("epoch", epoch, "accuracy", accuracy)

            # Graph
            self.logger.graph()

            # Checkpoint
            self._save_checkpoint(epoch, model)
