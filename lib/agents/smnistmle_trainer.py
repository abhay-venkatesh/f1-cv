from lib.agents.agent import Agent
from lib.utils.logger import Logger
from mle import optimal_basket
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from lib.models.mle_net import MLENet
from lib.datasets.mnistf1 import MNISTF1
from torch.utils.data import DataLoader


class SMNISTMLETrainer(Agent):
    def run(self):
        N_CLASSES = 10

        # Training set
        trainset = MNISTF1(self.config["dataset path"], train=True, download=True)
        train_loader = DataLoader(
            trainset, shuffle=True, batch_size=self.config["batch size"]
        )

        # Validation set
        valset = MNISTF1(self.config["dataset path"], train=False, download=True)
        val_loader = DataLoader(valset, batch_size=self.config["batch size"])

        # We train one model per class
        models = {}

        for c in tqdm(range(N_CLASSES)):
            self.logger = Logger(Path(self.config["stats folder"], "class_" + str(c)))

            # Model and optimizer
            model = MLENet().to(self.device)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.config["learning rate"]
            )

            # Load checkpoint if exists
            start_epochs = self._load_checkpoint(model)

            # Dataset
            for epoch in tqdm(range(start_epochs, self.config["epochs"])):
                total_loss = 0
                model.train()
                for X, Y in tqdm(train_loader):

                    # One vs. Rest Conversion
                    Y = Y[:, 0]
                    Y[Y == c] = 1
                    Y[Y != c] = 0

                    # Forward computation
                    X, Y = X.to(self.device), Y.to(self.device)
                    Y_ = model(X).reshape((1, -1))
                    loss = F.cross_entropy(Y_, Y)
                    total_loss += loss.item()

                    # Backpropagate
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    break

            models[c] = model
            break

        preds = np.zeros((len(val_loader.dataset), N_CLASSES))
        probs = np.zeros((len(val_loader.dataset), N_CLASSES))
        for c in range(N_CLASSES):
            model = models[c]
            model.eval()

            with torch.no_grad():
                # pred_layer = np.zeros(preds[:, c].shape)
                # prob_layer = np.zeros(probs[:, c].shape)
                for i, (X, Y) in enumerate(val_loader):
                    X, Y = X.to(self.device), Y.to(self.device)
                    Y_ = model(X).reshape((1, -1))
                    _, predicted = torch.max(Y_.data, 1)
                    preds[:, c][i] = predicted
                    probs[:, c][i] = torch.nn.Softmax(dim=1)(Y_)[0][1]
            break

        # Simple prediction
        scores = list()
        for x, y, _ in zip(preds, val_loader):
            scores.append(f1_score(x, y))
        score_simple = np.mean(scores)
        print("Simple score:", score_simple)

        # Optimized prediction
        scores = list()
        for x, y, _ in zip(probs, val_loader):
            pred = np.zeros(N_CLASSES)
            pred_idxs = optimal_basket(x)
            pred[pred_idxs] = 1
            scores.append(f1_score(pred, y))
        score_optimized = np.mean(scores)
        print("Optimized score:", score_optimized)
