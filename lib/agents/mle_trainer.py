from lib.agents.agent import Agent  # noqa E901
from sklearn.datasets import make_multilabel_classification
from tqdm import tqdm
import importlib
import inflection
import torch
import torch.nn.functional as F
from lib.utils.logger import Logger
from pathlib import Path


class MLETrainer(Agent):
    def run(self):
        # Use CPU for simplicity
        self.device = torch.device("cpu")

        #################
        # Prepare dataset
        N_CLASSES = 20
        MEAN_N_LABELS = 14
        N_TRAIN = 100
        N_TEST = 100
        X, Y = make_multilabel_classification(
            n_samples=N_TRAIN + N_TEST, n_classes=N_CLASSES, n_labels=MEAN_N_LABELS
        )
        # X_train.shape = (100, 20), X_test.shape = (100, 20)
        X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
        # Y_train.shape = (100, 20), Y_test.shape = (100, 20)
        Y_train, Y_test = Y[:N_TRAIN], Y[N_TRAIN:]

        # We train one model per class
        # models = {}

        for c in range(N_CLASSES):
            self.logger = Logger(Path(self.config["stats folder"], "class_" + str(c)))

            ############################
            # One vs Rest Transformation
            # (100, 20) -> (1, 2000)
            X_train_c, X_test_c = X_train.reshape(-1, 2000), X_test.reshape(-1, 2000)
            Y_train_c_i = Y_train[c]  # shape = (1, 1)
            Y_train_c = torch.zeros((2, 1))  # [0, 0]
            Y_train_c[Y_train_c_i] = 1  # either [1, 0] or [0, 1]
            Y_test_c_i = Y_test[c]  # shape = (1, 1)
            Y_test_c = torch.zeros((2, 1))  # [0, 0]
            Y_test_c[Y_test_c_i] = 1  # either [1, 0] or [0, 1]

            # Model and optimizer
            model_module = importlib.import_module(
                ("lib.models.{}").format(inflection.underscore(self.config["model"]))
            )
            Model = getattr(model_module, self.config["model"])
            model = Model().to(self.device).double()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.config["learning rate"]
            )

            # Load checkpoint if exists
            start_epochs = self._load_checkpoint(model)

            # Dataset
            for epoch in tqdm(range(start_epochs, self.config["epochs"])):
                total_loss = 0
                model.train()
                for X, Y in tqdm(zip(X_train_c, Y_train_c)):

                    # Forward computation
                    print(Y)
                    Y = (Y == 1).nonzero()
                    X, Y = X.to(self.device), Y.to(self.device)
                    Y_ = model(X)
                    print(Y_)
                    print(Y)
                    loss = F.cross_entropy(Y_, Y)
                    total_loss += loss.item()

                    # Backpropagate
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Log loss
                avg_loss = total_loss / len(X_train)
                self.logger.log("epoch", epoch, "loss", avg_loss)

                # Validate
                model.eval()
                total = 0
                correct = 0
                with torch.no_grad():
                    for X, Y in zip(X_test_c, Y_test_c):
                        X, Y = X.to(self.device), Y.to(self.device)
                        Y_ = model(X)
                        _, predicted = torch.max(Y_.data, 1)
                        total += Y.size(0)
                        correct += (predicted == Y).sum().item()
                accuracy = 100.0 * correct / total
                self.logger.log("epoch", epoch, "accuracy", accuracy)

                # Graph
                self.logger.graph()

                # Checkpoint
                # self._save_checkpoint(epoch, model)
