from lib.agents.agent import Agent  # noqa E901
from lib.utils.logger import Logger
from mle import optimal_basket
from pathlib import Path
from sklearn.datasets import make_multilabel_classification
from tqdm import tqdm
import importlib
import inflection
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


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
        models = {}

        for c in tqdm(range(N_CLASSES)):
            self.logger = Logger(Path(self.config["stats folder"], "class_" + str(c)))

            # One vs Rest Transformation
            # (100, 20) -> (100,)
            Y_train_c = Y_train[:, c]

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
            for epoch in range(start_epochs, self.config["epochs"]):
                total_loss = 0
                model.train()
                for X, Y in zip(X_train, Y_train_c):

                    # Forward computation
                    X, Y = torch.from_numpy(X), torch.tensor(Y).reshape((1,))
                    # X, Y = X.to(self.device), Y.to(self.device)
                    Y_ = model(X).reshape((1, -1))
                    loss = F.cross_entropy(Y_, Y)
                    total_loss += loss.item()

                    # Backpropagate
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            models[c] = model

        preds = np.zeros(Y_test.shape)
        probs = np.zeros(Y_test.shape)
        for c in range(N_CLASSES):
            model = models[c]
            model.eval()

            # One vs Rest Transformation
            # (100, 20) -> (100,)
            Y_test_c = Y_test[:, c]

            with torch.no_grad():
                pred_layer = np.zeros(Y_test_c.shape)
                prob_layer = np.zeros(Y_test_c.shape)
                for i, (X, Y) in enumerate(zip(X_test, Y_test_c)):
                    X, Y = torch.from_numpy(X), torch.tensor(Y).reshape((1,))
                    Y_ = model(X).reshape((1, -1))
                    _, predicted = torch.max(Y_.data, 1)
                    pred_layer[i] = predicted
                    prob_layer[i] = torch.nn.Softmax(dim=1)(Y_)[0][1]

            preds[:, c] = pred_layer
            probs[:, c] = prob_layer

        # Simple prediction
        scores = list()
        for x, y in zip(preds, Y_test):
            scores.append(f1_score(x, y))
        score_simple = np.mean(scores)
        print("Simple score:", score_simple)

        # Optimized prediction
        scores = list()
        for x, y in zip(probs, Y_test):
            pred = np.zeros(N_CLASSES)
            pred_idxs = optimal_basket(x)
            pred[pred_idxs] = 1
            scores.append(f1_score(pred, y))
        score_optimized = np.mean(scores)
        print("Optimized score:", score_optimized)

        """
        # Log loss
        avg_loss = total_loss / len(X_train)
        self.logger.log("epoch", epoch, "loss", avg_loss)

        # Validate
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for X, Y in zip(X_test, Y_test_c):
                # X, Y = X.to(self.device), Y.to(self.device)
                X, Y = torch.from_numpy(X), torch.tensor(Y).reshape((1,))
                Y_ = model(X).reshape((1, -1))
                _, predicted = torch.max(Y_.data, 1)
                total += Y.size(0)
                correct += (predicted == Y).sum().item()
        accuracy = 100.0 * correct / total
        self.logger.log("epoch", epoch, "accuracy", accuracy)

        # Graph
        self.logger.graph()

        # Checkpoint
        # self._save_checkpoint(epoch, model)
        """
