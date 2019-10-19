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
from lib.utils.functional import lagrange
from lib.models.logistic_regression import LogisticRegression


class MLEF1Trainer(Agent):
    def run_baseline(self, X, Y):
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
            model = LogisticRegression().to(self.device).double()
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
        pass

    def run(self):
        #################
        # Prepare dataset
        N_CLASSES = 20
        MEAN_N_LABELS = 14
        N_TRAIN = 100
        N_TEST = 100
        X, Y = make_multilabel_classification(
            n_samples=N_TRAIN + N_TEST, n_classes=N_CLASSES, n_labels=MEAN_N_LABELS
        )
        self.run_baseline(X, Y)
        self.run_f1(X, Y)

    def run_f1(self, X, Y):
        N_CLASSES = 20
        N_TRAIN = 100

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

            # Model
            model_module = importlib.import_module(
                ("lib.models.{}").format(inflection.underscore(self.config["model"]))
            )
            Model = getattr(model_module, self.config["model"])
            model = Model().double()

            # Load checkpoint if exists
            start_epochs = self._load_checkpoint(model)

            # Constants
            num_positives = Y_train_c.sum()

            # Primal variables
            tau = torch.rand(len(X_train), requires_grad=True)
            eps = torch.rand(1, requires_grad=True)
            w = torch.rand(1, requires_grad=True)

            # Dual variables
            lamb = torch.zeros(len(X_train))
            lamb.fill_(0.001)
            mu = torch.zeros(1)
            mu.fill_(0.001)
            gamma = torch.zeros(len(X_train))
            gamma.fill_(0.001)

            # Primal Optimization
            var_list = [
                {"params": model.parameters(), "lr": self.config["learning rate"]},
                {"params": tau, "lr": self.config["eta_tau"]},
                {"params": eps, "lr": self.config["eta_eps"]},
                {"params": w, "lr": self.config["eta_w"]},
            ]
            optimizer = torch.optim.SGD(var_list)

            # Dataset iterator
            train_iter = iter(zip(X_train, Y_train_c))

            # Count epochs and steps
            epochs = 0
            step = 0
            i = 0

            # Cache losses
            total_loss = 0
            total_t1_loss = 0
            total_t2_loss = 0

            # Train
            for outer in tqdm(range(start_epochs, self.config["n_outer"])):
                model.train()

                for inner in range(self.config["n_inner"]):
                    step += 1

                    # Sample
                    try:
                        X, Y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(zip(X_train, Y_train_c))
                        X, Y = next(train_iter)

                    # Forward computation
                    X, Y = torch.from_numpy(X), torch.tensor(Y).reshape((1,))
                    y0_, y1_ = model(X)
                    y1_ = y1_
                    y0_ = y0_.reshape((1, -1))
                    y0 = Y
                    y1 = Y

                    # Compute loss
                    t1_loss = F.cross_entropy(y0_, y0)
                    t2_loss = lagrange(
                        num_positives, y1_, y1, w, eps, tau[i], lamb[i], mu, gamma[i]
                    )
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
                    eps.data = torch.max(torch.zeros(1, dtype=torch.float), eps.data)

                    # Log and validate per epoch
                    i += 1
                    if (step + 1) % len(X_train) == 0:
                        epochs += 1

                        # Log loss
                        avg_loss = total_loss / len(X_train)
                        avg_t1_loss = total_t1_loss / len(X_train)
                        avg_t2_loss = total_t2_loss / len(X_train)
                        total_loss = 0
                        total_t1_loss = 0
                        total_t2_loss = 0
                        self.logger.log("epochs", epochs, "loss", avg_loss)
                        self.logger.log("epochs", epochs, "t1loss", avg_t1_loss)
                        self.logger.log("epochs", epochs, "t2loss", avg_t2_loss)

                        i = 0

                # Dual Updates
                with torch.no_grad():
                    mu_cache = 0
                    lamb_cache = torch.zeros_like(lamb).reshape((len(X_train), 1))
                    gamma_cache = torch.zeros_like(gamma).reshape((len(X_train), 1))
                    for j, (X, Y) in enumerate(zip(X_train, Y_train_c)):
                        # Forward computation
                        X, Y = torch.from_numpy(X), torch.tensor(Y).reshape((1,))
                        _, y1_ = model(X)
                        y1 = Y.reshape((1, -1))

                        # Cache for mu update
                        mu_cache += tau[j].sum()

                        # Lambda and gamma updates
                        y1 = y1.float()
                        y1_ = y1_.view(-1)

                        lamb_cache[j] += (
                            self.config["eta_lamb"] * (y1 * (tau[j] - (w * y1_)))
                        ).reshape((1))
                        gamma_cache[j] += self.config["eta_gamma"] * (
                            y1 * (tau[j] - eps)
                        ).reshape((1))

                    # Update data
                    mu.data += self.config["eta_mu"] * (mu_cache - 1)
                    lamb.data += lamb_cache.reshape((len(X_train)))
                    gamma.data += gamma_cache.reshape((len(X_train)))

            # Evaluate model
            model.eval()

            # One vs Rest Transformation
            # (100, 20) -> (100,)
            Y_test_c = Y_test[:, c]

            with torch.no_grad():
                pred_layer = np.zeros(Y_test_c.shape)
                for i, (X, Y) in enumerate(zip(X_test, Y_test_c)):
                    X, Y = torch.from_numpy(X), torch.tensor(Y).reshape((1,))
                    Y_, _ = model(X)
                    Y_ = Y_.reshape((1, -1))
                    _, predicted = torch.max(Y_.data, 1)
                    pred_layer[i] = predicted
                # print(f1_score(pred_layer, Y_test_c))

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
                    Y_, _ = model(X)
                    Y_ = Y_.reshape((1, -1))
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
