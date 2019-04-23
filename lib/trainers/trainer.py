from lib.utils.simple_ml_logger.logger import Logger
import torch


class Trainer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.logger = Logger(self.experiment.stats_folder)

    def train(self):
        raise NotImplementedError