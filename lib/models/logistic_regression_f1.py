import torch.nn as nn


class LogisticRegressionF1(nn.Module):
    def __init__(self, input_size=20, num_classes=2):
        super(LogisticRegressionF1, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc1(x)
        return out1, out2
