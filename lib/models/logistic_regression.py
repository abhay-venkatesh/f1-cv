import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_size=20, num_classes=2):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out
