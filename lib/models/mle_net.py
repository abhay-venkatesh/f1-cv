import torch.nn as nn


class MLENet(nn.Module):
    def __init__(self, input_size=4096, hidden_size=500, num_classes=2):
        super(MLENet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 64 * 64)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
