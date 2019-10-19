import torch.nn as nn


class MLENetF1(nn.Module):
    def __init__(self, input_size=4096, hidden_size=500, num_classes=2):
        super(MLENetF1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x = x.reshape(-1, 64 * 64)
        out = self.fc1(x)
        out = self.relu(out)
        out1 = self.fc2(out)
        out2 = self.fc3(out)
        return out1, out2