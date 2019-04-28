import torch.nn as nn


class BasicNet(nn.Module):
    # Reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
    def __init__(self, num_classes=10):
        super(BasicNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(
                kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(
                kernel_size=2, stride=2))
        self.fc = nn.Linear(16 * 16 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class BasicNetF1(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicNetF1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(
                kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(
                kernel_size=2, stride=2))
        self.fc1 = nn.Linear(16 * 16 * 32, num_classes)
        self.fc2 = nn.Linear(16 * 16 * 32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        return out1, out2
