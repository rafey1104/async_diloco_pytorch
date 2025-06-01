import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)    # (3, 32, 32) -> (6, 28, 28)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # (6, 28, 28) -> (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # (6, 14, 14) -> (16, 10, 10)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # (16, 10, 10) -> (16, 5, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 10 output classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 16 * 5 * 5)             # Flatten
        x = F.relu(self.fc1(x))               # FC1 + ReLU
        x = F.relu(self.fc2(x))               # FC2 + ReLU
        x = self.fc3(x)                       # FC3 (logits)
        return x
