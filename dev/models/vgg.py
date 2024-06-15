import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(128 * 10 * 10, 512, bias=False)
        self.fc2 = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 10 * 10)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ExpandedVGG(nn.Module):
    def __init__(self):
        super(ExpandedVGG, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 2 * 2, 4096, bias=False)
        self.fc2 = nn.Linear(4096, 4096, bias=False)
        self.fc3 = nn.Linear(4096, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Output size: (32, 64, 20, 20)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # Output size: (32, 128, 10, 10)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)  # Output size: (32, 256, 5, 5)
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool(x)  # Output size: (32, 512, 2, 2)
        x = x.view(-1, 512 * 2 * 2)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
