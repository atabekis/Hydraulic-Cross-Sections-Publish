import torch
from torch import nn


class ApplySymmetry(nn.Module):
    def __init__(self):
        super(ApplySymmetry, self).__init__()

    def forward(self, x):
        rotations = [torch.rot90(x, k, [2, 3]) for k in range(4)]
        flips = [torch.flip(x, [2]), torch.flip(x, [3])]
        flip_rotations = [torch.flip(torch.rot90(x, k, [2, 3]), [2]) for k in [1, 3]]
        symmetries = rotations + flips + flip_rotations
        return torch.stack(symmetries, dim=1)


class ConvNet(nn.Module):
    def __init__(self, pad=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=pad, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=pad, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=pad, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, sym_count, channels, height, width = x.size()
        x = x.view(batch_size * sym_count, channels, height, width)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.gap(x)
        x = x.view(batch_size, sym_count, -1)
        return x

class GroupAverage(nn.Module):
    def __init__(self):
        super(GroupAverage, self).__init__()

    def forward(self, x):
        return x.mean(dim=1)


def make_model(pad=1):
    return nn.Sequential(
        ApplySymmetry(),
        ConvNet(pad=pad),
        GroupAverage(),
        nn.Linear(64, 32, bias=False),
        nn.ReLU(),
        nn.Linear(32, 1, bias=False)
    )

