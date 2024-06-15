import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x


class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(128 * 5 * 5, 256, bias=False)
        self.fc2 = nn.Linear(256, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DeepSet(nn.Module):
    def __init__(self):
        super(DeepSet, self).__init__()
        self.embedding = EmbeddingNetwork()
        self.prediction = PredictionNetwork()

    def forward(self, x):
        transformations = self.apply_symmetries(x)

        embedded = [self.embedding(t) for t in transformations]

        aggregated = torch.stack(embedded).sum(dim=0)

        out = self.prediction(aggregated)
        return out

    def apply_symmetries(self, x):
        rotations = [torch.rot90(x, k, [2, 3]) for k in range(4)]  # id, r90, r180, r270
        flips = [torch.flip(x, [2]), torch.flip(x, [3])]
        flip_rotations = [torch.flip(torch.rot90(x, k, [2, 3]), [2]) for k in [1, 3]]
        return rotations + flips + flip_rotations


# ---------------------------------------------------
