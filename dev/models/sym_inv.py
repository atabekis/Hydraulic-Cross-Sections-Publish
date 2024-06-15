import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 5 * 5, 100, bias=False)
        self.fc2 = nn.Linear(100, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AugmentedCNN(nn.Module):
    def __init__(self):
        super(AugmentedCNN, self).__init__()
        self.base_cnn = BaseCNN()
        self.fc = nn.Linear(6, 1, bias=False)

    def forward(self, x):
        outputs = []
        transformations = [
            lambda x: x,                          # No transformation
            lambda x: torch.rot90(x, 1, [2, 3]),  # 90 degrees rotation
            lambda x: torch.rot90(x, 2, [2, 3]),  # 180 degrees rotation
            lambda x: torch.rot90(x, 3, [2, 3]),  # 270 degrees rotation
            lambda x: torch.flip(x, [2]),         # Flip horizontally
            lambda x: torch.flip(x, [3])          # Flip vertically
        ]

        for transform in transformations:
            transformed_x = transform(x)
            output = self.base_cnn(transformed_x)
            outputs.append(output)

        combined_out = torch.cat(outputs, dim=1)
        out = self.fc(combined_out)
        return out

# Example usage:
# model = AugmentedCNN()
# x = torch.randn(1, 1, 40, 40)  # example input
# output = model(x)
# print(output)
