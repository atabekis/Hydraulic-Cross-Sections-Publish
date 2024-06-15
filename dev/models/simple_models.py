import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 10, 128)  # Updated for 40x40 input size
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




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

# Rotational Symmetry Models
class Rot90CNN(BaseCNN):
    def forward(self, x):
        x = torch.rot90(x, 1, [2, 3])
        return super().forward(x)

class Rot180CNN(BaseCNN):
    def forward(self, x):
        x = torch.rot90(x, 2, [2, 3])
        return super().forward(x)

class Rot270CNN(BaseCNN):
    def forward(self, x):
        x = torch.rot90(x, 3, [2, 3])
        return super().forward(x)

# Flip Symmetry Models
class FlipXCNN(BaseCNN):
    def forward(self, x):
        x = torch.flip(x, [2])
        return super().forward(x)

class FlipYCNN(BaseCNN):
    def forward(self, x):
        x = torch.flip(x, [3])
        return super().forward(x)

# Combined Network
class CombinedCNN(nn.Module):
    def __init__(self):
        super(CombinedCNN, self).__init__()
        self.base_cnn = BaseCNN()
        self.rot90_cnn = Rot90CNN()
        self.rot180_cnn = Rot180CNN()
        self.rot270_cnn = Rot270CNN()
        self.flipx_cnn = FlipXCNN()
        self.flipy_cnn = FlipYCNN()
        self.fc = nn.Linear(6, 1, bias=False)

    def forward(self, x):
        base_out = self.base_cnn(x)
        rot90_out = self.rot90_cnn(x)
        rot180_out = self.rot180_cnn(x)
        rot270_out = self.rot270_cnn(x)
        flipx_out = self.flipx_cnn(x)
        flipy_out = self.flipy_cnn(x)
        combined_out = torch.cat([base_out, rot90_out, rot180_out, rot270_out, flipx_out, flipy_out], dim=1)
        out = self.fc(combined_out)
        return out

# Instantiate the combined model
# model = CombinedCNN()
#
# # Print model summary
# print(model)
