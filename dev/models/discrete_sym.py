from torch import nn
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
        print(f'x_in: {x.shape}')
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        print(f"x_out: {x.shape} ")
        return x



# Create a model instance and print its summary


# generate sym (augment) → apply each to CNN - linear asw → group average (for private)
# so group average is not in the gradient for the backprop!!!!!!