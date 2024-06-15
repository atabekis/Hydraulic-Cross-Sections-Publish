import torch
from torch import nn
import torch.nn.functional as F


class NoBiasDamnit(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(NoBiasDamnit, self).__init__(*args, **kwargs)
        self.bias = None


class GCNN(nn.Module):
    def __init__(self):
        super(GCNN, self).__init__()
        self.conv1 = NoBiasDamnit(1, 16, kernel_size=3, padding=1)
        self.conv2 = NoBiasDamnit(16, 32, kernel_size=3, padding=1)
        self.conv3 = NoBiasDamnit(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 40 * 40, 128, bias=False)
        self.fc2 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        symmetries = self.generate_symmetries(x)

        outputs = []
        for sym in symmetries:
            print(f'sym {sym.shape}')
            out = F.relu(self.conv1(sym.view(-1, 1, 40, 40)))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)
        print(f'{outputs.shape=}')

        averaged_output = torch.mean(outputs, dim=0)

        averaged_output = averaged_output.view(averaged_output.size(0), -1)
        print(f'{averaged_output.shape}=')

        x = F.relu(self.fc1(averaged_output))
        x = self.fc2(x)
        print(f'{x.shape=}')
        return x

    def generate_symmetries(self, x):
        rotations = [torch.rot90(x, k, [2, 3]) for k in range(4)]  # id, r90, r180, r270
        flips = [torch.flip(x, [3]), torch.flip(x, [2])]
        flip_rotations = [torch.flip(torch.rot90(x, k, [2, 3]), [2]) for k in [1, 3]]
        return rotations + flips + flip_rotations


# ----------------------------------------------

class BaseCNN(nn.Module):
    def __init__(self, bias=False):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        return x


class GCNN2(nn.Module):
    def __init__(self, biases=False):
        super(GCNN2, self).__init__()
        self.base_cnn = BaseCNN(bias=biases)
        self.fc1 = nn.Linear(64 * 5 * 5, 100, bias=biases)
        self.fc2 = nn.Linear(100, 1, bias=biases)

    def forward(self, x):
        symmetries = self.generate_symmetries(x)

        outputs = []
        for sym in symmetries:
            base_out = self.base_cnn(sym)
            outputs.append(base_out)

        outputs = torch.stack(outputs, dim=0)
        averaged_output = torch.mean(outputs, dim=0)

        averaged_output = averaged_output.view(averaged_output.size(0), -1)
        x = F.relu(self.fc1(averaged_output))
        x = self.fc2(x)

        return x

    def generate_symmetries(self, x):
        rotations = [torch.rot90(x, k, [2, 3]) for k in range(4)]  # id, r90, r180, r270
        flips = [torch.flip(x, [3]), torch.flip(x, [2])]
        flip_rotations = [torch.flip(torch.rot90(x, k, [2, 3]), [2]) for k in [1, 3]]
        return rotations + flips + flip_rotations


class GConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(GConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)

        # Add the specified symmetries: rot90, rot180, rot270, flipUD, flipLR, flipUD+rot90, flipUD+rot270
        x_rot90 = torch.rot90(x, 1, [2, 3])
        x_rot180 = torch.rot90(x, 2, [2, 3])
        x_rot270 = torch.rot90(x, 3, [2, 3])
        x_flipUD = torch.flip(x, [2])
        x_flipLR = torch.flip(x, [3])
        x_flipUD_rot90 = torch.rot90(x_flipUD, 1, [2, 3])
        x_flipUD_rot270 = torch.rot90(x_flipUD, 3, [2, 3])

        # Concatenate along the channel dimension
        x = torch.cat([x, x_rot90, x_rot180, x_rot270, x_flipUD, x_flipLR, x_flipUD_rot90, x_flipUD_rot270], dim=1)

        return x


class GEquivariantNetwork(nn.Module):
    def __init__(self):
        super(GEquivariantNetwork, self).__init__()
        self.gconv1 = GConv2D(1, 8, kernel_size=3, padding=1)
        self.gconv2 = GConv2D(64, 16, kernel_size=3, padding=1)
        self.gconv3 = GConv2D(128, 32, kernel_size=3, padding=1)
        self.gconv4 = GConv2D(256, 64, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        x = F.relu(self.gconv3(x))
        x = F.relu(self.gconv4(x))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x