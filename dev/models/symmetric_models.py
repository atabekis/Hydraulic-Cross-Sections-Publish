import torch
import torch.nn as nn
import torch.nn.functional as F


class SymNet(nn.Module):
    def __init__(self):
        super(SymNet, self).__init__()

    def forward(self, x):
        rotations = [torch.rot90(x, k, dims=[2, 3]) for k in range(4)]
        flips = [torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]  # fliplr, flipud
        flip_rotations = [torch.rot90(torch.flip(x, dims=[2]), k, dims=[2, 3]) for k in
                          [1, 3]]  # flipud + r90, flipud + r270
        return rotations + flips + flip_rotations


class CombineNet(nn.Module):
    def __init__(self, combine_type='mean'):
        super(CombineNet, self).__init__()
        self.combine_type = combine_type

    def forward(self, x):
        x = torch.stack(x, dim=0)
        if self.combine_type == 'mean':
            combined = torch.mean(x, dim=0)
        elif self.combine_type == 'sum':
            combined = torch.sum(x, dim=0)
        elif self.combine_type == 'max':
            combined, _ = torch.max(x, dim=0)
        elif self.combine_type == 'min':
            combined, _ = torch.min(x, dim=0)
        return combined


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(16 * 40 * 40, 1, bias=False),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class FinalNetwork(nn.Module):  # 0.05 over 100 epochs 128 bs
    def __init__(self, combine_type='mean'):
        super(FinalNetwork, self).__init__()
        self.symmetry_network = SymNet()
        self.cnn = ConvNet()
        self.combiner_network = CombineNet(combine_type=combine_type)

    def forward(self, x):
        symmetrical_inputs = self.symmetry_network(x)
        cnn_outputs = [self.cnn(sym_input) for sym_input in symmetrical_inputs]
        final_output = self.combiner_network(cnn_outputs)
        return final_output