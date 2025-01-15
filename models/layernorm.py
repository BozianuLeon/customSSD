import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.dim = dim
        self.eps = eps
        self.ln = nn.LayerNorm(dim, eps=self.eps) 

    def forward(self, x):
        # First, permute to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        # Apply LayerNorm over the last dimension (channels)
        x = self.ln(x)
        # Permute back to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        return x


