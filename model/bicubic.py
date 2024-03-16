import torch.nn.functional as F
import torch.nn as nn

class BICUBIC(nn.Module):
    def __init__(self, scale_factor):
        super(BICUBIC, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
