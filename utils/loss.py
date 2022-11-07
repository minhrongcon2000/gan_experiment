import torch
from typing_extensions import Literal


class WGANLoss(torch.nn.Module):
    def forward(self, fake_pred, real_pred=None):
        if real_pred is None:
            return -torch.mean(fake_pred)
        return torch.mean(real_pred - fake_pred)
