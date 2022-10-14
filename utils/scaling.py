import torch

class InputScaling(torch.nn.Module):
    def __init__(self, scaling: float) -> None:
        super().__init__()
        self.scaling = scaling
        
    def forward(self, X):
        return self.scaling * X