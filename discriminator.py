import torch

from utils.maxout import Maxout


class MNISTDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 784
        self.output_dim = 1
        self.model = torch.nn.Sequential(
            Maxout(self.input_dim, 5, 240),
            Maxout(240, 5, 240),
            torch.nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.model(X.view(-1, 784))