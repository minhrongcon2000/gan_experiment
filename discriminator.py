import torch

from utils.maxout import Maxout


class MNISTDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 784
        self.output_dim = 1
        self.model = torch.nn.Sequential(
            Maxout(self.input_dim, 5, 240),
            # Dropout is of UTMOST IMPORTANCE!!!!
            torch.nn.Dropout(0.8),
            Maxout(240, 5, 240),
            torch.nn.Linear(240, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.model(X.view(-1, 784))