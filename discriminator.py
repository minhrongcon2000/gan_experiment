import torch

from utils.maxout import Maxout
from utils.scaling import InputScaling


class MNISTDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 784
        self.output_dim = 1
        
        # Architecture refers to https://github.com/goodfeli/adversarial/blob/master/mnist.yaml 
        # and https://github.com/goodfeli/pylearn2
        self.h0 = torch.nn.Sequential(
            Maxout(self.input_dim, 5, 240), 
            torch.nn.Dropout(0.8), 
        )
        
        self.h1 = torch.nn.Sequential(
            Maxout(240, 5, 240),
            torch.nn.Dropout(0.5), 
        )
        
        self.y = torch.nn.Sequential(
            torch.nn.Linear(240, self.output_dim),
            torch.nn.Dropout(0.5),
            torch.nn.Sigmoid()
        )
        
    def forward(self, X):
        h0 = self.h0(X.view(-1, 784))
        h1 = self.h1(h0)
        y = self.y(h1)
        return y