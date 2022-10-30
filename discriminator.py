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


class DCGANDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, 5, 2, 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 5, 2, 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, 5, 2, 2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1024, 5, 2, 2),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 1024, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.model(X)
    
    
if __name__ == "__main__":
    model = DCGANDiscriminator()
    X = torch.randn(1, 3, 64, 64)
    print(model(X))
