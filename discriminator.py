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
    def __init__(self, sigmoid_applied=True) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Flatten()
        )
        self.sigmoid_applied = sigmoid_applied
        if self.sigmoid_applied:
            self.sigmoid = torch.nn.Sigmoid()
        self.model.apply(self.init_weights)
        
    def forward(self, X):
        if self.sigmoid_applied:
            return self.sigmoid(self.model(X))
        return self.model(X)
    
    def init_weights(self, layer: torch.nn.Module):
        className = layer.__class__.__name__
        if className.find("Conv") != -1:
            torch.nn.init.normal_(layer.weight.data, 0, 0.02)
        elif className.find("Batch") != -1:
            torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(layer.bias.data, 0)
    
    
if __name__ == "__main__":
    model = DCGANDiscriminator()
    X = torch.randn(12, 1, 64, 64)
    print(model(X).shape)
