import torch


class MNISTGenerator(torch.nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 784
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 1200),
            torch.nn.ReLU(),
            torch.nn.Linear(1200, 1200),
            torch.nn.ReLU(),
            torch.nn.Linear(1200, 784),
            torch.nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.model(X).view(-1, 1, 28, 28)


class DCGANGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 100
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )
        self.model.apply(self.init_weights)
        
    def forward(self, X) -> None:
        return self.model(X.reshape(-1, 100, 1, 1))
    
    def init_weights(self, layer: torch.nn.Module):
        className = layer.__class__.__name__
        if className.find("Conv") != -1:
            torch.nn.init.normal_(layer.weight.data, 0, 0.02)
            if layer.bias is not None:
                torch.nn.init.normal_(layer.bias.data, 0, 0.02)
        elif className.find("Batch") != -1:
            torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(layer.bias.data, 0)
    
if __name__ == "__main__":
    model = DCGANGenerator()
    X = torch.randn(16, 100)
    print(model(X).shape)