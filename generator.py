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
        self.project = torch.nn.Linear(100, 4 * 4 * 1024)
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 3, 5, 2, 2, 1, bias=False),
            torch.nn.Tanh(),
        )
        
    def forward(self, X) -> None:
        projection = self.project(X).reshape(-1, 1024, 4, 4)
        return self.model(projection)
    
if __name__ == "__main__":
    model = DCGANGenerator()
    X = torch.randn(16, 100)
    print(model(X).shape)