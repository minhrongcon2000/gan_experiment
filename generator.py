import torch


class MNISTGenerator(torch.nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 784
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(1024, self.output_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.model(X).view(-1, 1, 28, 28)