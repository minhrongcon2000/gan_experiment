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