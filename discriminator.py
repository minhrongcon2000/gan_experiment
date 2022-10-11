import torch


class MNISTDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 784
        self.output_dim = 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, self.output_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.model(X.view(-1, 784))