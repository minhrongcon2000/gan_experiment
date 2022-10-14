import torch


# Implement maxout layers according to https://arxiv.org/abs/1302.4389v4
class Maxout(torch.nn.Module):
    def __init__(self, input_dim=784, num_piece=2, num_unit=240):
        super().__init__()
        self.input_dim = input_dim
        self.num_unit = num_unit
        self.num_piece = num_piece
        self.linear = torch.nn.Linear(input_dim, self.num_piece * self.num_unit)
        
    def forward(self, X):
        return self.linear(X).view(-1, self.num_piece, self.num_unit).max(dim=1)[0]
