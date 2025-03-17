import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, Sequential

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNEncoder, self).__init__()
        layers = []
        layers.append((GCNConv(input_size, hidden_size), 'x, edge_index -> x'), )
        layers.append(BatchNorm(hidden_size))
        layers.append(nn.PReLU())

        layers.append((GCNConv(hidden_size, output_size), 'x, edge_index -> x'), )
        layers.append(BatchNorm(output_size))
        layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()

class MLP_Predictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()