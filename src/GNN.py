#imports
import torch
import torch.nn as nn
from torch_geometric import data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

data_object = torch.load("../data/data.pt")

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# model params
input_dim = 4
hidden_dim = 8
output_dim = 2

# model = GNN(input_dim, hidden_dim, output_dim)

print(data_object)