#imports
import torch
import traci
import torch.nn as nn
from torch_geometric import data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_duration, output_dim_phase):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.duration_head = nn.Linear(hidden_dim, output_dim_duration)
        self.phase_head = nn.Linear(hidden_dim, output_dim_phase)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        duration_out = F.relu(self.duration_head(x))
        phase_out = torch.argmax(self.phase_head(x), dim=1)

        return duration_out, phase_out