#imports
import torch
import traci
import torch.nn as nn
from torch_geometric import data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_dim_duration=1, output_dim_phase=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.duration_head = nn.Linear(output_dim, output_dim_duration)
        self.phase_head = nn.Linear(output_dim, output_dim_phase)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        duration_out = self.duration_head(x)
        phase_out = self.phase_head(x)

        return duration_out, phase_out