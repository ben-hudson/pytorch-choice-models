import torch

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class BellmanFordStep(MessagePassing):
    def forward(self, dist: torch.Tensor, cost: torch.Tensor, edge_index: torch.Tensor):
        edge_index, cost = add_self_loops(edge_index, cost, fill_value=0.0)
        updated_dist = self.propagate(edge_index, dist=dist, cost=cost)
        return updated_dist

    def message(self, dist_j: torch.Tensor, cost: torch.Tensor):
        return dist_j + cost
