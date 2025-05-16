import torch
import torch_geometric.utils

from torch_geometric.nn import MessagePassing


class EdgeProb(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr=None, flow="target_to_source", **kwargs)

    def forward(self, values: torch.Tensor, utils: torch.Tensor, edge_index: torch.Tensor):
        probs = self.propagate(edge_index, value=values, util=utils)
        return probs

    def message(self, value_j: torch.Tensor, util: torch.Tensor):
        return util + value_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        probs = torch_geometric.utils.softmax(inputs, index)
        return probs
