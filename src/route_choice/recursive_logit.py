import torch
import torch_geometric.nn
import torch_geometric.data
import torch_geometric.utils

from layers import BellmanFordStep


class RecursiveLogitLoss(torch_geometric.nn.MessagePassing):
    def __init__(self, reduction: str = "mean"):
        super().__init__(aggr=None, flow="target_to_source")
        self.reduction = reduction

    def forward(self, value: torch.Tensor, util: torch.Tensor, choice: torch.Tensor, edge_index: torch.Tensor):
        loss = self.propagate(edge_index, value=value, util=util, choice=choice)
        return loss

    def message(self, value_j: torch.Tensor, util: torch.Tensor, choice: torch.Tensor):
        # the value of an edge is the immediate utility (reward) of the edge + the value of the next node
        return value_j + util, choice

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        values, choices = inputs
        probs = torch_geometric.utils.softmax(values, index)
        nll = -probs[choices].log()

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            raise ValueError(f"unknown reduction: {self.reduction}")


class RecursiveLogit(torch.nn.Module):
    def __init__(self, n_feats: int, link_constant: bool = True, loss_reduction: str = "mean"):
        super().__init__()
        # it seems there are no alternative specific constants in the recursive logit model
        # there is often a "link constant" which is 1 for every link and has it's own beta
        # we implement this as a learned bias
        self.coeffs = torch.nn.Linear(n_feats, 1, bias=link_constant)
        self.message_passing = BellmanFordStep(aggr="max", flow="target_to_source")
        self.loss_fn = RecursiveLogitLoss(reduction=loss_reduction)

    def forward(
        self, feats: torch.Tensor, dest_mask: torch.Tensor, batch_index: torch.Tensor, edge_index: torch.Tensor
    ):
        # we get a batch of graphs, where each edge has a feature
        # first, we estimate the deterministic utility of each edge
        # then we calculate the value of each node using the message passing steps
        util = self.coeffs(feats)

        # here we will get value of every edge
        # the value is the opposite of the cost to go
        n_nodes = batch_index.size(0)
        value = -torch.inf * torch.ones((n_nodes, 1), dtype=torch.float32)
        value[dest_mask] = 0.0

        # need to propagate n_nodes - 1 times to converge
        # but the batch is many disconnected graphs, so we need to find the number of nodes in the largest one
        _, node_counts = torch.unique(batch_index, return_counts=True)
        n_steps = node_counts.max() - 1
        for _ in range(n_steps):
            value = self.message_passing(value, util, edge_index)
        # if the values change after they should have converged, we have a cycle
        cycle_check = self.message_passing(value, util, edge_index)
        assert (
            value == cycle_check
        ).all(), "values changed after they should have converged, indicating the graph contains a cycle"

        return value, util

    def train_step(self, batch: torch_geometric.data.Batch):
        value, util = self.forward(batch.feats, batch.dest, batch.batch, batch.edge_index)
        loss = self.loss_fn(value, util, batch.choice, batch.edge_index)
        return loss

    def get_params(self):
        params = dict(self.named_parameters())
        return {"beta_tt": params["coeffs.weight"].detach(), "beta_lc": params["coeffs.bias"].detach()}
