import torch
import torch_geometric.nn
import torch_geometric.data
import torch_geometric.utils

from layers import EdgeProb, ExpValueIteration


class RecursiveLogit(torch.nn.Module):
    def __init__(self, n_feats: int, link_constant: bool = True):
        super().__init__()
        # it seems there are no alternative specific constants in the recursive logit model
        # there is often a "link constant" which is 1 for every link and has it's own beta
        # we implement this as a learned bias
        self.coeffs = torch.nn.Linear(n_feats, 1, bias=link_constant)
        torch.nn.init.constant_(self.coeffs.weight, -1.0)
        if link_constant:
            torch.nn.init.constant_(self.coeffs.bias, -1.0)
        self.value_iteration = ExpValueIteration()
        self.edge_prob = EdgeProb()

    def forward(self, feats: torch.Tensor, dest_mask: torch.Tensor, edge_index: torch.Tensor, n_nodes: int):
        # we get a batch of graphs, where each edge has a feature
        # first, we estimate the deterministic utility of each edge
        # the value iteration step depends on utilities being negative for numerical stability
        util = self.coeffs(feats).clamp(min=-100, max=-1e-6)

        # then we calculate the value of each node using the message passing steps
        # we use number of nodes in the batch as the maximum number of iterations
        # it should be much less than this, because each individual graph in the batch is much smaller
        value, n_iters = self.value_iteration.iterate_to_convergence(util, edge_index, dest_mask, n_nodes)
        prob = self.edge_prob(value, util, edge_index)

        return value, util, prob

    def train_step(self, batch: torch_geometric.data.Batch, loss_reduction: str = "mean"):
        value, util, prob = self.forward(batch.feats, batch.dest, batch.edge_index, batch.num_nodes)
        # now to compute the loss of prob wrt to choice
        nll = -torch.log(prob[batch.choice])
        if loss_reduction == "mean":
            return nll.mean()
        elif loss_reduction == "sum":
            return nll.sum()
        else:
            raise ValueError(f"unknown reduction: {loss_reduction}")

    def get_params(self):
        params = dict(self.named_parameters())
        return {"beta": params["coeffs.weight"].detach(), "link_constant": params["coeffs.bias"].detach()}
