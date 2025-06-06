import torch
import torch_geometric.data

from .layers import EdgeProb, FixedPointSolver, ValueIterationSolver


class RecursiveLogit(torch.nn.Module):
    def __init__(self, n_feats: int, link_constant: bool = True, use_value_iteration: bool = False):
        super().__init__()
        # it seems there are no alternative specific constants in the recursive logit model
        # there is often a "link constant" which is 1 for every link and has it's own beta
        # we implement this as a learned bias
        self.coeffs = torch.nn.Linear(n_feats, 1, bias=link_constant)
        torch.nn.init.constant_(self.coeffs.weight, -1.0)
        if link_constant:
            torch.nn.init.constant_(self.coeffs.bias, -1.0)

        self.use_value_iteration = use_value_iteration
        if self.use_value_iteration:
            self.solver = ValueIterationSolver()
        else:
            self.solver = FixedPointSolver()
        self.edge_prob = EdgeProb()

    def forward(
        self, feats: torch.Tensor, dest_mask: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, n_nodes: int
    ):
        # we get a batch of graphs, where each edge has a feature
        # first, we estimate the deterministic utility of each edge
        # the value iteration step depends on utilities being negative for numerical stability
        util = self.coeffs(feats).clamp(min=-100, max=-1e-6)

        if self.use_value_iteration:
            # we calculate the value of each node using the message passing steps
            # we use number of nodes in the batch as the maximum number of iterations
            # it should be much less than this, because each individual graph in the batch is much smaller
            value, _ = self.solver.iterate_to_convergence(util, edge_index, dest_mask, max_iters=n_nodes)
        else:
            value = self.solver(util, edge_index, dest_mask, batch)

        prob = self.edge_prob(value, util, edge_index)

        return value, util, prob

    def train_step(self, batch: torch_geometric.data.Batch, loss_reduction: str = "mean"):
        value, util, prob = self.forward(batch.feats, batch.is_dest, batch.edge_index, batch.batch, batch.num_nodes)
        info = {"value": value.detach(), "util": util.detach(), "prob": prob.detach()}
        # now to compute the loss of prob wrt to choice
        nll = -torch.log(prob[batch.choice])
        if loss_reduction == "mean":
            return nll.mean(), info
        elif loss_reduction == "sum":
            return nll.sum(), info
        else:
            raise ValueError(f"unknown reduction: {loss_reduction}")

    def get_params(self):
        return {name: param.detach() for name, param in self.named_parameters()}
