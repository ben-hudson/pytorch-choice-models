import torch

from torch_geometric.nn import MessagePassing


class ExpValueIteration(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr="sum", flow="target_to_source", **kwargs)

    # remember these are exp(V)s and exp(u)s
    def forward(self, exp_values: torch.Tensor, exp_utils: torch.Tensor, edge_index: torch.Tensor):
        updated_dist = self.propagate(edge_index, exp_value=exp_values, exp_util=exp_utils)
        return updated_dist

    def message(self, exp_value_j: torch.Tensor, exp_util: torch.Tensor):
        return exp_util * exp_value_j  # exp(util + value) = exp(util)*exp(value)

    def update(self, inputs: torch.Tensor, exp_value: torch.Tensor):
        # inputs are the updated values
        return torch.max(inputs, exp_value)

    def iterate_to_convergence(
        self,
        utils: torch.Tensor,
        edge_index: torch.Tensor,
        sink_node_mask: torch.Tensor,
        max_iters: int,
        eps: float = 1e-6,
    ):
        exp_utils = utils.exp()
        exp_values = torch.zeros((sink_node_mask.size(0), 1), dtype=torch.float32)  # exp(-inf)
        exp_values[sink_node_mask] = 1.0  # exp(0)

        n_iters = 0
        while n_iters < max_iters:
            prev_exp_values = exp_values.detach().clone()
            exp_values = self.forward(exp_values, exp_utils, edge_index)
            if torch.isclose(exp_values, prev_exp_values, atol=eps).all():
                break
            n_iters += 1

        return exp_values.log(), n_iters
