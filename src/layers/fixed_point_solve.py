import torch
import torchdeq
import torch_geometric.utils


class FixedPointSolver(torch.nn.Module):
    def __init__(self, solver: str = "fixed_point_iter", max_iters: int = 1000, eps: float = 1e-16, **kwargs):
        super().__init__()
        self.deq = torchdeq.get_deq(
            ift=True,
            f_solver=solver,
            f_max_iter=max_iters,
            f_tol=eps,
            b_solver=solver,
            b_max_iter=max_iters,
            b_tol=eps,
            **kwargs
        )

    def forward(self, utils: torch.Tensor, edge_index: torch.Tensor, sink_node_mask: torch.Tensor, batch: torch.Tensor):
        exp_utils = utils.exp()

        # this automatically sums values on parallel edges, which is what we want
        M = torch_geometric.utils.to_dense_adj(edge_index, batch=batch, edge_attr=exp_utils.squeeze())
        b, real_node_mask = torch_geometric.utils.to_dense_batch(sink_node_mask, batch=batch, fill_value=False)

        batch_size, n_nodes = b.shape
        b = b.unsqueeze(-1).type_as(M)  # convert to a batch of column vectors

        # we want to solve Mz + b = f(z) = z
        # See https://www.sciencedirect.com/science/article/pii/S0191261513001276
        f = lambda z: torch.bmm(M, z) + b
        # the solution is basically 1 at the destination node and less than 1 everywhere else, so b is a good starting point
        z_0 = b.clone()
        z_out, info = self.deq(f, z_0)
        # for our setup, there is always one element in z_out
        z = z_out[-1]
        V = z[real_node_mask].clamp(min=0).log()
        V = V.masked_fill(torch.isinf(V), torch.nan)
        return V
