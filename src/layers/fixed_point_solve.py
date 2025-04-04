import torch
import torchdeq
import torch_geometric.utils


class FixedPointSolver(torch.nn.Module):
    def __init__(self, solver: str = "broyden", max_iters: int = 20, eps: float = 1e-6, **kwargs):
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
        b, real_node_mask = torch_geometric.utils.to_dense_batch(sink_node_mask, batch=batch, fill_value=torch.nan)

        batch_size, n_nodes = b.shape
        I = torch.eye(n_nodes).expand(batch_size, -1, -1)
        A = I - M
        b = b.float().unsqueeze(-1)  # convert to a batch of column vectors

        # we want to solve Az - b = f(z) = 0, but torchdeq solves f(z) = z
        # So we need to add z on the right hand side, i.e. Az - b + z = z
        f = lambda z: torch.bmm(A, z) - b + z
        z_0 = torch.ones(batch_size, n_nodes, 1)
        z_out, info = self.deq(f, z_0)
        # for our setup, there is always one element in z_out
        z = z_out[-1]
        V = z[real_node_mask].log()
        return V
