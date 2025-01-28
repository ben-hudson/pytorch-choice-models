import pytest
import networkx as nx
import numpy as np
import torch_geometric.utils
import torch

from layers.exp_value_iteration import ExpValueIteration


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_solve_bellman_eq(small_network: nx.MultiDiGraph):
    # first solve using system of linear equations method
    for u, v, k, cost in small_network.edges(keys=True, data="cost"):
        small_network.edges[u, v, k]["exp_util"] = np.exp(-cost)  # exp happens before summing

    # attr_matrix automatically sums values on parallel edges, which is what we want
    M, node_list = nx.attr_matrix(small_network, edge_attr="exp_util")

    dest = node_list.index(4)

    b = np.zeros_like(node_list, dtype=float)
    b[dest] = 1.0
    A = np.eye(M.shape[0]) - M
    z = np.linalg.solve(A, b)
    V = np.log(z)  # correct values from https://arxiv.org/abs/1905.00883v2
    value_true = torch.as_tensor(V, dtype=torch.float32)

    torch_graph = torch_geometric.utils.from_networkx(small_network)
    torch_graph.util = -torch_graph.cost.float().unsqueeze(1)
    layer = ExpValueIteration()
    max_iters = 100
    dest_mask = torch.as_tensor(b, dtype=torch.bool).unsqueeze(1)
    value, n_iters = layer.iterate_to_convergence(torch_graph.util, torch_graph.edge_index, dest_mask, max_iters)

    assert n_iters <= max_iters, f"values did not converge in {max_iters} iterations, something is wrong"
    assert torch.isclose(value.squeeze(), value_true).all(), f"values did not match ({value.squeeze()} vs {value_true})"
