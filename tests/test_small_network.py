import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils

from layers.edge_prob import EdgeProb
from layers.exp_value_iteration import ExpValueIteration
from route_choice.utils import solve_bellman_lin_eqs


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs(small_network: nx.MultiDiGraph):
    node_list = list(small_network.nodes)
    dest = len(node_list) - 1

    torch_graph = torch_geometric.utils.from_networkx(small_network)
    torch_graph.util = -torch_graph.cost.float().unsqueeze(1)
    value_iter = ExpValueIteration()
    max_iters = 100
    dest_mask = torch.zeros(torch_graph.num_nodes, dtype=torch.bool)
    dest_mask[dest] = True
    value, n_iters = value_iter.iterate_to_convergence(torch_graph.util, torch_graph.edge_index, dest_mask, max_iters)

    assert n_iters <= max_iters, f"values did not converge in {max_iters} iterations, something is wrong"
    assert torch.isclose(
        value.squeeze(), torch_graph.value, atol=1e-4
    ).all(), f"values did not match ({value.squeeze()} vs {torch_graph.value})"

    edge_prob = EdgeProb()
    prob = edge_prob(value, torch_graph.util, torch_graph.edge_index)

    assert torch.isclose(
        prob.squeeze(), torch_graph.prob, atol=1e-4
    ).all(), f"edge probs did not match ({prob.squeeze()} vs {torch_graph.prob})"


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_lin_eq(small_network: nx.MultiDiGraph):
    for u, v, k, cost in small_network.edges(keys=True, data="cost"):
        small_network.edges[u, v, k]["util"] = -cost

    values, node_list = solve_bellman_lin_eqs(small_network, util_key="util")
    for value_n, n in zip(values, node_list):
        assert np.isclose(
            value_n, small_network.nodes[n]["value"], atol=1e-4
        ), f"value did not match ({value_n} vs {small_network.nodes[n]['value']})"
