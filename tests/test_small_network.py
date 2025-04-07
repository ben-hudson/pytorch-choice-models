import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils

from layers import EdgeProb, ValueIterationSolver, FixedPointSolver
from route_choice.utils import solve_bellman_lin_eqs, get_edge_probs


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs_vi(small_network: nx.MultiDiGraph):
    node_list = list(small_network.nodes)
    dest = len(node_list) - 1

    torch_graph = torch_geometric.utils.from_networkx(small_network)
    torch_graph.util = -torch_graph.cost.float().unsqueeze(1)
    value_iter = ValueIterationSolver()
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
def test_values_and_probs_fixed_point(small_network: nx.MultiDiGraph):
    node_list = list(small_network.nodes)
    dest = len(node_list) - 1

    torch_graph = torch_geometric.utils.from_networkx(small_network)
    torch_graph.util = -torch_graph.cost.float().unsqueeze(1)
    max_iters = 100
    lin_eqs = FixedPointSolver(max_iters=max_iters)
    dest_mask = torch.zeros(torch_graph.num_nodes, dtype=torch.bool)
    dest_mask[dest] = True
    batch = torch.zeros(torch_graph.num_nodes, dtype=torch.int64)
    value = lin_eqs(torch_graph.util, torch_graph.edge_index, dest_mask, batch)

    assert torch.isclose(
        value.squeeze(), torch_graph.value, atol=1e-4
    ).all(), f"values did not match ({value.squeeze()} vs {torch_graph.value})"

    edge_prob = EdgeProb()
    prob = edge_prob(value, torch_graph.util, torch_graph.edge_index)

    assert torch.isclose(
        prob.squeeze(), torch_graph.prob, atol=1e-4
    ).all(), f"edge probs did not match ({prob.squeeze()} vs {torch_graph.prob})"


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs_lin_eqs(small_network: nx.MultiDiGraph):
    dest = 4

    for u, v, k, cost in small_network.edges(keys=True, data="cost"):
        small_network.edges[u, v, k]["util"] = -cost

    values = solve_bellman_lin_eqs(small_network, dest, util_key="util")
    for n, value in values.items():
        assert np.isclose(
            value, small_network.nodes[n]["value"], atol=1e-4
        ), f"value did not match ({value} vs {small_network.nodes[n]['value']})"

    nx.set_node_attributes(small_network, values, "value")
    edge_probs = get_edge_probs(small_network, util_key="util", value_key="value")
    for e, prob in edge_probs.items():
        assert np.isclose(
            prob, small_network.edges[e]["prob"], atol=1e-4
        ), f"edge prob did not match ({prob} vs {small_network.edges[e]['prob']})"
