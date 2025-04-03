import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils

from layers import EdgeProb, ExpValueIteration, ExpLinearEquations


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
def test_values_and_probs_lin_eq(small_network: nx.MultiDiGraph):
    node_list = list(small_network.nodes)
    dest = len(node_list) - 1

    torch_graph = torch_geometric.utils.from_networkx(small_network)
    torch_graph.util = -torch_graph.cost.float().unsqueeze(1)
    lin_eq = ExpLinearEquations()
    dest_mask = torch.zeros(torch_graph.num_nodes, dtype=torch.bool)
    dest_mask[dest] = True
    batch = torch.zeros(torch_graph.num_nodes, dtype=torch.int64)
    value = lin_eq(torch_graph.util, torch_graph.edge_index, dest_mask, batch)

    assert torch.isclose(
        value.squeeze(), torch_graph.value, atol=1e-4
    ).all(), f"values did not match ({value.squeeze()} vs {torch_graph.value})"

    edge_prob = EdgeProb()
    prob = edge_prob(value, torch_graph.util, torch_graph.edge_index)

    assert torch.isclose(
        prob.squeeze(), torch_graph.prob, atol=1e-4
    ).all(), f"edge probs did not match ({prob.squeeze()} vs {torch_graph.prob})"
