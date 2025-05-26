import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils

from route_choice.layers import EdgeProb, ValueIterationSolver, FixedPointSolver
from route_choice.data.utils import compute_values_probs_flows, normalize_attrs


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs_vi(small_network: nx.MultiDiGraph):
    source_graph, state_graph, orig, dest = small_network

    torch_graph = torch_geometric.utils.from_networkx(state_graph)

    value_iter = ValueIterationSolver(node_dim=0)
    edge_prob = EdgeProb(node_dim=0)

    max_iters = 100
    values, n_iters = value_iter.iterate_to_convergence(
        torch_graph.util, torch_graph.edge_index, torch_graph.is_dest, max_iters
    )
    probs = edge_prob(values, torch_graph.util, torch_graph.edge_index)

    assert n_iters <= max_iters, f"values did not converge in {max_iters} iterations, something is wrong"

    # to get the corresponding node value we have to index torch_graph -> state_graph -> source_graph
    state_list = list(state_graph.nodes)
    for i in range(torch_graph.num_nodes):
        # skip the dummy states because they don't exist in source_graph
        if not torch_graph.is_dummy[i]:
            state_idx = torch_graph.nx_node_idx[i]
            e = state_list[state_idx]
            n = e[1]  # destination node

            edge_prob = source_graph.edges[e]["prob"]
            assert torch.isclose(
                torch_geometric.utils.mask_select(probs, -1, torch_graph.edge_index[1] == i),
                torch.as_tensor(edge_prob),
                atol=1e-4,
            ).all()

            node_value = source_graph.nodes[n]["value"]
            assert torch.isclose(values[i], torch.as_tensor(node_value), atol=1e-4)


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs_fixed_point(small_network: nx.MultiDiGraph):
    source_graph, state_graph, orig, dest = small_network

    torch_graph = torch_geometric.utils.from_networkx(state_graph)
    batch = torch.zeros(torch_graph.num_nodes, dtype=torch.int64)

    lin_eqs = FixedPointSolver(max_iters=100)
    edge_prob = EdgeProb(node_dim=0)

    values = lin_eqs(torch_graph.util, torch_graph.edge_index, torch_graph.is_dest, batch)
    probs = edge_prob(values, torch_graph.util, torch_graph.edge_index)

    # to get the corresponding node value we have to index torch_graph -> state_graph -> source_graph
    state_list = list(state_graph.nodes)
    for i in range(torch_graph.num_nodes):
        # skip the dummy states because they don't exist in source_graph
        if not torch_graph.is_dummy[i]:
            state_idx = torch_graph.nx_node_idx[i]
            e = state_list[state_idx]
            n = e[1]  # destination node

            edge_prob = source_graph.edges[e]["prob"]
            assert torch.isclose(
                torch_geometric.utils.mask_select(probs, -1, torch_graph.edge_index[1] == i),
                torch.as_tensor(edge_prob),
                atol=1e-4,
            ).all()

            node_value = source_graph.nodes[n]["value"]
            assert torch.isclose(values[i], torch.as_tensor(node_value), atol=1e-4)


@pytest.mark.parametrize("small_network", [{"cyclic": False}, {"cyclic": True}], indirect=True)
def test_values_and_probs_lin_eqs(small_network: nx.MultiDiGraph):
    source_graph, state_graph, orig, dest = small_network

    M, state_list = nx.attr_matrix(state_graph, "M")
    V, P, F = compute_values_probs_flows(M, state_list.index(orig), state_list.index(dest))

    # the states are edges in the source graph
    for n, value in source_graph.nodes(data="value"):
        # the value of the edges going in to the node should be equal to the value of the node
        for u, v, k in source_graph.in_edges(n, keys=True):
            i = state_list.index((u, v, k))
            assert np.isclose(value, V[i], atol=1e-4), f"value {n} did not match {u, v, k} ({value} vs {V[i]})"

    for u, v, k, prob in source_graph.edges(keys=True, data="prob"):
        # the transition probabilities going into the edge should be equal to the probability of that edge
        i = state_list.index((u, v, k))
        is_close = np.isclose(prob, P[:, i], atol=1e-4) | (P[:, i] == 0)
        assert is_close.all(), f"prob {u, v, k} did not match ({prob} vs {P[:, i]})"
