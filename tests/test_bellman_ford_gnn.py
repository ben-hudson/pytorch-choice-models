import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils

from layers.bellman_ford import BellmanFordStep


@pytest.fixture
def random_graph():
    G = nx.fast_gnp_random_graph(10, 0.1, directed=True, seed=123)
    node_pos = nx.spring_layout(G)
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


def test_cost_to_go(random_graph: nx.DiGraph):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    dist_true = nx.single_source_bellman_ford_path_length(random_graph, source, weight="cost")

    torch_graph = torch_geometric.utils.from_networkx(random_graph)
    torch_graph.cost = torch_graph.cost.float().unsqueeze(1)

    dist = torch.inf * torch.ones((torch_graph.num_nodes, 1), dtype=torch.float32)
    dist[source] = 0.0
    layer = BellmanFordStep(aggr="min", flow="source_to_target")
    for _ in range(torch_graph.num_nodes - 1):
        dist = layer(dist, torch_graph.cost, torch_graph.edge_index)

    for n in dist_true:
        assert torch.isclose(
            torch.as_tensor(dist_true[n], dtype=torch.float32), dist[n]
        ), f"cost to go did not match for node {n} ({dist_true[n]} vs. {dist[n]})"


def test_shortest_path(random_graph: nx.DiGraph):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]
    path_true = nx.shortest_path(random_graph, source, sink, weight="cost")
    assert len(path_true) > 2, "path is trivial"

    torch_graph = torch_geometric.utils.from_networkx(random_graph)
    torch_graph.cost = torch_graph.cost.float().unsqueeze(1)

    # here we will get the cost to go from every node to the destination
    dist = torch.inf * torch.ones((torch_graph.num_nodes, 1), dtype=torch.float32)
    dist[sink] = 0.0
    layer = BellmanFordStep(aggr="min", flow="target_to_source")
    for _ in range(torch_graph.num_nodes - 1):
        dist = layer(dist, torch_graph.cost, torch_graph.edge_index)

    n = source
    path = [n]
    while n != sink:
        out_edges = torch_graph.edge_index[0] == n
        children = torch_graph.edge_index[1][out_edges]
        cost_to_go = torch_graph.cost[out_edges] + dist[children]
        n = children[torch.argmin(cost_to_go)].item()
        path.append(n)

    assert all([a == b for a, b in zip(path_true, path)]), f"paths did not match ({path_true} vs. {path})"
