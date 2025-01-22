import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils

from layers.bellman_ford import BellmanFordStep


@pytest.mark.parametrize(
    "random_strongly_connected_graph", [(10, 0.1, 123), (20, 0.2, 456), (30, 0.3, 789)], indirect=True
)
def test_cost_to_go(random_strongly_connected_graph: nx.DiGraph):
    node_list = list(random_strongly_connected_graph.nodes)
    # the nodes in the random graph are not necessarily sequential
    # pytorch geometric does not support this, so we refer to the nodes by their index in node_list instead
    source = 0
    dist_true = nx.single_source_dijkstra_path_length(random_strongly_connected_graph, node_list[source], weight="cost")

    torch_graph = torch_geometric.utils.from_networkx(random_strongly_connected_graph)
    torch_graph.cost = torch_graph.cost.float().unsqueeze(1)

    dist = torch.inf * torch.ones((torch_graph.num_nodes, 1), dtype=torch.float32)
    dist[source] = 0.0
    layer = BellmanFordStep(aggr="min", flow="source_to_target")
    for _ in range(torch_graph.num_nodes - 1):
        dist = layer(dist, torch_graph.cost, torch_graph.edge_index)

    for i, n in enumerate(node_list):
        assert torch.isclose(
            torch.as_tensor(dist_true[n], dtype=torch.float32), dist[i]
        ), f"cost to go did not match for node {n} ({dist_true[n]} vs. {dist[i]})"


@pytest.mark.parametrize(
    "random_strongly_connected_graph", [(10, 0.1, 123), (20, 0.2, 456), (30, 0.3, 789)], indirect=True
)
def test_shortest_path(random_strongly_connected_graph: nx.DiGraph):
    node_list = list(random_strongly_connected_graph.nodes)
    # the nodes in the random graph are not necessarily sequential
    # pytorch geometric does not support this, so we refer to the nodes by their index in node_list instead
    source = 0
    sink = len(node_list) - 1

    path_true = nx.shortest_path(random_strongly_connected_graph, node_list[source], node_list[sink], weight="cost")
    assert len(path_true) > 2, "path is trivial"

    torch_graph = torch_geometric.utils.from_networkx(random_strongly_connected_graph)
    torch_graph.cost = torch_graph.cost.float().unsqueeze(1)

    # here we will get the cost to go from every node to the destination
    dist = torch.inf * torch.ones((torch_graph.num_nodes, 1), dtype=torch.float32)
    dist[sink] = 0.0
    layer = BellmanFordStep(aggr="min", flow="target_to_source")
    for _ in range(torch_graph.num_nodes - 1):
        dist = layer(dist, torch_graph.cost, torch_graph.edge_index)

    i = source
    path = [node_list[i]]
    while i != sink:
        out_edges = torch_graph.edge_index[0] == i
        children = torch_graph.edge_index[1][out_edges]
        cost_to_go = torch_graph.cost[out_edges] + dist[children]
        i = children[torch.argmin(cost_to_go)].item()
        path.append(node_list[i])

    assert all([a == b for a, b in zip(path_true, path)]), f"paths did not match ({path_true} vs. {path})"
