import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.data
import torch_geometric.utils

from sklearn.preprocessing import StandardScaler
from typing import Any


@pytest.fixture
def random_strongly_connected_graph(request: pytest.FixtureRequest):
    max_nodes, edge_prob, seed = request.param

    # generate a graph
    H = nx.fast_gnp_random_graph(max_nodes, edge_prob, directed=True, seed=seed)
    # find the largest component
    largest_component = max(nx.strongly_connected_components(H), key=len)
    assert len(largest_component) > 2, "largest component is trivial"

    # take that component and add edge costs
    G = H.subgraph(largest_component).copy()
    node_pos = nx.spring_layout(G)
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


# https://arxiv.org/abs/1905.00883v2 figure 1, 2
@pytest.fixture
def small_network(request):
    G = nx.MultiDiGraph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edge(1, 2, cost=1)
    G.add_edge(1, 4, cost=2)
    G.add_edge(1, 4, cost=6)
    G.add_edge(2, 3, cost=1.5)
    G.add_edge(2, 4, cost=2)
    G.add_edge(3, 4, cost=1.5)
    if request.param.get("cyclic", False):
        G.add_edge(3, 1, cost=1)
    return G


# https://arxiv.org/abs/1905.00883v2 figure 3
@pytest.fixture
def route_choice_graph():
    G = nx.MultiDiGraph()
    G.add_node("o", pos=(0, 0))
    G.add_node("A", pos=(1, 0))
    G.add_node("B", pos=(2, 0))
    G.add_node("C", pos=(3, 0))
    G.add_node("D", pos=(4, 0))
    G.add_node("E", pos=(0, 1))
    G.add_node("F", pos=(1, 1))
    G.add_node("H", pos=(2, 1))
    G.add_node("I", pos=(3, 1))
    G.add_node("G", pos=(1, 2))
    G.add_node("d", pos=(4, 2))

    G.add_edge("o", "A", travel_time=0.3)
    G.add_edge("A", "B", travel_time=0.1)
    G.add_edge("B", "C", travel_time=0.1)
    G.add_edge("C", "D", travel_time=0.3)
    G.add_edge("o", "E", travel_time=0.4)
    G.add_edge("A", "F", travel_time=0.1)
    G.add_edge("B", "H", travel_time=0.2)
    G.add_edge("C", "I", travel_time=0.1)
    G.add_edge("C", "d", travel_time=0.9)
    G.add_edge("D", "d", travel_time=2.6)
    G.add_edge("E", "G", travel_time=0.3)
    G.add_edge("F", "G", travel_time=0.3)
    G.add_edge("F", "H", travel_time=0.2)
    G.add_edge("H", "d", travel_time=0.5)
    G.add_edge("H", "I", travel_time=0.2)
    G.add_edge("I", "d", travel_time=0.3)
    G.add_edge("G", "H", travel_time=0.6)
    G.add_edge("G", "d", travel_time=0.7)
    G.add_edge("G", "d", travel_time=2.8)

    return G


# https://arxiv.org/abs/1905.00883v2 section 5.1
@pytest.fixture
def route_choice_dataset(route_choice_graph: nx.MultiDiGraph, request: pytest.FixtureRequest):
    n_samples, seed = request.param
    rng = np.random.default_rng(seed)

    # edge features
    feat_attrs = ["travel_time"]
    n_feats = len(feat_attrs)

    orig = "o"
    dest = "d"
    # node features
    for n in route_choice_graph.nodes:
        route_choice_graph.nodes[n]["orig"] = n == orig
        route_choice_graph.nodes[n]["dest"] = n == dest

    # deterministic util
    beta_tt = -2.0  # coefficient for travel time
    beta_lc = -0.01  # coefficient for link constant (penalizes number of links in a path)
    for e in route_choice_graph.edges:
        travel_time = route_choice_graph.edges[e]["travel_time"]
        link_constant = 1
        determ_util = beta_tt * travel_time + beta_lc * link_constant
        route_choice_graph.edges[e]["determ_util"] = determ_util
        route_choice_graph.edges[e]["cost"] = -determ_util  # min cost = max util

    # deterministic value
    cost_to_go = nx.single_source_bellman_ford_path_length(route_choice_graph.reverse(), dest, weight="cost")
    for n in route_choice_graph.nodes:
        route_choice_graph.nodes[n]["cost_to_go"] = cost_to_go[n]
        route_choice_graph.nodes[n]["value"] = -cost_to_go[n]

    # now, sample paths
    samples = []
    for _ in range(n_samples):
        graph = route_choice_graph.copy()

        path = _sample_path(graph, orig, dest, util_key="determ_util", value_key="value", seed=seed)
        for e in graph.edges:
            graph.edges[e]["choice"] = e in path

        torch_graph = torch_geometric.utils.from_networkx(graph, group_edge_attrs=feat_attrs)
        samples.append(torch_graph)

    batch = torch_geometric.data.Batch.from_data_list(samples)

    feat_scaler = StandardScaler()
    feats_scaled_np = feat_scaler.fit_transform(batch.edge_attr.numpy())
    batch.feats = torch.as_tensor(feats_scaled_np, dtype=torch.float32)

    return batch, feat_scaler, n_feats


def _sample_path(
    graph: nx.MultiDiGraph, orig: Any, dest: Any, util_key: str = "util", value_key: str = "value", seed: int = None
):
    assert graph.is_multigraph() and graph.is_directed(), "expected a directed multigraph"

    rng = np.random.default_rng(seed)

    path = []
    n = orig
    while n != dest:
        util = {}
        for u, v, k, determ_util in graph.out_edges(n, keys=True, data=util_key):
            next_value = graph.nodes[v][value_key]
            random_util = rng.gumbel(0, 1) - np.euler_gamma
            util[u, v, k] = determ_util + next_value + random_util

        edge = max(util, key=lambda k: util[k])
        path.append(edge)
        n = edge[1]

    return path
