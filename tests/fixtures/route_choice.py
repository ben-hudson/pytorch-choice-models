import networkx as nx
import numpy as np
import pandas as pd
import pytest
import random
import torch
import torch_geometric.data
import torch_geometric.utils

from route_choice.utils import solve_bellman_lin_eqs, get_edge_probs
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


# https://arxiv.org/abs/1905.00883v2 figure 1
def small_acyclic_network():
    G = nx.MultiDiGraph()

    G.add_node(1, value=-1.5803)
    G.add_node(2, value=-1.6867)
    G.add_node(3, value=-1.5)
    G.add_node(4, value=0.0)

    G.add_edge(1, 2, cost=1, prob=0.3308)
    G.add_edge(1, 4, cost=2, prob=0.6572)
    G.add_edge(1, 4, cost=6, prob=0.0120)
    G.add_edge(2, 3, cost=1.5, prob=0.2689)
    G.add_edge(2, 4, cost=2, prob=0.7311)
    G.add_edge(3, 4, cost=1.5, prob=1.0)

    return G


# https://arxiv.org/abs/1905.00883v2 figure 2
def small_cyclic_network():
    G = nx.MultiDiGraph()

    G.add_node(1, value=-1.5496)
    G.add_node(2, value=-1.5968)
    G.add_node(3, value=-1.1998)
    G.add_node(4, value=0.0)

    G.add_edge(1, 2, cost=1, prob=0.3509)
    G.add_edge(1, 4, cost=2, prob=0.6374)
    G.add_edge(1, 4, cost=6, prob=0.0117)
    G.add_edge(2, 3, cost=1.5, prob=0.3318)
    G.add_edge(2, 4, cost=2, prob=0.6682)
    G.add_edge(3, 4, cost=1.5, prob=0.7407)
    G.add_edge(3, 1, cost=1, prob=0.2593)

    return G


@pytest.fixture
def small_network(request):
    if request.param.get("cyclic", False):
        return small_cyclic_network()
    else:
        return small_acyclic_network()


# https://arxiv.org/abs/1905.00883v2 figure 3
@pytest.fixture
def rl_tutorial_network():
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


@pytest.fixture
def borlange_network():
    col_dtypes = {"from": int, "to": int, "val": float}

    travel_time = pd.read_csv(
        "data/ATTRIBUTEestimatedtime.txt", sep="\t", header=None, names=col_dtypes.keys(), dtype=col_dtypes
    ).set_index(["from", "to"])

    turn_angle = pd.read_fwf(
        "data/ATTRIBUTEturnangles.txt", header=None, names=col_dtypes.keys(), dtype=col_dtypes
    ).set_index(["from", "to"])

    link_incidence = pd.read_fwf(
        "data/linkIncidence.txt", header=None, names=col_dtypes.keys(), dtype=col_dtypes
    ).set_index(["from", "to"])

    link_incidence["travel_time"] = travel_time["val"]
    link_incidence["turn_angle"] = turn_angle["val"]

    LEFT_TURN_ANGLE = np.pi * 40 / 180
    U_TURN_ANGLE = np.pi * 177 / 180
    link_incidence["left_turn"] = (turn_angle["val"] >= LEFT_TURN_ANGLE) & (turn_angle["val"] < U_TURN_ANGLE)
    link_incidence["u_turn"] = turn_angle["val"] >= U_TURN_ANGLE

    edge_list = link_incidence.drop("val", axis=1).dropna().reset_index()

    G = nx.from_pandas_edgelist(edge_list, source="from", target="to", edge_attr=True, create_using=nx.MultiDiGraph)
    return G


# https://arxiv.org/abs/1905.00883v2 section 5.1
@pytest.fixture
def rl_tutorial_dataset(rl_tutorial_network: nx.MultiDiGraph, request: pytest.FixtureRequest):
    n_samples = request.param.get("n_samples", 500)
    seed = request.param.get("seed", None)

    # edge features
    feat_attrs = ["travel_time"]
    n_feats = len(feat_attrs)

    orig = "o"
    dest = "d"
    # node features
    for n in rl_tutorial_network.nodes:
        rl_tutorial_network.nodes[n]["orig"] = n == orig
        rl_tutorial_network.nodes[n]["dest"] = n == dest

    # deterministic util
    beta_tt = -2.0  # coefficient for travel time
    beta_lc = -0.01  # coefficient for link constant (penalizes number of links in a path)
    for e in rl_tutorial_network.edges:
        travel_time = rl_tutorial_network.edges[e]["travel_time"]
        link_constant = 1
        determ_util = beta_tt * travel_time + beta_lc * link_constant
        rl_tutorial_network.edges[e]["determ_util"] = determ_util

    # deterministic value
    values = solve_bellman_lin_eqs(rl_tutorial_network, dest, util_key="determ_util")
    nx.set_node_attributes(rl_tutorial_network, values, "value")

    edge_probs = get_edge_probs(rl_tutorial_network, util_key="determ_util", value_key="value")
    nx.set_edge_attributes(rl_tutorial_network, edge_probs, "prob")

    # now, generate samples
    paths = _sample_paths(rl_tutorial_network, orig, dest, n_samples, prob_key="prob", seed=seed)

    samples = []
    for path in paths:
        graph = rl_tutorial_network.copy()

        for e in graph.edges:
            graph.edges[e]["choice"] = e in path

        torch_graph = torch_geometric.utils.from_networkx(graph, group_edge_attrs=feat_attrs)
        samples.append(torch_graph)

    batch = torch_geometric.data.Batch.from_data_list(samples)

    feat_scaler = StandardScaler()
    feats_scaled_np = feat_scaler.fit_transform(batch.edge_attr.numpy())
    batch.feats = torch.as_tensor(feats_scaled_np, dtype=torch.float32)

    return batch, feat_scaler, n_feats


def _sample_paths(graph: nx.MultiDiGraph, orig: Any, dest: Any, n_samples: int, prob_key: str = "prob", seed=None):
    assert graph.is_multigraph() and graph.is_directed(), "expected a directed multigraph"

    random.seed(seed)

    paths = []
    for _ in range(n_samples):

        path = []
        n = orig
        while n != dest:
            edges = []
            probs = []
            for u, v, k, prob in graph.out_edges(n, keys=True, data=prob_key):
                edges.append((u, v, k))
                probs.append(prob)
            edge = random.choices(edges, weights=probs, k=1)[0]  # random.choices supports weights, .choice does not
            path.append(edge)
            n = edge[1]

        paths.append(path)

    return paths
