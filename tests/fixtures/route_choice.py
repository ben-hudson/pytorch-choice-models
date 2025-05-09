import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
import torch_geometric.data
import torch_geometric.utils

from route_choice.utils import random_strongly_connected_graph, sample_paths, solve_bellman_lin_eqs
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def random_graph(request: pytest.FixtureRequest):
    max_nodes = request.param.get("max_nodes", 10)
    edge_prob = request.param.get("edge_prob", 0.1)
    seed = request.param.get("seed", None)

    return random_strongly_connected_graph(max_nodes, edge_prob, seed)


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
def small_network(request: pytest.FixtureRequest):
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
    graph = rl_tutorial_network.copy()
    n_samples = request.param.get("n_samples", 500)
    seed = request.param.get("seed", None)

    # edge features
    feat_attrs = ["travel_time"]
    n_feats = len(feat_attrs)

    orig = "o"
    dest = "d"
    # node features
    for n in graph.nodes:
        graph.nodes[n]["orig"] = n == orig
        graph.nodes[n]["dest"] = n == dest

    # deterministic util
    beta_tt = -2.0  # coefficient for travel time
    beta_lc = -0.01  # coefficient for link constant (penalizes number of links in a path)
    for e in graph.edges:
        travel_time = graph.edges[e]["travel_time"]
        link_constant = 1
        determ_util = beta_tt * travel_time + beta_lc * link_constant
        graph.edges[e]["determ_util"] = determ_util

    # deterministic value
    values = solve_bellman_lin_eqs(graph, dest, util_key="determ_util")
    nx.set_node_attributes(graph, values, "value")

    # now, generate samples
    # we want to make sure the edge ordering doesn't change once converted to PyG, so store it
    for i, e in enumerate(graph.edges):
        graph.edges[e]["nx_edge_index"] = i

    # this operation is slow so we only want to do it once
    torch_graph = torch_geometric.utils.from_networkx(graph, group_edge_attrs=feat_attrs)

    paths = sample_paths(graph, orig, dest, "determ_util", "value", n_samples, seed=seed)
    samples = []
    for path in paths:
        sample = torch_graph.clone()

        chosen_edges_mask = torch.as_tensor([e in path for e in graph.edges])  # convert path to mask
        sample.choice = chosen_edges_mask[torch_graph.nx_edge_index]  # reindex according to PyG edge order
        samples.append(sample)

    batch = torch_geometric.data.Batch.from_data_list(samples)

    feat_scaler = StandardScaler()
    feats_scaled_np = feat_scaler.fit_transform(batch.edge_attr.numpy())
    batch.feats = torch.as_tensor(feats_scaled_np, dtype=torch.float32)

    return batch, feat_scaler, n_feats
