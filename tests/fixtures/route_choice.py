import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
import torch_geometric.data
import torch_geometric.utils

from route_choice.utils import sample_paths, solve_bellman_lin_eqs
from route_choice.data import load_small_acyclic_network, load_small_cyclic_network, load_tutorial_network
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def random_graph(request: pytest.FixtureRequest):
    max_nodes = request.param.get("max_nodes", 10)
    edge_prob = request.param.get("edge_prob", 0.1)
    seed = request.param.get("seed", None)

    # generate a graph
    H = nx.fast_gnp_random_graph(max_nodes, edge_prob, directed=True, seed=seed)
    # find the largest component
    largest_component = max(nx.strongly_connected_components(H), key=len)
    assert len(largest_component) > 2, "largest component is trivial"

    # take that component and add edge costs
    G = H.subgraph(largest_component).copy()
    node_pos = nx.spring_layout(G)
    nx.set_node_attributes(G, node_pos, "pos")
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


@pytest.fixture
def small_network(request: pytest.FixtureRequest):
    if request.param.get("cyclic", False):
        return load_small_cyclic_network()
    else:
        return load_small_acyclic_network()


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
def rl_tutorial_dataset(request: pytest.FixtureRequest):
    graph = load_tutorial_network()
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
