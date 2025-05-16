import networkx as nx
import numpy as np
import pytest
import uuid

from route_choice.data.tutorial import (
    load_small_acyclic_network,
    load_small_cyclic_network,
    load_tutorial_network,
    ToyRouteChoiceDataset,
)
from route_choice.data.utils import get_state_graph
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
        source_graph = load_small_cyclic_network()
    else:
        source_graph = load_small_acyclic_network()

    orig = 1
    dest = 4
    feat_fn = lambda source_graph, k, a: {"cost": source_graph.edges[a]["cost"] if a is not None else 0}
    util_fn = lambda feats: -feats["cost"]
    util_scale = 1.0
    state_graph = get_state_graph(source_graph, orig, dest, feat_fn, util_fn, util_scale)

    return source_graph, state_graph, orig, dest


@pytest.fixture
def rl_tutorial_dataset(request: pytest.FixtureRequest):
    n_samples = request.param.get("n_samples", 500)
    seed = request.param.get("seed", None)

    orig = "o"
    dest = "d"
    feat_attrs = ["travel_time"]
    feat_fn = lambda source_graph, k, a: {"travel_time": source_graph.edges[a]["travel_time"] if a is not None else 0}
    util_fn = lambda feats: -2.0 * feats["travel_time"] - 0.01
    util_scale = 1.0

    dataset = ToyRouteChoiceDataset(
        f"/tmp/tutorial_network_dataset_{uuid.uuid4()}",
        load_tutorial_network(),
        feat_attrs,
        feat_fn,
        util_fn,
        util_scale,
        orig,
        dest,
        n_samples,
        seed=seed,
        force_reload=True,
    )

    feat_scaler = StandardScaler()
    feat_scaler.fit(dataset.edge_attr.numpy())

    return dataset, feat_scaler, len(feat_attrs)
