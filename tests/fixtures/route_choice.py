import math
import pytest
import uuid

from route_choice.data.tutorial import (
    load_small_acyclic_network,
    load_small_cyclic_network,
    load_tutorial_network,
    ToyRouteChoiceDataset,
)
from route_choice.data.utils import get_state_graph, normalize_attrs
from sklearn.preprocessing import StandardScaler


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
    state_graph = get_state_graph(source_graph, orig, dest, None, feat_fn)

    for k, a, feats in state_graph.edges(data=True):
        util = util_fn(feats)
        state_graph.edges[k, a]["util"] = util
        state_graph.edges[k, a]["M"] = math.exp(1 / util_scale * util)

    normalize_attrs(state_graph, on="nodes", attrs_to_keep="all")
    normalize_attrs(state_graph, on="edges", attrs_to_keep="all")

    for i, n in enumerate(state_graph.nodes):
        state_graph.nodes[n]["nx_node_idx"] = i

    return source_graph, state_graph, orig, dest


@pytest.fixture
def rl_tutorial_dataset(request: pytest.FixtureRequest):
    n_samples = request.param.get("n_samples", 500)
    seed = request.param.get("seed", None)

    orig = "o"
    dest = "d"
    edge_feat_attrs = ["travel_time"]
    edge_feat_fn = lambda source_graph, k, a: {
        "travel_time": source_graph.edges[a]["travel_time"] if a is not None else 0
    }
    util_fn = lambda feats: -2.0 * feats["travel_time"] - 0.01

    dataset = ToyRouteChoiceDataset(
        f"/tmp/tutorial_network_dataset_{uuid.uuid4()}",
        graph=load_tutorial_network(),
        node_feat_attrs=None,
        node_feat_fn=None,
        edge_feat_attrs=edge_feat_attrs,
        edge_feat_fn=edge_feat_fn,
        util_fn=util_fn,
        source=orig,
        target=dest,
        n_samples=n_samples,
        seed=seed,
        force_reload=True,
    )

    feat_scaler = StandardScaler()
    feat_scaler.fit(dataset.edge_attr.numpy())

    return dataset, feat_scaler, len(edge_feat_attrs)
