import networkx as nx
import numpy as np
import pytest
import torch


@pytest.fixture
def random_strongly_connected_graph(request):
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
