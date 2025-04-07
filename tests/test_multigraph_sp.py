import networkx as nx
import pytest
import random

from route_choice.utils import shortestpath_edges


@pytest.mark.parametrize(
    "random_graph",
    [
        {"max_nodes": 10, "edge_prob": 0.1, "seed": 123},
        {"max_nodes": 20, "edge_prob": 0.2, "seed": 456},
        {"max_nodes": 30, "edge_prob": 0.3, "seed": 789},
    ],
    indirect=True,
)
def test_multigraph_sp(random_graph: nx.DiGraph):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]

    # find the true shortest path between source and sink
    path_nodes = nx.shortest_path(random_graph, source, sink, weight="cost")
    assert len(path_nodes) > 2, "path is trivial"
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

    # convert to a multigraph
    multigraph = nx.MultiDiGraph(random_graph)

    # add a parallel edge with a slightly lower cost somewhere along the shortest path
    # the multigraph shortest path should use this edge instead
    selected_edge = random.choice(path_edges)
    selected_edge_cost = random_graph.edges[selected_edge]["cost"]
    multigraph.add_edge(*selected_edge, cost=selected_edge_cost * 0.9)

    multigraph_path_edges, _ = shortestpath_edges(multigraph, source, sink, weight="cost")

    for a, (u, v, k) in zip(path_edges, multigraph_path_edges):
        b = u, v
        assert a == b, f"expected the shortest path to be along the edge {a} not {b}"
        if a == selected_edge:
            assert k == 1, f"expected the shortest path to be along the edge {(u, v, 1)}"
