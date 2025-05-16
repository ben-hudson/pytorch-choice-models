import networkx as nx

from typing import Any


def shortestpath_edges(graph: nx.Graph, source: Any, target: Any, weight: str = "weight", method: str = "bellman-ford"):
    path_nodes = nx.shortest_path(graph, source=source, target=target, weight=weight, method=method)
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

    # it is not a multigraph, solution is easy
    if not graph.is_multigraph():
        path_length = nx.shortest_path_length(graph, source=source, target=target, weight=weight, method=method)
        return path_edges, path_length

    multigraph_path_edges = []
    multigraph_path_length = 0
    for u, v in path_edges:
        edges = filter(lambda e: e[1] == v, graph.edges(u, keys=True, data=weight))  # filter for edges between u and v
        shortest_edge = min(edges, key=lambda e: e[3])  # find the shortest one (e[3] is the weight)
        multigraph_path_edges.append(shortest_edge[0:3])
        multigraph_path_length += shortest_edge[3]

    return multigraph_path_edges, multigraph_path_length
