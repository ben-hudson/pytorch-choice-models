import networkx as nx
import numpy as np
import random
import warnings

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


def solve_bellman_lin_eqs(graph: nx.MultiDiGraph, target: Any, util_key: str = "util"):
    for u, v, k, util in graph.edges(keys=True, data=util_key):
        graph.edges[u, v, k]["exp_util"] = np.exp(util)  # exp happens before summing

    # attr_matrix automatically sums values on parallel edges, which is what we want
    M, node_list = nx.attr_matrix(graph, edge_attr="exp_util")

    target_idx = node_list.index(target)

    b = np.zeros_like(node_list, dtype=float)
    b[target_idx] = 1.0
    A = np.eye(M.shape[0]) - M
    z = np.linalg.solve(A, b)
    if (z < 0).any():
        warnings.warn("z has negative elements. This can happen when solving large matrices.")
    z = z.clip(min=0)
    V = np.log(z)

    values = {n: v for n, v in zip(node_list, V)}
    return values


def get_edge_probs(graph: nx.MultiDiGraph, util_key: str = "util", value_key: str = "value"):
    probs = {}

    for n in graph.nodes:
        exp_logits = {}
        exp_logit_sum = 0
        for u, v, k, util in graph.out_edges(n, keys=True, data=util_key):
            value = graph.nodes[v][value_key]
            exp_logit = np.exp(util + value)
            exp_logits[u, v, k] = exp_logit
            exp_logit_sum += exp_logit

        for e, exp_logit in exp_logits.items():
            assert e not in probs, f"{e} is already set!"
            probs[e] = exp_logit / exp_logit_sum

    return probs


def random_strongly_connected_graph(max_nodes, edge_prob, seed):
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


def sample_paths(graph: nx.MultiDiGraph, orig: Any, dest: Any, n_samples: int, prob_key: str = "prob", seed=None):
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
