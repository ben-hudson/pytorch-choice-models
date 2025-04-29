import networkx as nx
import numpy as np
import random
import torch
import torchdeq

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


def edge_data_iterator(G: nx.Graph, data_key: bool = True):
    if G.is_multigraph():
        for u, v, k, data in G.edges(keys=True, data=data_key):
            yield (u, v, k), data
    else:
        for u, v, data in G.edges(data=data_key):
            yield (u, v), data


# solve the system Mz + b = z
def linear_fp_solver(M: torch.Tensor, b: torch.Tensor, use_numpy: bool = False):
    n_nodes = b.size(-1)

    func = lambda z: torch.bmm(M, z.unsqueeze(2)).squeeze(2) + b

    if use_numpy:
        A_np = np.eye(n_nodes) - M.numpy()
        b_np = b.numpy()
        z_np = np.linalg.solve(A_np, b_np)
        z = torch.as_tensor(z_np).type_as(M)
    else:
        A = torch.eye(n_nodes) - M
        z, info = torch.linalg.solve_ex(A, b)

    indexing_list = []
    info = torchdeq.solver_stat_from_final_step(z, func(z))
    return z, indexing_list, info


def solve_bellman_lin_eqs(graph: nx.MultiDiGraph, target: Any, util_key: str = "util", is_neg: bool = False):
    for e, util in edge_data_iterator(graph, data_key=util_key):
        if is_neg:
            util = -util
        graph.edges[e]["exp_util"] = np.exp(util)  # exp happens before summing

    # attr_matrix automatically sums values on parallel edges, which is what we want
    attr_matrix, node_list = nx.attr_matrix(graph, edge_attr="exp_util")
    M = torch.as_tensor(attr_matrix)

    target_idx = node_list.index(target)
    b = torch.zeros(len(node_list)).type_as(M)
    b[target_idx] = 1.0

    z, _, _ = linear_fp_solver(M.unsqueeze(0), b.unsqueeze(0))
    V = z.squeeze().clamp(min=0).log()
    V = V.masked_fill(torch.isinf(V), torch.nan)

    values = {n: v for n, v in zip(node_list, V.numpy())}
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
    nx.set_node_attributes(G, node_pos, "pos")
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


def get_turn_angle(in_bearing, out_bearing):
    turn_angle = out_bearing - in_bearing
    if turn_angle > 180:
        turn_angle -= 360
    elif turn_angle < -180:
        turn_angle += 360
    return turn_angle
