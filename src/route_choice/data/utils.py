import networkx as nx
import numpy as np

from typing import Any, Callable, Iterable


def get_state_graph(
    source_graph: nx.MultiDiGraph, orig: Any, dest: Any, state_feat_fn: Callable = None, trans_feat_fn: Callable = None
) -> nx.DiGraph:
    state_graph = nx.line_graph(source_graph, create_using=nx.DiGraph)
    state_graph.add_node(orig, is_dummy=True, is_orig=True)
    state_graph.add_node(dest, is_dummy=True, is_dest=True)
    for k, is_dummy in state_graph.nodes(data="is_dummy", default=False):
        if not is_dummy and k[0] == orig:
            state_graph.add_edge(orig, k)
        if not is_dummy and k[1] == dest:
            state_graph.add_edge(k, dest)

        if state_feat_fn is not None:
            k_source_edge = None if is_dummy else k

            feats = state_feat_fn(source_graph, k_source_edge)
            state_graph.nodes[k].update(**feats)

    if trans_feat_fn is not None:
        for k, a in state_graph.edges:
            k_is_orig = state_graph.nodes[k].get("is_orig", False)
            a_is_dest = state_graph.nodes[a].get("is_dest", False)

            k_source_edge = None if k_is_orig else k
            a_source_edge = None if a_is_dest else a

            feats = trans_feat_fn(source_graph, k_source_edge, a_source_edge)
            state_graph.edges[k, a].update(**feats)

    return state_graph


def compute_values_probs_flows(M: np.ndarray, orig_idx: int, dest_idx: int):
    b = np.zeros(M.shape[0])
    b[dest_idx] = 1.0
    z = np.linalg.solve(np.eye(M.shape[0]) - M, b)
    V = np.log(z)

    P = np.zeros_like(M)
    for k, M_k in enumerate(M):
        # the transition probabilities are zero in the terminal state
        if k != dest_idx:
            P[k, :] = M_k * z / (M_k @ z)

    # the link flows, finally
    G = np.zeros(M.shape[0])
    G[orig_idx] = 1
    F = np.linalg.solve(np.eye(P.shape[0]) - P.T, G)

    return V, P, F


def normalize_attrs(graph: nx.DiGraph, on: str, attrs_to_keep: Iterable[str] = "all", default: Any = 0.0):
    def iter_element_data():
        if on == "nodes":
            for k, data in graph.nodes(data=True):
                yield k, data
        elif on == "edges":
            for k, a, data in graph.edges(data=True):
                yield (k, a), data
        else:
            raise ValueError("'on' must be 'nodes' or 'edges'.")

    if attrs_to_keep == "all":
        # get a superset of attrs
        attrs_to_keep = set()
        for elem, data in iter_element_data():
            for attr_name in data.keys():
                attrs_to_keep.add(attr_name)

    for elem, data in iter_element_data():
        # zero-fill the missing ones
        for attr_name in attrs_to_keep:
            if attr_name not in data:
                data[attr_name] = default
        # delete the extra ones
        for attr_name in data:
            if attr_name not in attrs_to_keep:
                del data[attr_name]
