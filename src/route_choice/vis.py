import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from collections import defaultdict
from typing import Any


def _iter_edges_between_nodes(
    G: nx.MultiDiGraph, u: Any, v: Any, data_key: bool = True, default: Any = None, include_reverse_edges: bool = False
):
    assert G.is_multigraph() and G.is_directed(), "G must be a directed multigraph"

    if G.has_edge(u, v):
        for k, data in G[u][v].items():
            if isinstance(data_key, bool):
                yield (u, v, k), data
            else:
                yield (u, v, k), data.get(data_key, default)

    # there is no point in going over self loops again!
    if include_reverse_edges and u != v:
        for e, data in _iter_edges_between_nodes(
            G, v, u, data_key=data_key, default=default, include_reverse_edges=False
        ):
            yield e, data


def _get_overlapping_edges(G: nx.MultiDiGraph):
    overlaps = defaultdict(list)  # key overlaps with
    checked = set()

    for u, v, k, geom1 in G.edges(keys=True, data="geometry"):
        e1 = (u, v, k)
        # if this edge hasn't already been flagged and has some geometry
        if e1 not in checked:
            # look at every other edge between the nodes, one at a time
            for e2, geom2 in _iter_edges_between_nodes(G, u, v, data_key="geometry", include_reverse_edges=True):
                # don't check against self
                if e1 != e2:
                    if geom1 is not None and geom2 is not None and ox.convert._is_same_geometry(geom1, geom2):
                        overlaps[e1].append(e2)
                checked.add(e2)

    return overlaps


def get_edge_alpha_by_attr(G: nx.MultiDiGraph, edge_attr: str, default: Any = None) -> pd.Series:
    overlapping_edges = _get_overlapping_edges(G)
    edge_alpha = {e: 1.0 for e in G.edges}

    for e, edges in overlapping_edges.items():
        # add the current edge in the list of overlapping edges
        edges.append(e)
        # find the one with the greatest edge_attr
        max_edge = max(edges, key=lambda e: G.edges[e].get(edge_attr, default))
        # make the other edges transparent
        for edge in edges:
            if edge != max_edge:
                edge_alpha[edge] = 0.0

    return pd.Series(edge_alpha)


def plot_edge_attr(G: nx.MultiDiGraph, edge_attr: str, cmap: str = "viridis", **kwargs):
    if "crs" in G.graph:
        # this is probably a osmnx graph
        edge_color = ox.plot.get_edge_colors_by_attr(G, edge_attr, cmap=cmap)
        edge_alpha = get_edge_alpha_by_attr(G, edge_attr)
        return ox.plot_graph(G, edge_color=edge_color, edge_alpha=edge_alpha, **kwargs)

    else:
        raise ValueError("graph does not have a CRS")
