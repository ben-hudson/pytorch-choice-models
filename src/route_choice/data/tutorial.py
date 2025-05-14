import pathlib
import pickle
from typing import Any, Callable, List, Mapping
from matplotlib import colors, cm, pyplot as plt
import networkx as nx
import numpy as np
import math
import torch
import torch_geometric.data
import torch_geometric.utils

from route_choice.data import get_choice_graph
from route_choice.utils import sample_paths, solve_bellman_lin_eqs


def load_small_acyclic_network():
    G = nx.MultiDiGraph()

    G.add_node(1, value=-1.5803, pos=(0, 0))
    G.add_node(2, value=-1.6867, pos=(1, 0))
    G.add_node(3, value=-1.5, pos=(1, -1))
    G.add_node(4, value=0.0, pos=(0, -1))

    G.add_edge(1, 2, cost=1, prob=0.3308)
    G.add_edge(1, 4, cost=2, prob=0.6572)
    G.add_edge(1, 4, cost=6, prob=0.0120)
    G.add_edge(2, 3, cost=1.5, prob=0.2689)
    G.add_edge(2, 4, cost=2, prob=0.7311)
    G.add_edge(3, 4, cost=1.5, prob=1.0)

    return G


def load_small_cyclic_network():
    G = nx.MultiDiGraph()

    G.add_node(1, value=-1.5496, pos=(0, 0))
    G.add_node(2, value=-1.5968, pos=(1, 0))
    G.add_node(3, value=-1.1998, pos=(1, -1))
    G.add_node(4, value=0.0, pos=(0, -1))

    G.add_edge(1, 2, cost=1, prob=0.3509)
    G.add_edge(1, 4, cost=2, prob=0.6374)
    G.add_edge(1, 4, cost=6, prob=0.0117)
    G.add_edge(2, 3, cost=1.5, prob=0.3318)
    G.add_edge(2, 4, cost=2, prob=0.6682)
    G.add_edge(3, 4, cost=1.5, prob=0.7407)
    G.add_edge(3, 1, cost=1, prob=0.2593)

    return G


def load_tutorial_network():
    G = nx.MultiDiGraph()
    G.add_node("o", pos=(0, 0))
    G.add_node("A", pos=(1, 0))
    G.add_node("B", pos=(2, 0))
    G.add_node("C", pos=(3, 0))
    G.add_node("D", pos=(4, 0))
    G.add_node("E", pos=(0, 1))
    G.add_node("F", pos=(1, 1))
    G.add_node("H", pos=(2, 1))
    G.add_node("I", pos=(3, 1))
    G.add_node("G", pos=(1, 2))
    G.add_node("d", pos=(4, 2))

    G.add_edge("o", "A", travel_time=0.3, flow=87.01)
    G.add_edge("A", "B", travel_time=0.1, flow=46.63)
    G.add_edge("B", "C", travel_time=0.1, flow=25.10)
    G.add_edge("C", "D", travel_time=0.3, flow=0.12)
    G.add_edge("o", "E", travel_time=0.4, flow=12.99)
    G.add_edge("A", "F", travel_time=0.1, flow=37.39)
    G.add_edge("B", "H", travel_time=0.2, flow=24.53)
    G.add_edge("C", "I", travel_time=0.1, flow=18.21)
    G.add_edge("C", "d", travel_time=0.9, flow=6.77)
    G.add_edge("D", "d", travel_time=2.6, flow=0.12)
    G.add_edge("E", "G", travel_time=0.3, flow=12.99)
    G.add_edge("F", "G", travel_time=0.3, flow=12.86)
    G.add_edge("F", "H", travel_time=0.2, flow=24.53)
    G.add_edge("H", "d", travel_time=0.5, flow=30.70)
    G.add_edge("H", "I", travel_time=0.2, flow=30.40)
    G.add_edge("I", "d", travel_time=0.3, flow=48.60)
    G.add_edge("G", "H", travel_time=0.6, flow=12.04)
    G.add_edge("G", "d", travel_time=0.7, flow=13.60)
    G.add_edge("G", "d", travel_time=2.8, flow=0.2)

    return G


def draw_networkx_edge_attr(
    G: nx.MultiDiGraph, pos: Mapping, edge_attr: Any = None, default_color="k", bend: float = 0.1, **kwargs
):
    # drawing multigraph edges is tricky, especially when we want to color them according to a value
    if edge_attr is not None:
        edge_vals = nx.get_edge_attributes(G, edge_attr)
        cmap = kwargs.pop("cmap", cm.get_cmap("viridis"))
        norm = colors.Normalize(min(edge_vals.values()), max(edge_vals.values()))
        sm = cm.ScalarMappable(norm, cmap)
        edge_colors = {e: sm.to_rgba(v) for e, v in edge_vals.items()}
    else:
        edge_colors = {}

    # draw edges one at a time, bending each one appropriately
    for e in G.edges:
        color = edge_colors.get(e, default_color)
        nx.draw_networkx_edges(
            G, pos, edgelist=[e], connectionstyle=f"arc3, rad={-e[2]*bend}", edge_color=color, **kwargs
        )


class ToyRouteChoiceDataset(torch_geometric.data.InMemoryDataset):
    def __init__(
        self,
        root: pathlib.Path,
        graph: nx.MultiDiGraph = None,
        feat_attrs: List[str] = None,
        feat_fn: Callable = None,
        util_fn: Callable = None,
        util_scale: float = None,
        source: Any = None,
        target: Any = None,
        n_samples: int = None,
        seed: int = None,
        **kwargs,
    ):
        self.source_graph = graph
        self.feat_attrs = feat_attrs
        self.feat_fn = feat_fn
        self.util_fn = util_fn
        self.util_scale = util_scale
        self.orig = source
        self.dest = target
        self.n_samples = n_samples
        self.seed = seed

        # this calls download and process if necessary
        # after calling super, the files are guaranteed to exist
        super().__init__(root, **kwargs)
        self.source_graph = pickle.load(open(self.raw_paths[0], "rb"))
        self.state_graph = pickle.load(open(self.processed_paths[0], "rb"))
        # self.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["source_graph.pkl"]

    @property
    def processed_file_names(self):
        return ["state_graph.pkl"]

    def get_state_graph(self, source_graph, orig, dest, feat_fn, util_fn, util_scale):
        state_graph = nx.line_graph(source_graph, create_using=nx.DiGraph)
        state_graph.add_node(orig, is_dummy=True, is_orig=True)
        state_graph.add_node(dest, is_dummy=True, is_dest=True)
        for e, is_dummy in state_graph.nodes(data="is_dummy", default=False):
            if not is_dummy and e[0] == orig:
                state_graph.add_edge(orig, e)
            if not is_dummy and e[1] == dest:
                state_graph.add_edge(e, dest)

        for k, a in state_graph.edges:
            k_is_orig = state_graph.nodes[k].get("is_orig", False)
            a_is_dest = state_graph.nodes[a].get("is_dest", False)

            k_source_edge = None if k_is_orig else k
            a_source_edge = None if a_is_dest else a

            feats = feat_fn(source_graph, k_source_edge, a_source_edge)
            state_graph.edges[k, a].update(**feats)

            util = util_fn(feats)
            state_graph.edges[k, a]["util"] = util
            state_graph.edges[k, a]["M"] = math.exp(1 / util_scale * util)

        return state_graph

    def compute_values_probs_flows(self, M, orig_idx, dest_idx):
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

    def download(self):
        assert all(
            [
                self.source_graph is not None,
                self.orig is not None,
                self.dest is not None,
            ]
        ), f"No file found at {self.raw_paths[0]}. Provide the necessary keyword arguments to regenerate the dataset."

        assert self.orig in self.source_graph.nodes, f"{self.orig} not in source_graph.nodes"
        assert self.dest in self.source_graph.nodes, f"{self.dest} not in source_graph.nodes"
        pickle.dump(self.source_graph, open(self.raw_paths[0], "wb"))

    def process(self):
        # TODO: load source graph here, and store orig and dest inside it
        assert all(
            [
                self.source_graph is not None,
                self.orig is not None,
                self.dest is not None,
                self.feat_fn is not None,
                self.util_fn is not None,
                self.util_scale is not None,
            ]
        ), f"No file found at {self.raw_paths[0]}. Provide the necessary keyword arguments to regenerate the dataset."
        state_graph = self.get_state_graph(
            self.source_graph, self.orig, self.dest, self.feat_fn, self.util_fn, self.util_scale
        )

        M, state_list = nx.attr_matrix(state_graph, "M")
        V, P, F = self.compute_values_probs_flows(M, state_list.index(self.orig), state_list.index(self.dest))

        # reassemble into node and edge-wise dicts
        for i, state_i in enumerate(state_list):
            state_graph.nodes[state_i]["value"] = V[i].item()
            state_graph.nodes[state_i]["unit_flow"] = F[i].item()

            for j, state_j in enumerate(state_list):
                if state_graph.has_edge(state_i, state_j):
                    state_graph.edges[state_i, state_j]["trans_prob"] = P[i, j].item()

        pickle.dump(state_graph, open(self.processed_paths[0], "wb"))

    def sample_paths(self):
        self.nx_graph = nx.read_gml(self.raw_paths[0])
        self.paths = pickle.load(open(self.raw_paths[1], "rb"))

        assert self.feat_attrs is not None, f"Provide the 'feat_attrs' keyword argument to reprocess the dataset."

        # we want to make sure the edge ordering doesn't change once converted to PyG, so store it
        for i, e in enumerate(self.nx_graph.edges):
            self.nx_graph.edges[e]["nx_edge_index"] = i

        # this operation is slow so we only want to do it once
        torch_graph = torch_geometric.utils.from_networkx(self.nx_graph, group_edge_attrs=self.feat_attrs)

        samples = []
        for path in self.paths:
            sample = torch_graph.clone()

            chosen_edges_mask = torch.as_tensor([e in path for e in self.source_graph.edges])  # convert path to mask
            sample.choice = chosen_edges_mask[torch_graph.nx_edge_index]  # reindex according to PyG edge order
            samples.append(sample)

        torch.save(torch_graph, self.processed_paths[0])
        self.save(samples, self.processed_paths[1])

    def plot_dataset(self):
        fig, axes = plt.subplots(1, 2)
        node_pos = nx.get_node_attributes(self.source_graph, "pos")
        node_labels = {n: n for n in self.source_graph.nodes}

        # first ax is value and util
        for e in self.source_graph.edges:
            self.source_graph.edges[e]["value"] = self.state_graph.nodes[e]["value"]
            self.source_graph.edges[e]["unit_flow"] = self.state_graph.nodes[e]["unit_flow"]

        nx.draw(self.source_graph, node_pos, labels=node_labels, ax=axes[0], edgelist=[])
        draw_networkx_edge_attr(self.source_graph, node_pos, ax=axes[0], edge_attr="value", cmap="viridis")

        # second ax is edge frequency
        nx.draw(self.source_graph, node_pos, labels=node_labels, ax=axes[1], edgelist=[])
        draw_networkx_edge_attr(self.source_graph, node_pos, ax=axes[1], edge_attr="unit_flow", cmap="viridis")

        return fig, axes
