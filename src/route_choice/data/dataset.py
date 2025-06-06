import math
import networkx as nx
import numpy as np
import pathlib
import pickle
import torch
import torch_geometric.data
import torch_geometric.utils

from typing import Any, Callable, List

from .utils import compute_values_probs_flows, get_state_graph, normalize_attrs


class SamplingException(Exception):
    pass


class RouteChoiceDataset(torch_geometric.data.InMemoryDataset):
    def __init__(
        self,
        root: pathlib.Path,
        graph: nx.MultiDiGraph = None,
        node_feat_attrs: List[str] = None,
        node_feat_fn: Callable = None,
        edge_feat_attrs: List[str] = None,
        edge_feat_fn: Callable = None,
        util_fn: Callable = None,
        source: Any = None,
        target: Any = None,
        n_samples: int = None,
        seed: int = None,
        **kwargs,
    ):
        self.source_graph = graph
        self.node_feat_attrs = node_feat_attrs
        self.node_feat_fn = node_feat_fn
        self.edge_feat_attrs = edge_feat_attrs
        self.edge_feat_fn = edge_feat_fn
        self.util_fn = util_fn
        self.orig = source
        self.dest = target
        self.n_samples = n_samples
        self.seed = seed

        # this calls download and process if necessary
        # after calling super, the files are guaranteed to exist
        super().__init__(root, **kwargs)
        self.source_graph = pickle.load(open(self.raw_paths[0], "rb"))
        self.state_graph = pickle.load(open(self.processed_paths[0], "rb"))
        self.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["source_graph.pkl"]

    @property
    def processed_file_names(self):
        return ["state_graph.pkl", "sampled_paths.pt"]

    def download(self):
        required_kwargs = ["source_graph", "orig", "dest"]
        assert all(
            getattr(self, kwarg, None) is not None for kwarg in required_kwargs
        ), f"No file found at {self.raw_paths}. Pass {required_kwargs} kwargs to rebuild the dataset."

        assert self.orig in self.source_graph.nodes, f"{self.orig} not in source_graph.nodes"
        assert self.dest in self.source_graph.nodes, f"{self.dest} not in source_graph.nodes"

        self.source_graph.graph["orig"] = self.orig
        self.source_graph.graph["dest"] = self.dest
        pickle.dump(self.source_graph, open(self.raw_paths[0], "wb"))

    def process(self):
        required_kwargs = ["util_fn", "n_samples"]
        assert all(
            getattr(self, kwarg, None) is not None for kwarg in required_kwargs
        ), f"No file(s) found at {self.processed_paths}. Pass {required_kwargs} kwargs to rebuild the dataset."

        source_graph = pickle.load(open(self.raw_paths[0], "rb"))
        orig = source_graph.graph["orig"]
        dest = source_graph.graph["dest"]

        # set up state graph
        state_graph = get_state_graph(source_graph, orig, dest, self.node_feat_fn, self.edge_feat_fn)
        pickle.dump(state_graph, open(self.processed_paths[0], "wb"))

        # prepare state_graph to be transformed to PyG
        normalize_attrs(state_graph, on="nodes", attrs_to_keep="all")
        normalize_attrs(state_graph, on="edges", attrs_to_keep="all")
        # add networkx indices so we can reindex from PyG
        # PyG automatically REINDEXES any attribute ending in "index" when batching, so we have to call this "idx"
        for i, k in enumerate(state_graph.nodes):
            state_graph.nodes[k]["nx_node_idx"] = i
        for i, (k, a) in enumerate(state_graph.edges):
            state_graph.edges[k, a]["nx_edge_idx"] = i
        torch_graph = torch_geometric.utils.from_networkx(
            state_graph, group_node_attrs=self.node_feat_attrs, group_edge_attrs=self.edge_feat_attrs
        )

        # sample paths
        rng = np.random.default_rng(self.seed)

        paths_list = self.sample_paths(state_graph, orig, dest, rng, prob_key="trans_prob")
        assert len(paths_list) > 0, "Unable to sample any paths."

        data_list = []
        for path in paths_list:
            edge_mask = torch.as_tensor([e in path for e in state_graph.edges])  # convert path to mask

            data = torch_graph.clone()
            data.path = edge_mask[data.nx_edge_idx]  # reindex according to PyG node order
            data_list.append(data)

        self.save(data_list, self.processed_paths[1])

    def sample_paths(self, state_graph: nx.DiGraph, orig: Any, dest: Any, rng: np.random.Generator, **kwargs):
        util_scale = kwargs.pop("util_scale", 1)

        for k, a, feats in state_graph.edges(data=True):
            util = self.util_fn(feats)
            state_graph.edges[k, a]["util"] = util
            state_graph.edges[k, a]["M"] = math.exp(1 / util_scale * util)

        M, state_list = nx.attr_matrix(state_graph, "M")
        V, P, F = compute_values_probs_flows(M, state_list.index(orig), state_list.index(dest))

        # reassemble into node and edge-wise dicts
        for i, state_i in enumerate(state_list):
            state_graph.nodes[state_i]["value"] = V[i].item()
            state_graph.nodes[state_i]["unit_flow"] = F[i].item()

            for j, state_j in enumerate(state_list):
                if state_graph.has_edge(state_i, state_j):
                    state_graph.edges[state_i, state_j]["trans_prob"] = P[i, j].item()

        paths = []
        for _ in range(self.n_samples):
            try:
                path = self.sample_path(state_graph, orig, dest, rng, prob_key="trans_prob")
                paths.append(path)
            except SamplingException:
                pass

        return paths

    def sample_path(
        self,
        state_graph: nx.DiGraph,
        orig: Any,
        dest: Any,
        rng: np.random.Generator,
        prob_key: str = "prob",
        max_length: int = 1000,
    ):
        k = orig
        path = []
        while k != dest and len(state_graph.out_edges(k)) > 0 and len(path) < max_length:
            transitions = [t for t in state_graph.out_edges(k, data=prob_key, default=0)]
            _, actions, probs = zip(*transitions)
            sample = rng.multinomial(1, probs)
            sampled_action = actions[np.argmax(sample)]
            path.append((k, sampled_action))

            k = sampled_action

        if k != dest:
            raise SamplingException(f"Unable to sample path from {orig} to {dest} in less than {max_length} steps.")

        return path

    def plot_dataset(self):
        raise NotImplementedError("You need to override this method.")
