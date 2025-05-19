import networkx as nx
import numpy as np
import pathlib
import pickle
import torch
import torch_geometric.data
import torch_geometric.utils

from typing import Any, Callable, List

from .utils import SamplingException, compute_values_probs_flows, get_state_graph, normalize_attrs, sample_path


class RouteChoiceDataset(torch_geometric.data.InMemoryDataset):
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
        required_kwargs = ["feat_attrs", "feat_fn", "util_fn", "util_scale", "n_samples"]
        assert all(
            getattr(self, kwarg, None) is not None for kwarg in required_kwargs
        ), f"No file(s) found at {self.processed_paths}. Pass {required_kwargs} kwargs to rebuild the dataset."

        source_graph = pickle.load(open(self.raw_paths[0], "rb"))
        orig = source_graph.graph["orig"]
        dest = source_graph.graph["dest"]

        # set up state graph
        state_graph = get_state_graph(source_graph, orig, dest, self.feat_fn, self.util_fn, self.util_scale)

        M, state_list = nx.attr_matrix(state_graph, "M")
        V, P, F = compute_values_probs_flows(M, state_list.index(orig), state_list.index(dest))

        # reassemble into node and edge-wise dicts
        for i, state_i in enumerate(state_list):
            state_graph.nodes[state_i]["value"] = V[i].item()
            state_graph.nodes[state_i]["unit_flow"] = F[i].item()

            for j, state_j in enumerate(state_list):
                if state_graph.has_edge(state_i, state_j):
                    state_graph.edges[state_i, state_j]["trans_prob"] = P[i, j].item()

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
        torch_graph = torch_geometric.utils.from_networkx(state_graph, group_edge_attrs=self.feat_attrs)

        # sample paths
        data_list = []
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_samples):
            try:
                path_edges = sample_path(state_graph, orig, dest, rng, prob_key="trans_prob")
                edge_mask = torch.as_tensor([e in path_edges for e in state_graph.edges])  # convert path to mask

                data = torch_graph.clone()
                data.path_edges = edge_mask[data.nx_edge_idx]  # reindex according to PyG node order
                data_list.append(data)

            except SamplingException:
                pass

        assert len(data_list) > 0, "Unable to sample any paths."
        self.save(data_list, self.processed_paths[1])

    def plot_dataset(self):
        raise NotImplementedError("You need to override this method.")
