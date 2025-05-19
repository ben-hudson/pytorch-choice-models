import networkx as nx

from matplotlib import colors, cm
from typing import Any, Dict, Mapping

from .dataset import RouteChoiceDataset


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
    G: nx.MultiDiGraph, pos: Mapping, edge_attr: Any = None, default_color="k", bend: float = 0.1, **kwargs: Dict
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


class ToyRouteChoiceDataset(RouteChoiceDataset):
    def plot_dataset(self):
        # only import matplotlib when we actually want to do some plotting
        from matplotlib import pyplot as plt

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
