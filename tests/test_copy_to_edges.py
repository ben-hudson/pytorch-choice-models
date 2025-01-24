import networkx as nx
import pytest
import torch

from layers.copy_to_edges import CopyToEdges
from torch_geometric.utils import from_networkx


@pytest.fixture
def path_graph():
    graph = nx.path_graph(5, create_using=nx.DiGraph)
    torch_graph = from_networkx(graph)
    torch_graph.node_index = torch.arange(torch_graph.num_nodes).unsqueeze(1)
    return torch_graph


@pytest.fixture
def star_graph():
    # nx.star_graph does not support directed graphs, so we have to make our own
    n_nodes = 5
    graph = nx.DiGraph()
    graph.add_node(0)  # the center node
    for n in range(1, n_nodes):
        graph.add_edge(0, n)

    torch_graph = from_networkx(graph)
    torch_graph.node_index = torch.arange(torch_graph.num_nodes).unsqueeze(1)
    return torch_graph


def test_path_in(path_graph):
    layer = CopyToEdges(dir="to_incoming")
    next_nodes = layer(path_graph.node_index, path_graph.edge_index)
    # if we copy the node index to the incoming edges, each edge will hold the node index of the next node
    next_nodes_true = path_graph.node_index[1:]
    assert (next_nodes == next_nodes_true).all(), f"expected {next_nodes_true.squeeze()} but got {next_nodes.squeeze()}"


def test_path_out(path_graph):
    layer = CopyToEdges(dir="to_outgoing")
    prev_nodes = layer(path_graph.node_index, path_graph.edge_index)
    # if we copy the node index to the outgoing edges, each edge will hold the node index of the prev node
    prev_nodes_true = path_graph.node_index[:-1]
    assert (prev_nodes == prev_nodes_true).all(), f"expected {prev_nodes_true.squeeze()} but got {prev_nodes.squeeze()}"


def test_star_in(star_graph):
    layer = CopyToEdges(dir="to_incoming")
    outer_nodes = layer(star_graph.node_index, star_graph.edge_index)
    # the center node is 0, so we expect the edges to hold the node index of the outer nodes (1-n)
    outer_nodes_true = torch.arange(1, star_graph.num_nodes).unsqueeze(1)
    assert (
        outer_nodes == outer_nodes_true
    ).all(), f"expected {outer_nodes_true.squeeze()} but got {outer_nodes.squeeze()}"


def test_star_out(star_graph):
    layer = CopyToEdges(dir="to_outgoing")
    outer_nodes = layer(star_graph.node_index, star_graph.edge_index)
    # the center node is 0, so we expect the all edges to hold 0
    outer_nodes_true = torch.zeros(star_graph.num_nodes - 1, 1)
    assert (
        outer_nodes == outer_nodes_true
    ).all(), f"expected {outer_nodes_true.squeeze()} but got {outer_nodes.squeeze()}"
