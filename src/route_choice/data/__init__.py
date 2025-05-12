from .osm import *
from .tutorial import *

from typing import Any, Callable, List


# clockwise rotation is positive
def get_turn_angle(in_bearing: float, out_bearing: float):
    turn_angle = out_bearing - in_bearing
    if turn_angle > 180:
        turn_angle -= 360
    elif turn_angle < -180:
        turn_angle += 360
    return turn_angle


def get_line_graph(
    network_graph: nx.MultiDiGraph,
    origs: List[Any],
    dests: List[Any],
    get_edge_feats: Callable = None,
    get_edge_to_edge_feats: Callable = None,
) -> nx.DiGraph:
    choice_graph = nx.line_graph(network_graph, create_using=nx.DiGraph)

    for u, v in choice_graph.edges:
        if get_edge_feats is not None:
            # copy edge feats from the origin edge (e.g. travel time)
            for name, feat in get_edge_feats(network_graph, u).items():
                choice_graph.edges[u, v][name] = feat
        if get_edge_to_edge_feats is not None:
            # get edge-to-edge features (e.g. turn angle)
            # these can also include node features from the shared node
            for name, feat in get_edge_to_edge_feats(network_graph, u, v).items():
                choice_graph.edges[u, v][name] = feat

    choice_graph.add_nodes_from(origs + dests)
    for orig in origs:
        for u in choice_graph.nodes:
            # skip the node we just added
            if u in origs or u in dests:
                continue
            # if there is an edge that leaves the origin node
            elif orig == u[0]:
                # connect that edge to the new dummy edge
                choice_graph.add_edge(orig, u)

    for dest in dests:
        for u in choice_graph.nodes:
            # skip the node we just added
            if u in origs or u in dests:
                continue
            # if there is an edge that enters the destination node
            elif dest == u[1]:
                # connect that edge to the new dummy edge
                choice_graph.add_edge(u, dest)

    return choice_graph
