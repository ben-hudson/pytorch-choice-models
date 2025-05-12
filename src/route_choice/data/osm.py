import osmnx as ox

from typing import Tuple


def load_osm_network(
    center_point: Tuple[float, float],
    bbox_meters: int = 1000,
    simplify: bool = True,
    consolidate_intersections: bool = True,
):
    G = ox.graph.graph_from_point(
        center_point, dist=bbox_meters, dist_type="bbox", network_type="drive", simplify=simplify
    )
    if consolidate_intersections:
        G_proj = ox.projection.project_graph(G)
        G_simpl = ox.simplification.consolidate_intersections(
            G_proj, rebuild_graph=True, tolerance=10.0, dead_ends=False
        )
        G = ox.projection.project_graph(G_simpl, to_latlong=True)

    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    G = ox.bearing.add_edge_bearings(G)

    return G
