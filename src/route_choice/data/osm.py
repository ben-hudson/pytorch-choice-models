import osmnx as ox

from typing import Tuple

from .dataset import RouteChoiceDataset


def load_osm_network(
    center_point: Tuple[float, float],
    bbox_meters: int = 1000,
    simplify: bool = True,
    consolidate_intersections: bool = True,
    consolidate_meters: float = 10.0,
):
    G = ox.graph.graph_from_point(
        center_point, dist=bbox_meters, dist_type="bbox", network_type="drive", simplify=simplify
    )
    if consolidate_intersections:
        G_proj = ox.projection.project_graph(G)
        G_simpl = ox.simplification.consolidate_intersections(
            G_proj, rebuild_graph=True, tolerance=consolidate_meters, dead_ends=False
        )
        G = ox.projection.project_graph(G_simpl, to_latlong=True)

    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    G = ox.bearing.add_edge_bearings(G)

    return G


# clockwise rotation is positive
def get_turn_angle(in_bearing: float, out_bearing: float):
    turn_angle = out_bearing - in_bearing
    if turn_angle > 180:
        turn_angle -= 360
    elif turn_angle < -180:
        turn_angle += 360
    return turn_angle


class OSMRouteChoiceDataset(RouteChoiceDataset):
    pass
