from __future__ import annotations

import random

import osmnx as ox
from networkx import MultiDiGraph
from shapely.geometry import Point, Polygon


class OpenStreetMapService:
    @staticmethod
    def get_place_polygon(place_name: str) -> Polygon:
        """
        Download the administrative boundary polygon of a place.

        Example:
            polygon = OpenStreetMapService.get_place_polygon(
                "Ho Chi Minh City, Vietnam"
            )
        """
        gdf = ox.geocode_to_gdf(place_name)
        return gdf.geometry.iloc[0]

    @staticmethod
    def load_drive_graph(
        place_name: str,
        simplify: bool = True,
    ) -> MultiDiGraph:
        """
        Download the drivable road network.
        """
        return ox.graph_from_place(
            place_name,
            network_type="drive",
            simplify=simplify,
        )

    @staticmethod
    def load_drive_graph_from_polygon(
        polygon: Polygon,
        simplify: bool = True,
    ) -> MultiDiGraph:
        """
        Download the drivable road network inside a polygon.
        """
        return ox.graph_from_polygon(
            polygon,
            network_type="drive",
            simplify=simplify,
        )

    @staticmethod
    def generate_random_points(
        polygon: Polygon,
        count: int,
        seed: int | None = None,
    ) -> list[dict]:
        """
        Generate random coordinates inside a polygon.
        """

        rng = random.Random(seed)

        min_x, min_y, max_x, max_y = polygon.bounds

        points = []

        while len(points) < count:
            longitude = rng.uniform(min_x, max_x)
            latitude = rng.uniform(min_y, max_y)

            point = Point(longitude, latitude)

            if polygon.contains(point):
                points.append(
                    {
                        "id": len(points),
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                )

        return points

    @staticmethod
    def generate_random_road_nodes(
        graph: MultiDiGraph,
        count: int,
        seed: int | None = None,
    ) -> list[dict]:
        """
        Randomly select nodes from the road network.
        """

        rng = random.Random(seed)

        node_ids = list(graph.nodes)

        if count > len(node_ids):
            raise ValueError(
                f"Requested {count} nodes but graph only has {len(node_ids)} nodes."
            )

        selected = rng.sample(node_ids, count)

        result = []

        for index, node_id in enumerate(selected):
            node = graph.nodes[node_id]

            result.append(
                {
                    "id": index,
                    "osm_node_id": node_id,
                    "latitude": node["y"],
                    "longitude": node["x"],
                }
            )

        return result

    @staticmethod
    def generate_random_points_on_roads(
        place_name: str,
        count: int,
        seed: int | None = None,
    ) -> tuple[MultiDiGraph, list[dict]]:
        """
        Generate random coordinates inside the city and snap them
        to the nearest road node.
        """

        polygon = OpenStreetMapService.get_place_polygon(place_name)

        graph = OpenStreetMapService.load_drive_graph_from_polygon(
            polygon
        )

        rng = random.Random(seed)

        min_x, min_y, max_x, max_y = polygon.bounds

        points = []
        used_nodes = set()

        while len(points) < count:
            longitude = rng.uniform(min_x, max_x)
            latitude = rng.uniform(min_y, max_y)

            point = Point(longitude, latitude)

            if not polygon.contains(point):
                continue

            node_id = ox.distance.nearest_nodes(
                graph,
                X=longitude,
                Y=latitude,
            )

            if node_id in used_nodes:
                continue

            used_nodes.add(node_id)

            node = graph.nodes[node_id]

            points.append(
                {
                    "id": len(points),
                    "osm_node_id": node_id,
                    "latitude": node["y"],
                    "longitude": node["x"],
                }
            )

        return graph, points

    @staticmethod
    def nearest_node(
        graph: MultiDiGraph,
        latitude: float,
        longitude: float,
    ) -> int:
        """
        Return the nearest road node.
        """
        return ox.distance.nearest_nodes(
            graph,
            X=longitude,
            Y=latitude,
        )

    @staticmethod
    def shortest_path(
        graph: MultiDiGraph,
        source_node: int,
        target_node: int,
    ) -> list[int]:
        """
        Compute the shortest path using edge length.
        """
        return ox.routing.shortest_path(
            graph,
            source_node,
            target_node,
            weight="length",
        )

    @staticmethod
    def shortest_path_length(
        graph: MultiDiGraph,
        source_node: int,
        target_node: int,
    ) -> float:
        """
        Compute shortest path length in meters.
        """

        route = ox.routing.shortest_path(
            graph,
            source_node,
            target_node,
            weight="length",
        )

        return float(
            sum(
                ox.routing.route_to_gdf(graph, route)["length"]
            )
        )