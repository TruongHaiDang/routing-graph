from __future__ import annotations

import random

import networkx as nx
import numpy as np
import osmnx as ox
from networkx import MultiDiGraph


class OpenStreetMapService:
    """Load road networks and select reproducible routing nodes."""

    @staticmethod
    def load_drive_graph(
        place_name: str,
        simplify: bool = True,
    ) -> MultiDiGraph:
        graph = ox.graph_from_place(
            place_name,
            network_type="drive",
            simplify=simplify,
        )

        return ox.truncate.largest_component(
            graph,
            strongly=True,
        )

    @staticmethod
    def generate_random_road_nodes(
        graph: MultiDiGraph,
        count: int,
        seed: int | None = None,
    ) -> list[dict]:
        if count < 2:
            raise ValueError("count must be at least 2")

        node_ids = list(graph.nodes)
        if count > len(node_ids):
            raise ValueError(
                f"Requested {count} nodes but graph only has {len(node_ids)} nodes."
            )

        selected = random.Random(seed).sample(node_ids, count)
        return [
            {
                "id": index,
                "osm_node_id": node_id,
                "latitude": graph.nodes[node_id]["y"],
                "longitude": graph.nodes[node_id]["x"],
            }
            for index, node_id in enumerate(selected)
        ]

    @classmethod
    def generate_random_points_on_roads(
        cls,
        place_name: str,
        count: int,
        seed: int | None = None,
    ) -> tuple[MultiDiGraph, list[dict]]:
        graph = cls.load_drive_graph(place_name)
        return graph, cls.generate_random_road_nodes(graph, count, seed)

    @staticmethod
    def distance_matrix(
        graph: MultiDiGraph,
        nodes: list[dict],
    ) -> np.ndarray:
        """Compute selected-node distances with one Dijkstra run per source."""
        node_ids = [node["osm_node_id"] for node in nodes]
        matrix = np.full((len(node_ids), len(node_ids)), np.inf, dtype=np.float32)
        np.fill_diagonal(matrix, 0.0)

        for source_index, source_id in enumerate(node_ids):
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                source_id,
                weight="length",
            )
            for target_index, target_id in enumerate(node_ids):
                if source_index != target_index:
                    matrix[source_index, target_index] = lengths.get(
                        target_id, float("inf")
                    )

        return matrix
