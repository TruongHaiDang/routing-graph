from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from networkx import MultiDiGraph

from RoutingGraphEnv.envs.openstreetmap_services import OpenStreetMapService


class RoutingGraphEnv(gym.Env):
    """Gymnasium environment for finding a short round trip through road nodes."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        num_nodes: int = 5,
        place_name: str = "Tan Binh District, Ho Chi Minh City, Vietnam",
        graph: MultiDiGraph | None = None,
        nodes: list[dict[str, Any]] | None = None,
    ) -> None:
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if (graph is None) != (nodes is None):
            raise ValueError("graph and nodes must be provided together")

        self.render_mode = render_mode
        self.num_nodes = num_nodes
        self.place_name = place_name
        self.window_size = 512
        self.window = None
        self.clock = None

        if graph is None:
            graph, nodes = OpenStreetMapService.generate_random_points_on_roads(
                place_name=place_name,
                count=num_nodes,
                seed=42,
            )
            print("Graph", graph)
            print("Node", nodes)
        elif len(nodes) != num_nodes:
            raise ValueError("len(nodes) must equal num_nodes")

        self.graph = graph
        self.nodes = nodes
        self.distance_matrix = OpenStreetMapService.distance_matrix(graph, nodes)
        print("Distance matrix", self.distance_matrix)
        if not np.isfinite(self.distance_matrix).all():
            raise ValueError("Selected road nodes are not mutually reachable")

        self.observation_space = spaces.Dict(
            {
                "visit_order": spaces.Box(
                    low=-1,
                    high=num_nodes - 1,
                    shape=(num_nodes,),
                    dtype=np.int32,
                ),
                "visited": spaces.MultiBinary(num_nodes),
                "current_node": spaces.Discrete(num_nodes),
            }
        )
        self.action_space = spaces.Discrete(num_nodes)

        self._visit_order = np.full(num_nodes, -1, dtype=np.int32)
        self._visited = np.zeros(num_nodes, dtype=np.int8)
        self._current_node = 0
        self._step_count = 0
        self._total_distance = 0.0
        self._render_positions: np.ndarray | None = None

    def _get_obs(self) -> dict[str, np.ndarray | int]:
        return {
            "visit_order": self._visit_order.copy(),
            "visited": self._visited.copy(),
            "current_node": self._current_node,
        }

    def _get_info(self) -> dict[str, float]:
        return {"total_distance": self._total_distance}

    def action_masks(self) -> np.ndarray:
        return np.logical_not(self._visited).astype(dtype=np.int8)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, float]]:
        super().reset(seed=seed)
        start_node = 0 if options is None else int(options.get("start_node", 0))
        if not self.action_space.contains(start_node):
            raise ValueError("start_node is outside the action space")

        self._visit_order.fill(-1)
        self._visited.fill(0)
        self._visit_order[0] = start_node
        self._visited[start_node] = 1
        self._current_node = start_node
        self._step_count = 1
        self._total_distance = 0.0

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, float]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        action = int(action)
        if self._visited[action]:
            return self._get_obs(), -1.0, False, False, {
                **self._get_info(),
                "invalid_action": True,
            }

        distance = float(self.distance_matrix[self._current_node, action])
        self._total_distance += distance
        self._current_node = action
        self._visit_order[self._step_count] = action
        self._visited[action] = 1
        self._step_count += 1

        terminated = self._step_count == self.num_nodes
        if terminated:
            return_distance = float(self.distance_matrix[action, self._visit_order[0]])
            distance += return_distance
            self._total_distance += return_distance

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), -distance, terminated, False, self._get_info()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray | None:
        import pygame

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill("white")
        if self._render_positions is None:
            coordinates = np.asarray(
                [[node["longitude"], node["latitude"]] for node in self.nodes],
                dtype=np.float64,
            )
            minimum = coordinates.min(axis=0)
            span = np.maximum(coordinates.max(axis=0) - minimum, 1e-12)
            self._render_positions = 30 + (coordinates - minimum) / span * (
                self.window_size - 60
            )
            self._render_positions[:, 1] = (
                self.window_size - self._render_positions[:, 1]
            )
        positions = self._render_positions

        visited = self._visit_order[: self._step_count]
        for source, target in zip(visited, visited[1:]):
            pygame.draw.line(canvas, "royalblue", positions[source], positions[target], 3)
        for index, position in enumerate(positions):
            color = "royalblue" if self._visited[index] else "gray"
            pygame.draw.circle(canvas, color, position, 8)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
