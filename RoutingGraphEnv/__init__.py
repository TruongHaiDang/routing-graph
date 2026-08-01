from gymnasium.envs.registration import register

register(
    id="RoutingGraphEnv/RoutingGraph-v0",
    entry_point="RoutingGraphEnv.envs:RoutingGraphEnv",
)
