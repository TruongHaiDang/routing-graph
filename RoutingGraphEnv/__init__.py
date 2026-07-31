from gymnasium.envs.registration import register

register(
    id="RoutingGraphEnv/GridWorld-v0",
    entry_point="RoutingGraphEnv.envs:GridWorldEnv",
)
