import gymnasium as gym
import RoutingGraphEnv  # Registers the environment with Gymnasium.


env = gym.make(
    "RoutingGraphEnv/RoutingGraph-v0",
    place_name="Tan Binh District, Ho Chi Minh City, Vietnam",
    num_nodes=10,
)

observation, info = env.reset(seed=42)

print("Observation:")
print(observation)

print("Info:")
print(info)

terminated = False
truncated = False

while not terminated and not truncated:
    action = env.action_space.sample(mask=env.unwrapped.action_masks())

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        {
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "observation": observation,
            "info": info,
        }
    )

env.close()
