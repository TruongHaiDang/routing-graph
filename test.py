import gymnasium as gym
import RoutingGraphEnv  # Registers the environment with Gymnasium.
from time import sleep
from pprint import pprint


env = gym.make(
    "RoutingGraphEnv/RoutingGraph-v0",
    place_name="Tan Binh District, Ho Chi Minh City, Vietnam",
    num_nodes=10,
    render_mode="human"
)

observation, info = env.reset(seed=42)

terminated = False
truncated = False

while not terminated and not truncated:
    action = env.action_space.sample(mask=env.unwrapped.action_masks())

    observation, reward, terminated, truncated, info = env.step(action)
    print("observation")
    pprint(observation)
    print("reward")
    pprint(reward)
    print("terminated")
    pprint(terminated)
    print("truncated")
    pprint(truncated)
    print("info")
    pprint(info)

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
    print("-"*100)
    sleep(1)

env.close()
