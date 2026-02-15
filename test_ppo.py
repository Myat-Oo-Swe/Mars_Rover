from env.mars_rover_env import MarsRoverEnv
from stable_baselines3 import PPO

env = MarsRoverEnv()
model = PPO.load("mars_rover_ppo")

obs, _ = env.reset()

for _ in range(300):

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    # convert normalized obs back to grid
    x = int(obs[0] * 9)
    y = int(obs[1] * 9)

    print("Position:", (x, y), "Reward:", round(reward, 3))

    if terminated or truncated:
        print("Mission Finished")
        break
