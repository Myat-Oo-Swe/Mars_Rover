from env.old_mars_rover_env import MarsRoverEnv
from stable_baselines3 import DQN

env = MarsRoverEnv()
model = DQN.load("mars_rover_dqn")

obs, _ = env.reset()

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    print("Pos:", obs[:2], "Reward:", reward)

    if done:
        print("Mission Finished")
        break
