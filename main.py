from env.mars_rover_env import MarsRoverEnv
import random

env = MarsRoverEnv()

obs, _ = env.reset()

for _ in range(50):
    action = random.randint(0, 3)
    obs, reward, done, _, _ = env.step(action)
    print("Pos:", obs[:2], "Reward:", reward)

    if done:
        print("Episode finished")
        break
