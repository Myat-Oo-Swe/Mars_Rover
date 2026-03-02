from env.mars_rover_env import MarsRoverEnv
from stable_baselines3 import PPO
import numpy as np

# -------------------------
# FUNCTION TO PRINT MAP
# -------------------------
def print_map(env, path=None, title="MAP"):

    print(f"\n===== {title} =====\n")

    for i in range(env.size):
        row = ""
        for j in range(env.size):

            if (i, j) == (0, 0):
                row += " S "
            elif (i, j) == (env.goal[0], env.goal[1]):
                row += " G "
            elif path is not None and (i, j) in path:
                row += " * "
            elif env.terrain[i, j] == 1:
                row += " X "
            else:
                row += " . "

        print(row)


# -------------------------
# LOAD ENV + MODEL
# -------------------------
env = MarsRoverEnv()
model = PPO.load("mars_rover_ppo")

obs, _ = env.reset()

# Show terrain BEFORE movement
print_map(env, title="INITIAL MARS TERRAIN")

total_reward = 0
path = []
done = False

# -------------------------
# RUN EPISODE
# -------------------------
while not done:

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward

    x = int(obs[0] * (env.size - 1))
    y = int(obs[1] * (env.size - 1))

    path.append((x, y))

    done = terminated or truncated


# Show terrain AFTER movement
print_map(env, path=path, title="FINAL MAP WITH ROVER PATH")


# -------------------------
# MISSION SUMMARY
# -------------------------
print("\n===== MISSION SUMMARY =====")
print(f"Success: {np.array_equal(env.pos, env.goal)}")
print(f"Total Steps: {env.steps}")
print(f"Total Reward: {round(total_reward,2)}")
print(f"Energy Remaining: {round(env.energy,2)}")
print(f"Hazard Hits: {env.hazard_hits}")
print("===========================")