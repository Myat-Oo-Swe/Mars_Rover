from env.mars_rover_env import MarsRoverEnv
from stable_baselines3 import PPO
import numpy as np
import random

# -------------------------
# SETTINGS
# -------------------------

NUM_MISSIONS = 1000

env = MarsRoverEnv()
model = PPO.load("mars_rover_ppo")

success_count = 0
total_steps = []
energy_used = []
hazard_hits = []


# -------------------------
# RUN MISSIONS
# -------------------------

for mission in range(NUM_MISSIONS):

    # random start/goal
    start = (
        random.randint(0, env.size-1),
        random.randint(0, env.size-1)
    )

    goal = (
        random.randint(0, env.size-1),
        random.randint(0, env.size-1)
    )

    obs, _ = env.reset(options={
        "start_pos": start,
        "goal_pos": goal
    })

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    success = np.array_equal(env.pos, env.goal)

    if success:
        success_count += 1

    total_steps.append(env.steps)
    energy_used.append(env.max_energy - env.energy)
    hazard_hits.append(env.hazard_hits)

    print(f"Mission {mission+1}: success={success} steps={env.steps}")


# -------------------------
# RESULTS
# -------------------------

print("\n===== BENCHMARK RESULTS =====")

print(f"Total Missions: {NUM_MISSIONS}")
print(f"Success Rate: {round(success_count/NUM_MISSIONS*100,2)} %")

print(f"Avg Steps: {round(np.mean(total_steps),2)}")

print(f"Avg Energy Used: {round(np.mean(energy_used),2)}")

print(f"Avg Hazard Hits: {round(np.mean(hazard_hits),2)}")

print("=============================")