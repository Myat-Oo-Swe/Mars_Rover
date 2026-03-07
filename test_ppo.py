from env.mars_rover_env import MarsRoverEnv
from stable_baselines3 import PPO
import numpy as np


# -------------------------
# PRINT TERRAIN MAP
# -------------------------
def print_map(env, path=None, title="MAP"):

    print(f"\n===== {title} =====\n")

    for i in range(env.size):

        row = ""

        for j in range(env.size):

            if (i,j) == tuple(env.start):
                row += " S "

            elif (i,j) == tuple(env.goal):
                row += " G "

            elif (i,j) == tuple(env.pos):
                row += " R "

            elif path is not None and (i,j) in path:
                row += " * "

            else:

                t = env.terrain[i,j]

                if t == 0:
                    row += " . "

                elif t == 1:
                    row += " D "

                elif t == 2:
                    row += " K "   # rock (changed from R to avoid confusion)

                elif t == 3:
                    row += " L "

                elif t == 4:
                    row += " X "

        print(row)


# -------------------------
# LOAD ENV + MODEL
# -------------------------
env = MarsRoverEnv()

model = PPO.load("mars_rover_ppo")

obs,_ = env.reset(options={
    "start_pos":(2,2),
    "goal_pos":(18,18)
})

# store start location
env.start = tuple(env.pos)

print_map(env,title="INITIAL MARS TERRAIN")


total_reward = 0
path = []
done = False


# -------------------------
# RUN EPISODE
# -------------------------
while not done:

    action,_ = model.predict(obs,deterministic=True)

    obs,reward,terminated,truncated,_ = env.step(action)

    total_reward += reward

    # use true rover position
    path.append(tuple(env.pos))

    done = terminated or truncated


print_map(env,path=path,title="FINAL MAP WITH ROVER PATH")


# -------------------------
# MISSION SUMMARY
# -------------------------
print("\n===== MISSION SUMMARY =====")

print(f"Success: {np.array_equal(env.pos,env.goal)}")
print(f"Total Steps: {env.steps}")
print(f"Total Reward: {round(total_reward,2)}")
print(f"Energy Remaining: {round(env.energy,2)}")
print(f"Hazard Hits: {env.hazard_hits}")

print("===========================")