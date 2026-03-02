from env.mars_rover_env import MarsRoverEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# Create env
env = MarsRoverEnv()
env = Monitor(env)

check_env(env)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("mars_rover_ppo")
