from env.old_mars_rover_env import MarsRoverEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Create environment
env = MarsRoverEnv()

# Check if environment follows Gym rules
check_env(env)

# Create model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    verbose=1
)

# Train
model.learn(total_timesteps=150000)

# Save
model.save("mars_rover_dqn")
