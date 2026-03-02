import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarsRoverEnv(gym.Env):

    def __init__(self):
        super(MarsRoverEnv, self).__init__()

        # ----- MAP -----
        self.size = 10

        # Actions: 0=forward, 1=left, 2=right, 3=idle
        self.action_space = spaces.Discrete(4)

        # IMPORTANT: normalize observations (PPO learns MUCH better)
        # All values scaled to 0..1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = 200
        self.reset()

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = np.array([0, 0], dtype=np.int32)
        self.direction = 1  # 0=N, 1=E, 2=S, 3=W
        self.goal = np.array([9, 9], dtype=np.int32)

        self.energy = 100.0
        self.steps = 0

        return self._get_obs(), {}

    # -------------------------------------------------
    # OBSERVATION (NORMALIZED)
    # -------------------------------------------------
    def _get_obs(self):
        dist = np.linalg.norm(self.goal - self.pos)

        return np.array([
            self.pos[0] / (self.size - 1),
            self.pos[1] / (self.size - 1),
            self.direction / 3.0,
            self.energy / 100.0,
            dist / np.sqrt(2*(self.size-1)**2)
        ], dtype=np.float32)

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------
    def step(self, action):

        self.steps += 1

        terminated = False
        truncated = False

        # small time penalty
        reward = -0.05

        old_dist = np.linalg.norm(self.goal - self.pos)

        # ----- TURN LEFT -----
        if action == 1:
            self.direction = (self.direction - 1) % 4
            self.energy -= 0.3

        # ----- TURN RIGHT -----
        elif action == 2:
            self.direction = (self.direction + 1) % 4
            self.energy -= 0.3

        # ----- MOVE FORWARD -----
        elif action == 0:

            move = {
                0: np.array([-1, 0]),   # North
                1: np.array([0, 1]),    # East
                2: np.array([1, 0]),    # South
                3: np.array([0, -1])    # West
            }[self.direction]

            new_pos = self.pos + move

            # boundary check
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                self.pos = new_pos
                self.energy -= 1.0
            else:
                reward -= 1.0  # wall hit penalty

        # ----- IDLE -----
        elif action == 3:
            self.energy -= 0.2
            reward -= 0.1

        # ----- PROGRESS REWARD -----
        new_dist = np.linalg.norm(self.goal - self.pos)
        reward += (old_dist - new_dist) * 2.5

        # ----- GOAL -----
        if np.array_equal(self.pos, self.goal):
            reward += 100
            terminated = True

        # ----- ENERGY -----
        if self.energy <= 0:
            reward -= 40
            terminated = True

        # ----- STEP LIMIT -----
        if self.steps >= self.max_steps:
            truncated = True

        self.energy = max(self.energy, 0)

        return self._get_obs(), reward, terminated, truncated, {}
