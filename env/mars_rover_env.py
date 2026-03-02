import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarsRoverEnv(gym.Env):

    def __init__(self, size=10, hazard_prob=0.2, view_size=5):
        super(MarsRoverEnv, self).__init__()

        self.size = size
        self.hazard_prob = hazard_prob
        self.view_size = view_size  # must be odd number (3,5,7...)

        self.max_steps = 200
        self.max_energy = 100

        # 4 actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation:
        # [pos_x, pos_y] + flattened local terrain window
        obs_size = 2 + (view_size * view_size)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.reset()

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate terrain (0 = safe, 1 = hazard)
        self.terrain = np.random.choice(
            [0, 1],
            size=(self.size, self.size),
            p=[1 - self.hazard_prob, self.hazard_prob]
        )

        # Start & goal
        self.pos = np.array([0, 0])
        self.goal = np.array([self.size - 1, self.size - 1])

        # Ensure start and goal are safe
        self.terrain[0, 0] = 0
        self.terrain[self.goal[0], self.goal[1]] = 0

        self.steps = 0
        self.energy = self.max_energy
        self.hazard_hits = 0

        return self._get_obs(), {}

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------
    def step(self, action):

        self.steps += 1
        self.energy -= 1

        old_pos = self.pos.copy()

        # Move rover
        if action == 0:   # up
            self.pos[0] -= 1
        elif action == 1: # down
            self.pos[0] += 1
        elif action == 2: # left
            self.pos[1] -= 1
        elif action == 3: # right
            self.pos[1] += 1

        # Keep inside map
        self.pos = np.clip(self.pos, 0, self.size - 1)

        reward = -0.1  # small step penalty

        # Hazard penalty
        if self.terrain[self.pos[0], self.pos[1]] == 1:
            reward -= 10
            self.hazard_hits += 1

        # Distance shaping reward
        old_dist = np.linalg.norm(old_pos - self.goal)
        new_dist = np.linalg.norm(self.pos - self.goal)
        reward += (old_dist - new_dist)

        # Goal reward
        terminated = False
        if np.array_equal(self.pos, self.goal):
            reward += 200
            terminated = True

        truncated = False
        if self.steps >= self.max_steps or self.energy <= 0:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    # -------------------------------------------------
    # PARTIAL OBSERVATION
    # -------------------------------------------------
    def _get_obs(self):

        # Normalized position
        pos_norm = self.pos / (self.size - 1)

        # Get local window
        half = self.view_size // 2

        padded = np.pad(self.terrain, pad_width=half, mode='constant', constant_values=1)

        x, y = self.pos
        x_p = x + half
        y_p = y + half

        local_view = padded[
            x_p - half:x_p + half + 1,
            y_p - half:y_p + half + 1
        ]

        local_view = local_view.flatten()

        obs = np.concatenate([pos_norm, local_view]).astype(np.float32)

        return obs