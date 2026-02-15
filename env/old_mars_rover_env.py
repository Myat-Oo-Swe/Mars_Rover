import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MarsRoverEnv(gym.Env):

    def __init__(self):
        super(MarsRoverEnv, self).__init__()

        # Map size (10x10 grid)
        self.size = 10

        # Actions: 0=forward, 1=left, 2=right, 3=idle
        self.action_space = spaces.Discrete(4)

        # Observation: x, y, direction, energy, distance_to_goal
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(5,), dtype=np.float32
        )

        self.reset()

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start position
        self.pos = np.array([0, 0], dtype=np.int32)
        self.direction = 1  # 0=N, 1=E, 2=S, 3=W

        # Goal
        self.goal = np.array([9, 9], dtype=np.int32)

        # Energy
        self.energy = 100

        # Step counter
        self.steps = 0

        return self._get_obs(), {}

    # ---------------- OBSERVATION ----------------
    def _get_obs(self):
        dist = np.linalg.norm(self.goal - self.pos)

        return np.array([
            self.pos[0],
            self.pos[1],
            self.direction,
            self.energy,
            dist
        ], dtype=np.float32)

    # ---------------- STEP FUNCTION ----------------
    def step(self, action):
        self.steps += 1
        done = False
        reward = -0.5   # time penalty (encourage faster arrival)

        # -------- DISTANCE BEFORE ACTION --------
        old_dist = np.linalg.norm(self.goal - self.pos)

        # -------- ACTIONS --------
        # Turn left
        if action == 1:
            self.direction = (self.direction - 1) % 4
            self.energy -= 0.5

        # Turn right
        elif action == 2:
            self.direction = (self.direction + 1) % 4
            self.energy -= 0.5

        # Move forward
        elif action == 0:
            move = {
                0: np.array([-1, 0]),  # North
                1: np.array([0, 1]),   # East
                2: np.array([1, 0]),   # South
                3: np.array([0, -1])   # West
            }[self.direction]

            new_pos = self.pos + move

            # Boundary check
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                self.pos = new_pos
                self.energy -= 1
            else:
                # hitting boundary = bad
                reward -= 3

        # Idle (small penalty)
        elif action == 3:
            self.energy -= 0.2
            reward -= 1

        # -------- DISTANCE AFTER ACTION --------
        new_dist = np.linalg.norm(self.goal - self.pos)

        # Reward for moving closer to goal
        reward += (old_dist - new_dist) * 2.0

        # -------- GOAL REACHED --------
        if np.array_equal(self.pos, self.goal):
            reward += 200
            done = True

        # -------- ENERGY DEPLETED --------
        if self.energy <= 0:
            reward -= 80
            done = True

        # -------- STEP LIMIT --------
        if self.steps >= 200:
            done = True

        return self._get_obs(), reward, done, False, {}
