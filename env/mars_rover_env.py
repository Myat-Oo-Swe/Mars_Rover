"""
mars_rover_env.py  —  Mars Rover Navigation Environment (fully revised)
=======================================================================
Change tags
-----------
FIX-1   Explicit NUM_TERRAIN_TYPES in np_random.choice (not WALL_SENTINEL).
FIX-2   Per-axis wall detection — diagonal wall-slides no longer free.
FIX-3   TERRAIN_REWARD scaled by move distance (matches energy cost).
FIX-4   Exploration bonus decays over episode steps.
FIX-5   Goal reward proportional to remaining energy.
FIX-6   Rich info dict from step() and reset().
FIX-7   visited tracking uses a boolean numpy array (faster).
FIX-8   BFS reachability check + corridor fallback for synthetic terrain.
FIX-9   render() ASCII stub (required by Gymnasium wrappers).
FIX-10  Terrain probabilities as a named class constant.

MOLA-1  reset() accepts a "terrain" key in options so train_ppo and the
        visualizer can inject a real MOLA patch instead of the synthetic
        procedural generator.
MOLA-2  BFS / corridor fallback are SKIPPED for injected terrain — the
        caller (MolaTerrainCache) is responsible for patch quality.
        This prevents silently carving fake roads through real Mars data.

NEW-1   Exploration bonus removed — MOLA geographic curriculum already
        provides terrain diversity; the bonus was rewarding aimless
        wandering and causing 95 % of M1 failures (energy exhaustion).
        visited array is kept for the info dict only.
NEW-2   Asymmetric goal-shaping: retreating from the goal is penalised
        at 2× the weight of advancing, discouraging backtracking.
NEW-3   Energy-urgency penalty grows as energy depletes, making the agent
        intrinsically aware that burning time is costly.
NEW-4   Energy-exhaustion penalty raised from -10 to -25 so failing to
        reach the goal is unambiguously worse than a slow success.
"""

from __future__ import annotations
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarsRoverEnv(gym.Env):
    """
    Mars Rover Navigation Environment
    ==================================
    Terrain types
    -------------
    0 SAFE   1 SAND   2 ROCK   3 SLOPE   4 HAZARD

    Observation  (7 + view_size²  floats, all in [0, 1])
    -----------------------------------------------------
    pos_x  pos_y  goal_x  goal_y  dir_x  dir_y  energy  local_view…

    Actions
    -------
    0 up | 1 down | 2 left | 3 right |
    4 up-left | 5 up-right | 6 down-left | 7 down-right

    MOLA terrain injection
    ----------------------
    env.reset(options={
        "terrain":   <(size,size) int8 array with values 0-4>,
        "start_pos": (row, col),
        "goal_pos":  (row, col),
    })
    """

    metadata = {"render_modes": []}

    SAFE, SAND, ROCK, SLOPE, HAZARD = 0, 1, 2, 3, 4
    WALL_SENTINEL     = 5
    NUM_TERRAIN_TYPES = 5                              # [FIX-1]
    TERRAIN_PROBS     = [0.55, 0.20, 0.15, 0.07, 0.03]  # [FIX-10]

    ENERGY_COST    = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4}
    TERRAIN_REWARD = {0: 0.0, 1: -0.3, 2: -0.5, 3: -1.0, 4: -5.0}

    _MOVES = np.array([
        (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),
        (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
    ], dtype=np.float32)
    _MOVE_DISTS = np.linalg.norm(_MOVES, axis=1)

    def __init__(self, size: int = 20, view_size: int = 5):
        super().__init__()
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd")

        self.size       = size
        self.view_size  = view_size
        self.max_steps  = 400
        self.max_energy = 100.0

        self.action_space      = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(7 + view_size * view_size,),
            dtype=np.float32,
        )

        self.terrain:     np.ndarray | None = None
        self.pos:         np.ndarray | None = None
        self.goal:        np.ndarray | None = None
        self.visited:     np.ndarray | None = None
        self.steps:       int   = 0
        self.energy:      float = self.max_energy
        self.hazard_hits: int   = 0

    # ── helpers ───────────────────────────────────────────────────────────

    def _generate_terrain(self) -> np.ndarray:
        return self.np_random.choice(
            self.NUM_TERRAIN_TYPES,
            size=(self.size, self.size),
            p=self.TERRAIN_PROBS,
        ).astype(np.int8)

    def _is_reachable(self) -> bool:
        start = tuple(self.pos)
        goal  = tuple(self.goal)
        if start == goal:
            return True
        seen: set[tuple[int, int]] = {start}
        q:    deque[tuple[int, int]] = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                node   = (nr, nc)
                if 0 <= nr < self.size and 0 <= nc < self.size and node not in seen:
                    if node == goal:
                        return True
                    seen.add(node)
                    q.append(node)
        return False

    def _clear_path_fallback(self) -> None:
        r, c   = int(self.pos[0]),  int(self.pos[1])
        gr, gc = int(self.goal[0]), int(self.goal[1])
        while c != gc:
            self.terrain[r, c] = self.SAFE
            c += 1 if gc > c else -1
        while r != gr:
            self.terrain[r, c] = self.SAFE
            r += 1 if gr > r else -1

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start = (0, 0)
        goal  = (self.size - 1, self.size - 1)

        if options is not None:
            if "start_pos" in options:
                start = tuple(options["start_pos"])
            if "goal_pos" in options:
                goal = tuple(options["goal_pos"])

        self.pos  = np.array(start, dtype=np.int32)
        self.goal = np.array(goal,  dtype=np.int32)

        # ── [MOLA-1]  Injected terrain (real MOLA patch) ─────────────────
        if options is not None and "terrain" in options:
            injected = np.asarray(options["terrain"], dtype=np.int8)
            if injected.shape != (self.size, self.size):
                raise ValueError(
                    f"Injected terrain shape {injected.shape} != "
                    f"({self.size}, {self.size})"
                )
            self.terrain = injected.copy()
            # Force start/goal to SAFE — always safe to do
            self.terrain[self.pos[0],  self.pos[1]]  = self.SAFE
            self.terrain[self.goal[0], self.goal[1]] = self.SAFE
            # [MOLA-2]  Skip BFS — cache pre-filters reachable patches

        else:
            # ── Synthetic terrain — BFS guarantee ────────────────────────
            for _ in range(10):
                self.terrain = self._generate_terrain()
                self.terrain[self.pos[0],  self.pos[1]]  = self.SAFE
                self.terrain[self.goal[0], self.goal[1]] = self.SAFE
                if self._is_reachable():
                    break
            else:
                self._clear_path_fallback()

        self.steps       = 0
        self.energy      = self.max_energy
        self.hazard_hits = 0
        self.visited     = np.zeros((self.size, self.size), dtype=bool)  # [FIX-7]
        self.visited[self.pos[0], self.pos[1]] = True

        return self._get_obs(), self._build_info()

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: int):
        if self.terrain is None:
            raise RuntimeError("Call reset() before step()")

        self.steps += 1
        old_pos = self.pos.copy()
        move    = self._MOVES[action]
        dist    = self._MOVE_DISTS[action]

        new_x = int(np.clip(self.pos[0] + move[0], 0, self.size - 1))
        new_y = int(np.clip(self.pos[1] + move[1], 0, self.size - 1))

        # [FIX-2]  Per-axis wall check
        blocked_x = (move[0] != 0) and (new_x == self.pos[0])
        blocked_y = (move[1] != 0) and (new_y == self.pos[1])
        hit_wall  = blocked_x or blocked_y

        terrain_type = int(self.terrain[new_x, new_y])

        # energy
        energy_cost = 1.0 if hit_wall else self.ENERGY_COST[terrain_type] * dist
        self.energy = max(self.energy - energy_cost, 0.0)
        self.pos    = np.array([new_x, new_y], dtype=np.int32)

        # reward
        reward = -0.05
        if hit_wall:
            reward -= 0.5
        else:
            reward += self.TERRAIN_REWARD[terrain_type] * dist  # [FIX-3]
            if terrain_type == self.HAZARD:
                self.hazard_hits += 1

            # [NEW-1]  Track visited cells for info dict only — no bonus
            if not self.visited[new_x, new_y]:
                self.visited[new_x, new_y] = True

            # [NEW-2]  Asymmetric goal shaping — retreating penalised 2× harder
            diagonal = np.sqrt(2.0) * (self.size - 1)
            old_dist = np.linalg.norm(old_pos  - self.goal)
            new_dist = np.linalg.norm(self.pos - self.goal)
            delta    = (old_dist - new_dist) / diagonal
            if delta >= 0:
                reward += delta * 2.0   # closer  → reward (unchanged)
            else:
                reward += delta * 4.0   # farther → double penalty

        # [NEW-3]  Energy-urgency: small penalty that grows as energy depletes
        energy_fraction = self.energy / self.max_energy
        reward -= 0.10 * (1.0 - energy_fraction)

        terminated = bool(np.array_equal(self.pos, self.goal))
        if terminated:
            reward += 50.0 + 10.0 * (self.energy / self.max_energy)  # [FIX-5]

        energy_depleted = self.energy <= 0.0
        truncated       = (self.steps >= self.max_steps) or energy_depleted
        if energy_depleted and not terminated:
            reward -= 25.0  # [NEW-4]  was -10.0 — exhaustion is unambiguously bad

        return self._get_obs(), reward, terminated, truncated, self._build_info()

    # ── obs ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        pos_norm    = (self.pos  / (self.size - 1)).astype(np.float32)
        goal_norm   = (self.goal / (self.size - 1)).astype(np.float32)
        energy_norm = np.array([self.energy / self.max_energy], dtype=np.float32)

        delta = (self.goal - self.pos).astype(np.float32)
        norm  = float(np.linalg.norm(delta))
        dir_to_goal = (
            ((delta / norm) + 1.0) / 2.0
            if norm > 0.0
            else np.array([0.5, 0.5], dtype=np.float32)
        )

        half   = self.view_size // 2
        padded = np.pad(self.terrain, half, mode="constant",
                        constant_values=self.WALL_SENTINEL)
        xp, yp     = self.pos[0] + half, self.pos[1] + half
        local_view = (
            padded[xp - half : xp + half + 1,
                   yp - half : yp + half + 1]
            .flatten().astype(np.float32) / self.WALL_SENTINEL
        )

        return np.concatenate([pos_norm, goal_norm, dir_to_goal, energy_norm, local_view])

    # ── info / render ─────────────────────────────────────────────────────

    def _build_info(self) -> dict:
        return {
            "steps":            self.steps,
            "energy":           self.energy,
            "energy_fraction":  self.energy / self.max_energy,
            "hazard_hits":      self.hazard_hits,
            "explored_cells":   int(self.visited.sum()) if self.visited is not None else 0,
            "distance_to_goal": float(np.linalg.norm(self.pos - self.goal)),
        }

    def render(self):                                       # [FIX-9]
        SYMBOLS = {0: ".", 1: "S", 2: "r", 3: "s", 4: "H"}
        for row in range(self.size):
            line = []
            for col in range(self.size):
                if   np.array_equal([row, col], self.pos):  line.append("R")
                elif np.array_equal([row, col], self.goal): line.append("G")
                else: line.append(SYMBOLS[int(self.terrain[row, col])])
            print(" ".join(line))
        print(f"  Steps:{self.steps}  Energy:{self.energy:.1f}"
              f"  Hazards:{self.hazard_hits}\n")