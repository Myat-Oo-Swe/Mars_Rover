import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarsRoverEnv(gym.Env):
    """
    Mars Rover Navigation Environment
    ==================================
    The rover must navigate a procedurally-generated terrain grid from a start
    position to a goal position while managing limited energy.

    Terrain types
    -------------
    0  safe    – flat ground, cheapest to traverse
    1  sand    – loose surface, slightly costly
    2  rock    – rough ground, costly and damages instruments
    3  slope   – steep incline, very costly
    4  hazard  – dangerous area (crevasse / radiation), heavy penalty

    Observation (7 + view_size²  floats, all in [0, 1])
    ----------------------------------------------------
    pos_x, pos_y            normalised current position
    goal_x, goal_y          normalised goal position
    dir_x, dir_y            normalised unit vector toward goal mapped to [0,1]
    energy                  remaining energy fraction
    local_view (flattened)  terrain patch centred on rover (5 = out-of-bounds wall)

    Actions
    -------
    0 up | 1 down | 2 left | 3 right | 4 up-left | 5 up-right | 6 down-left | 7 down-right
    """

    metadata = {"render_modes": []}

    # ── Terrain constants ──────────────────────────────────────────────────────
    SAFE, SAND, ROCK, SLOPE, HAZARD = 0, 1, 2, 3, 4
    WALL_SENTINEL = 5          # used only in the padded observation, never on the map

    ENERGY_COST = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4}

    # Reward signal per terrain type (aligned with energy cost severity)
    TERRAIN_REWARD = {0: 0.0, 1: -0.3, 2: -0.5, 3: -1.0, 4: -5.0}

    # ── 8-direction moves (precomputed once at class level) ───────────────────
    _MOVES = np.array([
        (-1,  0),   # 0 up
        ( 1,  0),   # 1 down
        ( 0, -1),   # 2 left
        ( 0,  1),   # 3 right
        (-1, -1),   # 4 up-left
        (-1,  1),   # 5 up-right
        ( 1, -1),   # 6 down-left
        ( 1,  1),   # 7 down-right
    ], dtype=np.float32)

    _MOVE_DISTS = np.linalg.norm(_MOVES, axis=1)   # [1, 1, 1, 1, √2, √2, √2, √2]

    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, size: int = 20, view_size: int = 5):
        super().__init__()

        if view_size % 2 == 0:
            raise ValueError("view_size must be odd")

        self.size      = size
        self.view_size = view_size

        self.max_steps  = 400
        self.max_energy = 100.0

        self.action_space = spaces.Discrete(8)

        # obs: pos(2) + goal(2) + dir_to_goal(2) + energy(1) + local_view(view_size²)
        obs_size = 7 + view_size * view_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Attributes initialised properly; no reset() call here (gymnasium convention)
        self.terrain:     np.ndarray | None = None
        self.pos:         np.ndarray | None = None
        self.goal:        np.ndarray | None = None
        self.steps:       int   = 0
        self.energy:      float = self.max_energy
        self.hazard_hits: int   = 0
        self.visited:     set   = set()

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # BUG FIX #1 – use self.np_random (seeded RNG) instead of np.random
        self.terrain = self.np_random.choice(
            self.WALL_SENTINEL,
            size=(self.size, self.size),
            p=[0.55, 0.20, 0.15, 0.07, 0.03],
        ).astype(np.int8)

        start = (0, 0)
        goal  = (self.size - 1, self.size - 1)

        if options is not None:
            if "start_pos" in options:
                start = tuple(options["start_pos"])
            if "goal_pos" in options:
                goal = tuple(options["goal_pos"])

        self.pos  = np.array(start, dtype=np.int32)
        self.goal = np.array(goal,  dtype=np.int32)

        # Guarantee start and goal are always traversable
        self.terrain[self.pos[0],  self.pos[1]]  = self.SAFE
        self.terrain[self.goal[0], self.goal[1]] = self.SAFE

        self.steps       = 0
        self.energy      = self.max_energy
        self.hazard_hits = 0
        self.visited     = {tuple(self.pos)}

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        # BUG FIX #5 – explicit guard rather than cryptic AttributeError
        if self.terrain is None:
            raise RuntimeError("reset() must be called before step()")

        self.steps += 1
        old_pos = self.pos.copy()

        move = self._MOVES[action]
        dist = self._MOVE_DISTS[action]

        new_x = int(np.clip(self.pos[0] + move[0], 0, self.size - 1))
        new_y = int(np.clip(self.pos[1] + move[1], 0, self.size - 1))

        # BUG FIX #2 – detect wall collision (clipped position unchanged, move non-zero)
        hit_wall = (
            new_x == self.pos[0]
            and new_y == self.pos[1]
            and (move[0] != 0 or move[1] != 0)
        )

        terrain_type = int(self.terrain[new_x, new_y])

        # ── Energy ──────────────────────────────────────────────────────────
        if hit_wall:
            energy_cost = 1.0                              # flat cost for bumping boundary
        else:
            energy_cost = self.ENERGY_COST[terrain_type] * dist

        self.energy = max(self.energy - energy_cost, 0.0)
        self.pos    = np.array([new_x, new_y], dtype=np.int32)

        # ── Reward ──────────────────────────────────────────────────────────
        reward = -0.05   # per-step time penalty

        if hit_wall:
            # BUG FIX #2 – penalise wall bumps so the agent learns not to run into edges
            reward -= 0.5

        else:
            # BUG FIX #6 – all terrain types that cost extra energy now also give
            # a proportional reward penalty (sand was missing this before)
            reward += self.TERRAIN_REWARD[terrain_type]

            if terrain_type == self.HAZARD:
                self.hazard_hits += 1

            # Exploration bonus for visiting new cells
            pos_tuple = tuple(self.pos)
            if pos_tuple not in self.visited:
                reward += 0.2
                self.visited.add(pos_tuple)

            # BUG FIX #7 – normalise distance shaping by the grid diagonal so the
            # signal magnitude is scale-independent regardless of map size
            diagonal = np.sqrt(2.0) * (self.size - 1)
            old_dist = np.linalg.norm(old_pos  - self.goal)
            new_dist = np.linalg.norm(self.pos - self.goal)
            reward  += (old_dist - new_dist) / diagonal * 2.0

        # ── Termination ─────────────────────────────────────────────────────
        terminated = bool(np.array_equal(self.pos, self.goal))
        if terminated:
            reward += 200.0

        energy_depleted = self.energy <= 0.0
        truncated = (self.steps >= self.max_steps) or energy_depleted

        # BUG FIX #10 – penalise energy exhaustion so the agent treats energy as
        # a genuine constraint rather than an incidental truncation trigger
        if energy_depleted and not terminated:
            reward -= 10.0

        return self._get_obs(), reward, terminated, truncated, {}

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        pos_norm    = (self.pos  / (self.size - 1)).astype(np.float32)
        goal_norm   = (self.goal / (self.size - 1)).astype(np.float32)
        energy_norm = np.array([self.energy / self.max_energy], dtype=np.float32)

        # BUG FIX #8 – explicit unit-vector direction toward goal gives the agent
        # a cleaner directional signal instead of relying on implicit arithmetic
        delta = (self.goal - self.pos).astype(np.float32)
        norm  = float(np.linalg.norm(delta))
        if norm > 0.0:
            dir_to_goal = ((delta / norm) + 1.0) / 2.0   # [-1,1] → [0,1]
        else:
            dir_to_goal = np.array([0.5, 0.5], dtype=np.float32)

        # BUG FIX #3 – pad with WALL_SENTINEL (5) so the agent can distinguish
        # "out-of-bounds boundary" from a real hazard tile (4)
        half   = self.view_size // 2
        padded = np.pad(
            self.terrain,
            pad_width=half,
            mode="constant",
            constant_values=self.WALL_SENTINEL,
        )

        xp, yp     = self.pos[0] + half, self.pos[1] + half
        local_view = (
            padded[xp - half : xp + half + 1,
                   yp - half : yp + half + 1]
            .flatten()
            .astype(np.float32)
            / self.WALL_SENTINEL           # normalise to [0, 1]
        )

        return np.concatenate([
            pos_norm,
            goal_norm,
            dir_to_goal,
            energy_norm,
            local_view,
        ])