"""
train_ppo.py  —  PPO training for MarsRoverEnv with real MOLA terrain
======================================================================
All changes are tagged with [FIX-N] (prior fixes) or [MOLA-N] (new).

Prior fixes retained
--------------------
FIX-1   Curriculum advances on success rate, not wall-clock steps.
FIX-2   Success detection uses `terminated` flag.
FIX-3   Inner envs NOT wrapped in Monitor — VecMonitor handles stats,
        so set_attr("curriculum_phase") reaches the real env object.
FIX-4   All random sampling uses self.np_random (seeded gymnasium RNG).
FIX-5   torch.nn imported normally.
FIX-6   VecNormalize for obs + reward normalisation.
FIX-7   Entropy coefficient decays manually in callback (SB3 only accepts
        a plain float for ent_coef, not a schedule function).
FIX-8   N_STEPS raised to 2048 for better GAE quality at γ=0.995.
FIX-9   Eval logs mean_steps_to_goal and mean_energy_at_goal per check.
FIX-10  Goal sampling uses direct offset sampling (no rejection-loop bias).
FIX-11  eval_env.close() called after training.
FIX-12  Eval env seeded for reproducible comparisons.

MOLA integration
----------------
MOLA-1  MolaTerrainCache (from mola_tutorial) pre-loads N real Mars patches
        at startup — zero disk I/O per reset during training.
MOLA-2  CurriculumMarsRoverEnv.reset() uses cache.sample_for_stage() when
        MOLA is available, injecting real terrain via the env's new
        options["terrain"] key.  Falls back to synthetic when MOLA data
        is absent so training works without the 2 GB file.
MOLA-3  Geographic curriculum replaces distance curriculum:
          phase < 0.20  →  flat northern plains (easy)
          phase < 0.40  →  plains + craters (medium)
          phase < 0.60  →  volcanoes + canyons (hard)
          phase >= 0.60 →  full random anywhere
        Phase boundaries tightened vs. the original 0.25/0.50/0.75 so the
        hardest stage (full random) occupies 40 % of total curriculum time.
MOLA-4  Start and goal positions are sampled randomly within the grid
        for all curriculum stages (not fixed 0,0 → N-1,N-1) to prevent
        the agent from memorising specific entry/exit corners.

NEW fixes (reward-shaping & curriculum, matched to mars_rover_env.py NEW-N)
---------------------------------------------------------------------------
NEW-6   Two-tier promotion threshold:
          • Stages 0→1 and 1→2 still promote at 70 % success.
          • Stage 2→3 (into full-random terrain) requires 85 % success —
            the agent must master hard terrain before graduating to the
            hardest, preventing premature arrival at phase 3 underprepared.
        STAGE_PHASES adjusted to [0.0, 0.20, 0.40, 0.60, 1.0] so the
        hardest phase covers 40 % of curriculum time once reached.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch.nn as nn                                      # [FIX-5]

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecMonitor, VecNormalize,               # [FIX-6]
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from env.mars_rover_env import MarsRoverEnv

# Optional MOLA — gracefully absent if file not downloaded yet
try:
    from mola_tutorial import MolaTerrainCache, MolaTerrain  # [MOLA-1]
    MOLA_PATH   = Path(__file__).parent / "mola_dem.tif"
    MOLA_AVAILABLE = MOLA_PATH.exists()
except ImportError:
    MOLA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumMarsRoverEnv(MarsRoverEnv):
    """
    Wraps MarsRoverEnv with MOLA real terrain + geographic curriculum.

    When MOLA is available
    ----------------------
    reset() calls cache.sample_for_stage(curriculum_phase) which returns
    a real Mars terrain patch matching the current difficulty stage, then
    injects it via options["terrain"].  Start and goal are sampled randomly
    anywhere in the grid (not fixed corners) [MOLA-4].

    When MOLA is not available
    --------------------------
    Falls back to the original distance-based curriculum with synthetic
    terrain so training still works without the 2 GB download.

    Curriculum phases  [NEW-6]
    --------------------------
    Phase boundaries tightened to [0.0, 0.20, 0.40, 0.60, 1.0] so the
    full-random stage occupies 40 % of total curriculum time.
    """

    # [NEW-6]  Tighter phase boundaries — hardest stage covers 40 % of curriculum
    STAGE_PHASES    = [0.0, 0.20, 0.40, 0.60, 1.0]
    STAGE_MAX_DISTS = [5,   10,   15,   None]   # used only for synthetic fallback

    def __init__(
        self,
        size:     int              = 20,
        view_size: int             = 5,
        cache:    MolaTerrainCache | None = None,   # [MOLA-1]
    ):
        super().__init__(size=size, view_size=view_size)
        self.curriculum_phase: float = 0.0
        self._cache = cache    # None → synthetic fallback

    # ── curriculum helpers ────────────────────────────────────────────────

    def _max_dist(self) -> int:
        """Distance budget used only for synthetic fallback."""
        for i, threshold in enumerate(self.STAGE_PHASES[1:], start=0):
            if self.curriculum_phase < threshold:
                return self.STAGE_MAX_DISTS[i]
        return self.size - 1

    def _sample_goal_near(self, start: tuple, max_d: int) -> tuple:
        """[FIX-10]  Direct offset sampling — no rejection-loop bias."""
        r0, c0 = start
        for _ in range(200):
            dr = int(self.np_random.integers(-max_d, max_d + 1))
            dc = int(self.np_random.integers(-max_d, max_d + 1))
            if max(abs(dr), abs(dc)) < 1:
                continue
            r = int(np.clip(r0 + dr, 0, self.size - 1))
            c = int(np.clip(c0 + dc, 0, self.size - 1))
            if (r, c) != start:
                return (r, c)
        return ((r0 + 1) % self.size, c0)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # Seed np_random first so subsequent np_random calls are seeded
        if seed is not None or not hasattr(self, "np_random") or self.np_random is None:
            # Call parent with no options just to seed; we override below
            super().reset(seed=seed)

        if options is not None and (
            "start_pos" in options or "goal_pos" in options or "terrain" in options
        ):
            # Explicit override — pass through unchanged
            return super().reset(seed=seed, options=options)

        # ── [MOLA-2]  Real terrain path ───────────────────────────────────
        if self._cache is not None:
            terrain = self._cache.sample_for_stage(self.curriculum_phase)  # [MOLA-3]

            # [MOLA-4]  Random start/goal anywhere in grid
            start = (
                int(self.np_random.integers(0, self.size)),
                int(self.np_random.integers(0, self.size)),
            )
            # Goal: random position different from start
            for _ in range(200):
                goal = (
                    int(self.np_random.integers(0, self.size)),
                    int(self.np_random.integers(0, self.size)),
                )
                if goal != start:
                    break

            return super().reset(seed=None, options={
                "start_pos": start,
                "goal_pos":  goal,
                "terrain":   terrain,
            })

        # ── Synthetic fallback (distance curriculum) ──────────────────────
        max_d = self._max_dist()
        start = (
            int(self.np_random.integers(0, self.size)),
            int(self.np_random.integers(0, self.size)),
        )
        goal = self._sample_goal_near(start, max_d)
        return super().reset(seed=None, options={
            "start_pos": start,
            "goal_pos":  goal,
        })


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Every check_freq steps:
      • [FIX-7]   Decay ent_coef manually (SB3 only accepts float for ent_coef).
      • [FIX-1]   Promote curriculum stage when eval success >= threshold.
      • [FIX-2]   Success detection via terminated flag.
      • [FIX-9]   Log mean_steps_to_goal and mean_energy_at_goal.
      • [NEW-6]   Two-tier promotion: 70 % for stages 0→2, 85 % for stage 2→3.
      • Save model whenever eval success rate improves.
    """

    PROMOTION_THRESHOLD       = 0.70   # [FIX-1]  stages 0→1 and 1→2
    FINAL_PROMOTION_THRESHOLD = 0.85   # [NEW-6]  stage 2→3 (into full-random)

    def __init__(
        self,
        eval_env:        CurriculumMarsRoverEnv,
        train_envs,
        total_timesteps: int,
        check_freq:      int = 20_000,
        n_eval_episodes: int = 30,
        verbose:         int = 1,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.train_envs      = train_envs
        self.total_timesteps = total_timesteps
        self.check_freq      = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success    = 0.0
        self.init_ent_coef   = 0.05    # [FIX-7]

    def _current_stage(self) -> int:
        phase  = self.eval_env.curriculum_phase
        stages = CurriculumMarsRoverEnv.STAGE_PHASES
        for i in range(len(stages) - 1):
            if phase < stages[i + 1]:
                return i
        return len(stages) - 2

    def _try_promote(self, success_rate: float) -> bool:
        stage  = self._current_stage()
        stages = CurriculumMarsRoverEnv.STAGE_PHASES
        if stage >= len(stages) - 2:
            return False

        # [NEW-6]  Two-tier threshold: last promotion into full-random is harder
        is_final_promotion = (stage == len(stages) - 3)
        threshold = (
            self.FINAL_PROMOTION_THRESHOLD
            if is_final_promotion
            else self.PROMOTION_THRESHOLD
        )

        if success_rate < threshold:
            return False

        new_phase = stages[stage + 1]
        self.train_envs.set_attr("curriculum_phase", new_phase)  # [FIX-3]
        self.eval_env.curriculum_phase = new_phase
        if self.verbose:
            print(f"  ↑ Curriculum promoted → stage {stage + 1}  "
                  f"(phase={new_phase:.2f})  "
                  f"[threshold used: {threshold * 100:.0f}%]")
        return True

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # [FIX-7]  Decay ent_coef manually
        progress_remaining = max(0.0, 1.0 - self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = float(self.init_ent_coef * progress_remaining)

        # Eval always on full-random / hard terrain
        self.eval_env.curriculum_phase = 1.0

        successes   = 0
        steps_list  = []
        energy_list = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
            if terminated:                           # [FIX-2]
                successes += 1
                steps_list.append(self.eval_env.steps)
                energy_list.append(self.eval_env.energy)

        success_rate = successes / self.n_eval_episodes

        if self.verbose:
            mean_steps  = np.mean(steps_list)  if steps_list  else float("nan")
            mean_energy = np.mean(energy_list) if energy_list else float("nan")
            stage       = self._current_stage()
            # Show which threshold applies for the next promotion attempt
            stages      = CurriculumMarsRoverEnv.STAGE_PHASES
            is_final    = (stage == len(stages) - 3)
            next_thresh = (
                self.FINAL_PROMOTION_THRESHOLD if is_final
                else self.PROMOTION_THRESHOLD
            )
            print(
                f"[{self.num_timesteps:>9,} steps]  "
                f"phase={self.eval_env.curriculum_phase:.2f}  "
                f"success={success_rate * 100:.1f}%"
                f"(need {next_thresh * 100:.0f}%)  "
                f"mean_steps={mean_steps:.0f}  "         # [FIX-9]
                f"mean_energy={mean_energy:.1f}  "       # [FIX-9]
                f"ent_coef={self.model.ent_coef:.4f}"    # [FIX-7]
            )

        self._try_promote(success_rate)

        if success_rate > self.best_success:
            self.best_success = success_rate
            self.model.save("mars_rover_ppo_best")
            if self.verbose:
                print(f"  ↑ new best ({success_rate * 100:.1f}%) — saved")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env(rank: int, cache: MolaTerrainCache | None):
    def _init():
        # [FIX-3]  No Monitor wrapper — VecMonitor on vec_env handles stats
        #          so set_attr("curriculum_phase") reaches the env directly.
        env = CurriculumMarsRoverEnv(cache=cache)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Hyperparameters ───────────────────────────────────────────────────
    N_ENVS          = 8
    TOTAL_STEPS     = 3_000_000
    BATCH_SIZE      = 512
    N_STEPS         = 2048          # [FIX-8]
    N_EPOCHS        = 10
    GAMMA           = 0.995
    GAE_LAMBDA      = 0.95
    CLIP_RANGE      = 0.2
    INIT_ENT_COEF   = 0.05          # [FIX-7]
    VF_COEF         = 0.5
    MAX_GRAD_NORM   = 0.5
    INIT_LR         = 3e-4

    # ── [MOLA-1]  Build terrain cache at startup ──────────────────────────
    cache = None
    if MOLA_AVAILABLE:
        print("\n  MOLA file detected — building terrain cache…")
        cache = MolaTerrainCache(
            n_patches = 400,
            size      = 20,
            seed      = 42,
            verbose   = True,
        )
        terrain_src = f"MOLA real Mars terrain ({len(cache)} patches)"
    else:
        terrain_src = "synthetic procedural (MOLA not downloaded)"
        print(f"\n  mola_dem.tif not found — using synthetic terrain.")
        print(f"  To use real Mars data: python mola_tutorial.py --download\n")

    # ── Vectorised training envs ──────────────────────────────────────────
    # Pass cache to every worker — each worker gets the same pre-loaded
    # patch list (read-only numpy arrays, safe to share across fork).
    vec_env = SubprocVecEnv([make_env(i, cache) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(                                # [FIX-6]
        vec_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        gamma       = GAMMA,
    )

    # ── Eval env — always full-random (phase=1.0), with MOLA if available ─
    eval_env = CurriculumMarsRoverEnv(cache=cache)
    eval_env.reset(seed=42)                                # [FIX-12]
    eval_env.curriculum_phase = 1.0

    # ── PPO model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch      = [256, 256, 256],
        activation_fn = nn.Tanh,                           # [FIX-5]
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = linear_schedule(INIT_LR),
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        gamma           = GAMMA,
        gae_lambda      = GAE_LAMBDA,
        clip_range      = CLIP_RANGE,
        ent_coef        = INIT_ENT_COEF,          # float; callback decays it [FIX-7]
        vf_coef         = VF_COEF,
        max_grad_norm   = MAX_GRAD_NORM,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = "./tb_logs/",
    )

    # ── Callback ──────────────────────────────────────────────────────────
    curriculum_cb = CurriculumCallback(
        eval_env        = eval_env,
        train_envs      = vec_env,
        total_timesteps = TOTAL_STEPS,
        check_freq      = 20_000,
        n_eval_episodes = 30,
        verbose         = 1,
    )

    # ── Info ──────────────────────────────────────────────────────────────
    stages = CurriculumMarsRoverEnv.STAGE_PHASES
    print("=" * 62)
    print("  Mars Rover PPO Training")
    print(f"  envs={N_ENVS}  total_steps={TOTAL_STEPS:,}")
    print(f"  network=3×256  lr={INIT_LR} (linear decay)")
    print(f"  ent_coef={INIT_ENT_COEF} → 0 (manual decay in callback)")
    print(f"  VecNormalize=ON")
    print(f"  Terrain: {terrain_src}")
    print(f"  Curriculum phases: {stages}  [NEW-6]")
    print(f"  Curriculum: {'geographic (MOLA regions)' if cache else 'distance-based (synthetic)'}")
    print(f"  Promotion thresholds: "
          f"stages 0→2 = {CurriculumCallback.PROMOTION_THRESHOLD * 100:.0f}%  "
          f"stage 2→3 = {CurriculumCallback.FINAL_PROMOTION_THRESHOLD * 100:.0f}%  [NEW-6]")
    print("=" * 62)

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = curriculum_cb,
        progress_bar    = True,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    model.save("mars_rover_ppo")
    vec_env.save("vec_normalize.pkl")                      # [FIX-6]

    eval_env.close()                                       # [FIX-11]
    vec_env.close()

    print("\nTraining complete.")
    print("  Final model      : mars_rover_ppo.zip")
    print("  Best model       : mars_rover_ppo_best.zip")
    print("  Normalizer stats : vec_normalize.pkl")
    print("\nTo load for inference:")
    print("  model   = PPO.load('mars_rover_ppo_best')")
    print("  vec_env = VecNormalize.load('vec_normalize.pkl', vec_env)")
    print("  vec_env.training    = False")
    print("  vec_env.norm_reward = False")


if __name__ == "__main__":
    main()