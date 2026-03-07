"""
train_ppo.py  –  Improved PPO training for MarsRoverEnv
=========================================================
Changes vs naive baseline
─────────────────────────
1. Domain randomisation  – random start/goal every episode so the policy
                           generalises rather than memorising one path.
2. Curriculum learning   – start with short-distance goals and gradually
                           increase max distance as the agent improves.
                           This solves the sparse-reward cold-start problem.
3. Larger, tuned network – 3 layers of 256 units instead of the SB3 default
                           (2×64).  Navigation over a 32-dim obs needs more
                           capacity.
4. Parallel envs         – 8 SubprocVecEnvs for faster, more diverse rollouts.
5. Tuned PPO params      – larger rollout buffer, clipped LR schedule, higher
                           entropy coefficient to prevent premature convergence.
6. SuccessRateCallback   – tracks goal-reach % during training so you can see
                           the policy actually improving.
7. Model saved at best   – checkpoint saved whenever eval success rate improves.
"""

import os
import random
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
def linear_schedule(initial_value: float):
    """Returns a schedule function that linearly decays from initial_value to 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule
from stable_baselines3.common.monitor import Monitor

from env.mars_rover_env import MarsRoverEnv


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumMarsRoverEnv(MarsRoverEnv):
    """
    Wraps MarsRoverEnv with curriculum and domain randomisation.

    Curriculum stages
    -----------------
    Stage 0  (phase < 0.25)  max Chebyshev distance = 5   (easy)
    Stage 1  (phase < 0.50)  max Chebyshev distance = 10  (medium)
    Stage 2  (phase < 0.75)  max Chebyshev distance = 15  (hard)
    Stage 3  (phase ≥ 0.75)  full random anywhere          (generalisation)

    `phase` is a float in [0, 1] set by the training callback.
    """

    def __init__(self, size: int = 20, view_size: int = 5):
        super().__init__(size=size, view_size=view_size)
        self.curriculum_phase: float = 0.0   # 0.0 = easy, 1.0 = full random

    def _max_dist(self) -> int:
        p = self.curriculum_phase
        if p < 0.25:
            return 5
        elif p < 0.50:
            return 10
        elif p < 0.75:
            return 15
        else:
            return self.size - 1   # full board

    def reset(self, seed=None, options=None):
        # FIX A – if no explicit positions provided, sample them with curriculum
        if options is None or ("start_pos" not in options and "goal_pos" not in options):
            max_d  = self._max_dist()
            margin = max(1, max_d)

            start = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )

            # keep sampling until goal is within curriculum distance
            for _ in range(1000):
                goal = (
                    random.randint(0, self.size - 1),
                    random.randint(0, self.size - 1),
                )
                # Chebyshev distance
                dist = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
                if 1 <= dist <= margin:
                    break

            options = {"start_pos": start, "goal_pos": goal}

        return super().reset(seed=seed, options=options)


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Every `check_freq` steps:
      • evaluates success rate over `n_eval_episodes` episodes
      • advances curriculum phase based on total training progress
      • prints a training log line
    """

    def __init__(
        self,
        eval_env: CurriculumMarsRoverEnv,
        train_envs,                      # VecEnv reference to update phases
        total_timesteps: int,
        check_freq:       int   = 20_000,
        n_eval_episodes:  int   = 30,
        verbose:          int   = 1,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.train_envs      = train_envs
        self.total_timesteps = total_timesteps
        self.check_freq      = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success    = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # ── advance curriculum phase ──────────────────────────────────────
        phase = min(1.0, self.num_timesteps / self.total_timesteps)

        # update phase in every training env (SubprocVecEnv requires set_attr)
        self.train_envs.set_attr("curriculum_phase", phase)
        self.eval_env.curriculum_phase = 1.0   # eval always on full-random

        # ── evaluate success rate ─────────────────────────────────────────
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
            if np.array_equal(self.eval_env.pos, self.eval_env.goal):
                successes += 1

        success_rate = successes / self.n_eval_episodes

        if self.verbose:
            print(
                f"[{self.num_timesteps:>9,} steps]  "
                f"curriculum_phase={phase:.2f}  "
                f"eval_success={success_rate*100:.1f}%"
            )

        # ── save best model ───────────────────────────────────────────────
        if success_rate > self.best_success:
            self.best_success = success_rate
            self.model.save("mars_rover_ppo_best")
            if self.verbose:
                print(f"  ↑ new best ({success_rate*100:.1f}%) — model saved")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env(rank: int):
    def _init():
        env = CurriculumMarsRoverEnv()
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Hyperparameters ───────────────────────────────────────────────────
    N_ENVS          = 8          # FIX B – parallel rollout workers
    TOTAL_STEPS     = 3_000_000  # FIX B – 3M steps; increase to 5M if needed
    BATCH_SIZE      = 512
    N_STEPS         = 1024       # steps per env per rollout → buffer = 8×1024
    N_EPOCHS        = 10
    GAMMA           = 0.995      # high γ to value long-horizon goal-reaching
    GAE_LAMBDA      = 0.95
    CLIP_RANGE      = 0.2
    ENT_COEF        = 0.01       # FIX B – entropy bonus prevents early convergence
    VF_COEF         = 0.5
    MAX_GRAD_NORM   = 0.5
    INIT_LR         = 3e-4

    # ── Vectorised training envs ──────────────────────────────────────────
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    # ── Eval env (full-random, no curriculum) ────────────────────────────
    eval_env = CurriculumMarsRoverEnv()
    eval_env.curriculum_phase = 1.0

    # ── PPO model ─────────────────────────────────────────────────────────
    # FIX C – 3-layer 256-unit network; the default 2×64 is too small for
    # a 32-dim observation that encodes terrain, position, direction and energy
    policy_kwargs = dict(
        net_arch=[256, 256, 256],
        activation_fn=__import__("torch.nn", fromlist=["Tanh"]).Tanh,
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = linear_schedule(INIT_LR),   # FIX B – decaying LR
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        gamma           = GAMMA,
        gae_lambda      = GAE_LAMBDA,
        clip_range      = CLIP_RANGE,
        ent_coef        = ENT_COEF,
        vf_coef         = VF_COEF,
        max_grad_norm   = MAX_GRAD_NORM,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = "./tb_logs/",
    )

    # ── Callback ─────────────────────────────────────────────────────────
    curriculum_cb = CurriculumCallback(
        eval_env        = eval_env,
        train_envs      = vec_env,
        total_timesteps = TOTAL_STEPS,
        check_freq      = 20_000,
        n_eval_episodes = 30,
        verbose         = 1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Mars Rover PPO Training")
    print(f"  envs={N_ENVS}  total_steps={TOTAL_STEPS:,}")
    print(f"  network=3×256  lr={INIT_LR} (linear decay)")
    print("=" * 60)

    model.learn(
        total_timesteps  = TOTAL_STEPS,
        callback         = curriculum_cb,
        progress_bar     = True,
    )

    # ── Save final model ──────────────────────────────────────────────────
    model.save("mars_rover_ppo")
    vec_env.close()
    print("\nTraining complete.  Final model: mars_rover_ppo.zip")
    print("Best model (highest eval success): mars_rover_ppo_best.zip")


if __name__ == "__main__":
    main()