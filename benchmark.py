"""
benchmark.py  —  MOLA Random Location Benchmark with A* Comparison
===================================================================
Runs N missions on real MOLA Mars terrain.
Every mission uses:
  • A randomly sampled geographic location from the MOLA DEM
  • A randomly sampled start position within the terrain patch
  • A randomly sampled goal position within the terrain patch
  • A* oracle to compute the optimal path (same cost model as rover)

Metrics reported per mission and in aggregate:
  success / fail_mode / steps / energy / hazard_hits /
  optimal_energy / optimal_steps / efficiency_pct / extra_steps

Usage
-----
  # Single model benchmark  →  CSV: benchmark_<model>.csv
  python benchmark.py
  python benchmark.py --model mars_rover_ppo_best
  python benchmark.py --model mars_rover_ppo_best --missions 500

  # Old vs new comparison   →  CSV: comparison_<model1>_vs_<model2>.csv
  python benchmark.py --model mars_rover_ppo_best --model2 mars_rover_ppo_old
  python benchmark.py --model mars_rover_ppo_best --model2 mars_rover_ppo_old --normalizer2 none

  # Override CSV name explicitly (or pass '' to disable)
  python benchmark.py --csv my_results.csv
  python benchmark.py --csv ''

  # Other options
  python benchmark.py --missions 1000
  python benchmark.py --missions 500 --seed 7
  python benchmark.py --size 30
  python benchmark.py --min-dist 5 --max-dist 27

Requirements
------------
  mola_dem.tif must exist (python mola_tutorial.py --download)
  vec_normalize.pkl auto-detected if present
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.mars_rover_env import MarsRoverEnv

try:
    from mola_tutorial import MolaTerrain
    _MOLA_PATH = Path(__file__).parent / "mola_dem.tif"
except ImportError:
    MolaTerrain = None
    _MOLA_PATH  = None

# ── Same cost tables as MarsRoverEnv ──────────────────────────────────────────
_ENERGY_COST = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4}
_MOVES_8 = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, 1.4142135623730951),
    (-1,  1, 1.4142135623730951),
    ( 1, -1, 1.4142135623730951),
    ( 1,  1, 1.4142135623730951),
]

TERRAIN_NAMES = {0: "Safe", 1: "Sand", 2: "Rock", 3: "Slope", 4: "Hazard"}

# Sentinel — means the user did NOT pass --csv, so we auto-name it
_CSV_UNSET = "__AUTO__"


# ─────────────────────────────────────────────────────────────────────────────
# CSV NAME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_stem(model_path: str) -> str:
    """Return the model filename stem, sanitised for use in a CSV name."""
    return Path(model_path).stem.replace(" ", "_")


def _resolve_csv_path(csv_arg: str, model1: str, model2: Optional[str]) -> str:
    """
    Return the CSV output path.

    Rules
    -----
    - ''           → disabled; the caller checks for this.
    - any explicit string → used verbatim.
    - _CSV_UNSET   → auto-named:
        single model  : benchmark_<model1>.csv
        two models    : comparison_<model1>_vs_<model2>.csv
    """
    if csv_arg == "":
        return ""                   # disabled
    if csv_arg != _CSV_UNSET:
        return csv_arg              # explicit override
    stem1 = _safe_stem(model1)
    if model2:
        stem2 = _safe_stem(model2)
        return f"comparison_{stem1}_vs_{stem2}.csv"
    return f"benchmark_{stem1}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# A* ORACLE
# ─────────────────────────────────────────────────────────────────────────────

def astar(
    terrain: np.ndarray,
    start:   tuple[int, int],
    goal:    tuple[int, int],
) -> tuple[list, float]:
    """
    Optimal path via A* with the exact rover cost model.
    Heuristic: min(dx,dy)×√2 + |dx-dy| (admissible, octile distance).
    Returns (path, total_energy) or ([], inf) if unreachable.
    """
    size = terrain.shape[0]
    sr, sc = start
    gr, gc = goal

    def h(r: int, c: int) -> float:
        dx, dy = abs(gr - r), abs(gc - c)
        return min(dx, dy) * 1.4142135623730951 + abs(dx - dy)

    g = np.full((size, size), np.inf, dtype=np.float64)
    g[sr, sc] = 0.0
    came: dict[tuple, Optional[tuple]] = {(sr, sc): None}
    heap = [(h(sr, sc), 0.0, sr, sc)]

    while heap:
        _, gc_cost, r, c = heapq.heappop(heap)
        if gc_cost > g[r, c] + 1e-9:
            continue
        if (r, c) == (gr, gc):
            break
        for dr, dc, dist in _MOVES_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            ng = gc_cost + _ENERGY_COST[int(terrain[nr, nc])] * dist
            if ng < g[nr, nc] - 1e-9:
                g[nr, nc] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))

    if np.isinf(g[gr, gc]):
        return [], float("inf")

    path, node = [], (gr, gc)
    while node:
        path.append(list(node))
        node = came.get(node)
    path.reverse()
    return path, float(g[gr, gc])


# ─────────────────────────────────────────────────────────────────────────────
# VECNORMALIZE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self, model: PPO, env: MarsRoverEnv,
                 normalizer_path: Optional[str]):
        self._model     = model
        self._vec_norm  = None
        self.normalized = False

        if normalizer_path and Path(normalizer_path).exists():
            try:
                venv = DummyVecEnv([lambda: env])
                self._vec_norm = VecNormalize.load(normalizer_path, venv)
                self._vec_norm.training    = False
                self._vec_norm.norm_reward = False
                self.normalized = True
                print(f"  VecNormalize  : loaded from {normalizer_path}")
            except Exception as e:
                print(f"  VecNormalize  : load failed ({e}) — raw obs")
        else:
            print("  VecNormalize  : not found — raw obs")

    def predict(self, obs: np.ndarray) -> int:
        if self._vec_norm is not None:
            obs = self._vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MISSION
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    predictor: Predictor,
    env:       MarsRoverEnv,
    start:     tuple[int, int],
    goal:      tuple[int, int],
    terrain:   np.ndarray,
    lat:       float,
    lon:       float,
) -> dict:
    """Run one rover episode and compute A* comparison."""
    obs, _ = env.reset(options={"start_pos": start, "goal_pos": goal,
                                "terrain": terrain})

    # A* oracle — sees full map
    opt_path, opt_energy = astar(env.terrain, start, goal)

    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(predictor.predict(obs))
        done = terminated or truncated

    rover_energy = float(env.max_energy - env.energy)

    # ── Failure mode ──────────────────────────────────────────────────────
    if terminated:
        fail_mode = None
    elif env.energy <= 0 and env.hazard_hits >= 5:
        fail_mode = "hazard_death"
    elif env.energy <= 0:
        fail_mode = "energy_exhausted"
    elif env.steps >= env.max_steps:
        fail_mode = "timeout"
    else:
        fail_mode = "unknown"

    # ── A* comparison ─────────────────────────────────────────────────────
    if terminated and not np.isinf(opt_energy) and opt_energy > 0 and rover_energy > 0:
        efficiency = round(min(100.0, opt_energy / rover_energy * 100), 1)
    else:
        efficiency = None

    extra_steps = (env.steps - (len(opt_path) - 1)) if (opt_path and terminated) else None

    # ── Chebyshev distance of the mission ─────────────────────────────────
    cheb_dist = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))

    # ── Terrain composition of optimal path ──────────────────────────────
    terrain_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for p in opt_path:
        terrain_counts[int(terrain[p[0], p[1]])] += 1

    return {
        "lat":            round(lat, 3),
        "lon":            round(lon, 3),
        "start":          list(start),
        "goal":           list(goal),
        "cheb_dist":      cheb_dist,
        "success":        bool(terminated),
        "fail_mode":      fail_mode,
        "steps":          env.steps,
        "energy_used":    round(rover_energy, 2),
        "energy_left":    round(float(env.energy), 2),
        "hazard_hits":    env.hazard_hits,
        "opt_reachable":  len(opt_path) > 0,
        "opt_energy":     round(opt_energy, 2) if not np.isinf(opt_energy) else None,
        "opt_steps":      len(opt_path) - 1    if opt_path else None,
        "efficiency_pct": efficiency,
        "extra_steps":    extra_steps,
        "terrain_on_opt": terrain_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mean(vals: list) -> str:
    v = [x for x in vals if x is not None]
    return f"{np.mean(v):.2f}" if v else "—"

def _pct(n: int, total: int) -> str:
    return f"{n}/{total}  ({n / total * 100:.1f}%)"

def _bar(pct: float, width: int = 28) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# PER-MISSION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_mission_table(results: list[dict], env: MarsRoverEnv):
    print(f"\n{'─'*100}")
    print(f"  {'#':>4}  {'Lat':>7}  {'Lon':>8}  {'Start→Goal':<16}  "
          f"{'Dist':>5}  {'Result':<16}  {'Steps':>5}  {'Energy':>10}  "
          f"{'Haz':>4}  {'Opt E':>7}  {'Eff%':>6}  {'Xtra':>5}")
    print(f"{'─'*100}")

    for i, r in enumerate(results, 1):
        mark  = "✓" if r["success"] else "✗"
        stat  = "SUCCESS" if r["success"] else (r["fail_mode"] or "FAILED")
        sg    = f"({r['start'][0]},{r['start'][1]})→({r['goal'][0]},{r['goal'][1]})"
        eff   = f"{r['efficiency_pct']}%" if r["efficiency_pct"]  is not None else "—"
        opt_e = f"{r['opt_energy']}"      if r["opt_energy"]       is not None else "—"
        xtra  = f"+{r['extra_steps']}"   if r["extra_steps"]       is not None else "—"
        e_str = f"{r['energy_used']:.1f}/{env.max_energy:.0f}"

        print(f"  {i:>4}  {r['lat']:>7.2f}  {r['lon']:>8.2f}  {sg:<16}  "
              f"{r['cheb_dist']:>5}  {mark} {stat:<14}  {r['steps']:>5}  "
              f"{e_str:>10}  {r['hazard_hits']:>4}  {opt_e:>7}  "
              f"{eff:>6}  {xtra:>5}")

    print(f"{'─'*100}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], env: MarsRoverEnv, label: str = "MODEL"):
    n  = len(results)
    ok = [r for r in results if r["success"]]
    fl = [r for r in results if not r["success"]]

    eff_vals  = [r["efficiency_pct"] for r in ok if r["efficiency_pct"] is not None]
    xtra_vals = [r["extra_steps"]    for r in ok if r["extra_steps"]    is not None]

    # Failure breakdown
    by_mode: dict[str, int] = {}
    for r in fl:
        key = r["fail_mode"] or "unknown"
        by_mode[key] = by_mode.get(key, 0) + 1

    # Efficiency by Chebyshev distance bucket
    dist_buckets = {
        "short  (1–6)":  [r for r in ok if  1 <= r["cheb_dist"] <=  6],
        "medium (7–13)": [r for r in ok if  7 <= r["cheb_dist"] <= 13],
        "long   (14+)":  [r for r in ok if r["cheb_dist"] >= 14],
    }

    print(f"\n{'='*72}")
    print(f"  {label.upper()}  —  {n} MOLA RANDOM MISSIONS")
    print(f"{'='*72}")

    # ── Overall ───────────────────────────────────────────────────────────
    sr = len(ok) / n * 100
    print(f"\n  Overall success    : {_pct(len(ok), n)}")
    print(f"  {_bar(sr)}  {sr:.1f}%\n")

    # ── Rover stats ───────────────────────────────────────────────────────
    print(f"  ── Rover stats (all {n} missions) {'─'*32}")
    print(f"  Avg steps          : {_mean([r['steps']       for r in results])}")
    print(f"  Avg energy used    : {_mean([r['energy_used'] for r in results])} / {env.max_energy:.0f}")
    print(f"  Avg hazard hits    : {_mean([r['hazard_hits'] for r in results])}")
    print(f"  Avg Chebyshev dist : {_mean([r['cheb_dist']   for r in results])}")

    if ok:
        print(f"\n  ── Successful missions ({len(ok)}) {'─'*36}")
        print(f"  Avg steps          : {_mean([r['steps']       for r in ok])}")
        print(f"  Avg energy used    : {_mean([r['energy_used'] for r in ok])}")
        print(f"  Avg energy left    : {_mean([r['energy_left'] for r in ok])}")
        print(f"  Avg hazard hits    : {_mean([r['hazard_hits'] for r in ok])}")

    # ── Failures ──────────────────────────────────────────────────────────
    if fl:
        print(f"\n  ── Failures ({len(fl)}) {'─'*50}")
        labels = {
            "energy_exhausted": "energy exhausted — agent wandered",
            "hazard_death":     "hazard death — depleted via hazards",
            "timeout":          "step limit reached",
            "unknown":          "other / unclassified",
        }
        for mode, count in sorted(by_mode.items(), key=lambda x: -x[1]):
            print(f"  {labels.get(mode, mode):<42} {_pct(count, n)}")

        print(f"\n  Failed mission details:")
        for r in fl:
            sg = f"({r['start'][0]},{r['start'][1]})→({r['goal'][0]},{r['goal'][1]})"
            print(f"    lat={r['lat']:>7.2f}  lon={r['lon']:>8.2f}  "
                  f"{sg:<16}  dist={r['cheb_dist']:>2}  "
                  f"{r['fail_mode'] or 'FAILED':<20}  "
                  f"steps={r['steps']}  haz={r['hazard_hits']}")

    # ── A* comparison ─────────────────────────────────────────────────────
    if eff_vals:
        print(f"\n  ── A* efficiency (successful missions, n={len(ok)}) {'─'*22}")
        print(f"  Avg efficiency     : {float(np.mean(eff_vals)):.1f}%"
              f"  (rover energy / A* optimal energy × 100)")
        print(f"  Median efficiency  : {float(np.median(eff_vals)):.1f}%")
        print(f"  Best efficiency    : {max(eff_vals):.1f}%")
        print(f"  Worst efficiency   : {min(eff_vals):.1f}%")
        if xtra_vals:
            avg_x = float(np.mean(xtra_vals))
            print(f"  Avg extra steps    : {'+' if avg_x >= 0 else ''}{avg_x:.1f} vs A* optimal")
            print(f"  Max extra steps    : +{max(xtra_vals)} (worst wander)")

        print(f"\n  Efficiency distribution:")
        buckets_eff = [
            ("≥ 90%  near-optimal",  sum(1 for e in eff_vals if e >= 90)),
            ("70–90% good",          sum(1 for e in eff_vals if 70 <= e < 90)),
            ("50–70% acceptable",    sum(1 for e in eff_vals if 50 <= e < 70)),
            ("<  50% wasteful",      sum(1 for e in eff_vals if e < 50)),
        ]
        for blabel, count in buckets_eff:
            bar = _bar(count / len(eff_vals) * 100, 24)
            print(f"    {blabel:<22}  {bar}  {_pct(count, len(eff_vals))}")

    # ── Efficiency by distance ─────────────────────────────────────────────
    has_dist_data = any(
        r["efficiency_pct"] is not None
        for bkt in dist_buckets.values()
        for r in bkt
    )
    if has_dist_data:
        print(f"\n  Efficiency by route distance:")
        for dlabel, bkt_missions in dist_buckets.items():
            bkt_eff = [r["efficiency_pct"] for r in bkt_missions
                       if r["efficiency_pct"] is not None]
            if bkt_missions:
                n_bkt  = sum(1 for r in results if (
                    (1  <= r["cheb_dist"] <= 6  and dlabel.startswith("short")) or
                    (7  <= r["cheb_dist"] <= 13 and dlabel.startswith("medium")) or
                    (r["cheb_dist"] >= 14        and dlabel.startswith("long"))
                ))
                n_ok_b = len(bkt_missions)
                sr_bkt = n_ok_b / n_bkt * 100 if n_bkt > 0 else 0
                avg_e  = f"{float(np.mean(bkt_eff)):.1f}%" if bkt_eff else "—"
                print(f"    {dlabel}   success={sr_bkt:.0f}%  "
                      f"avg_efficiency={avg_e}  n={n_bkt}")

    # ── Terrain on optimal paths ───────────────────────────────────────────
    total_cells = sum(
        sum(r["terrain_on_opt"].values())
        for r in ok if r["opt_reachable"]
    )
    if total_cells > 0:
        combined: dict[int, int] = {k: 0 for k in range(5)}
        for r in ok:
            if r["opt_reachable"]:
                for k, v in r["terrain_on_opt"].items():
                    combined[k] += v
        print(f"\n  Terrain on optimal paths (across all successful missions):")
        for tid, name in TERRAIN_NAMES.items():
            count = combined[tid]
            pct   = count / total_cells * 100
            bar   = _bar(pct, 20)
            print(f"    {name:<8}  {bar}  {count:>5} cells  ({pct:.1f}%)")

    print(f"\n{'='*72}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT  —  single model
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(results: list[dict], path: str, label: str = ""):
    fields = [
        "mission_id", "lat", "lon", "start", "goal", "cheb_dist",
        "success", "fail_mode", "steps", "energy_used", "energy_left",
        "hazard_hits", "opt_reachable", "opt_energy", "opt_steps",
        "efficiency_pct", "extra_steps",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(results, 1):
            w.writerow({"mission_id": i, **r})
    tag = f" [{label}]" if label else ""
    print(f"\n  CSV saved{tag} → {path}  ({len(results)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# HEAD-TO-HEAD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def print_head_to_head(
    r1: list[dict], r2: list[dict],
    env: MarsRoverEnv,
    label1: str, label2: str,
):
    """Side-by-side comparison table on identical missions."""
    n   = len(r1)
    ok1 = [r for r in r1 if r["success"]]
    ok2 = [r for r in r2 if r["success"]]

    eff1  = [r["efficiency_pct"] for r in ok1 if r["efficiency_pct"] is not None]
    eff2  = [r["efficiency_pct"] for r in ok2 if r["efficiency_pct"] is not None]
    xtr1  = [r["extra_steps"]    for r in ok1 if r["extra_steps"]    is not None]
    xtr2  = [r["extra_steps"]    for r in ok2 if r["extra_steps"]    is not None]

    both_ok   = sum(1 for a, b in zip(r1, r2) if     a["success"] and     b["success"])
    only_m1   = sum(1 for a, b in zip(r1, r2) if     a["success"] and not b["success"])
    only_m2   = sum(1 for a, b in zip(r1, r2) if not a["success"] and     b["success"])
    both_fail = sum(1 for a, b in zip(r1, r2) if not a["success"] and not b["success"])

    def _w(v1: str, v2: str, higher_better: bool = True) -> tuple[str, str]:
        try:
            f1 = float(v1.rstrip("%").rstrip(" ←"))
            f2 = float(v2.rstrip("%").rstrip(" ←"))
            if higher_better:
                if f1 > f2 + 0.05: return v1 + " ←", v2
                if f2 > f1 + 0.05: return v1, v2 + " ←"
            else:
                if f1 < f2 - 0.05: return v1 + " ←", v2
                if f2 < f1 - 0.05: return v1, v2 + " ←"
        except (ValueError, AttributeError):
            pass
        return v1, v2

    L = max(len(label1), len(label2), 22)

    print(f"\n{'='*72}")
    print(f"  HEAD-TO-HEAD  —  {n} identical terrain + positions")
    print(f"{'='*72}")
    print(f"\n  {'Metric':<32}  {label1:<{L}}  {label2:<{L}}")
    print(f"  {'─'*32}  {'─'*L}  {'─'*L}")

    def row(metric: str, v1: str, v2: str, hb: bool = True):
        a, b = _w(v1, v2, hb)
        print(f"  {metric:<32}  {a:<{L}}  {b:<{L}}")

    row("Success rate",
        f"{len(ok1)/n*100:.1f}% ({len(ok1)}/{n})",
        f"{len(ok2)/n*100:.1f}% ({len(ok2)}/{n})")
    row("Avg steps (all missions)",
        _mean([r["steps"]       for r in r1]),
        _mean([r["steps"]       for r in r2]), hb=False)
    row("Avg energy used (all)",
        _mean([r["energy_used"] for r in r1]),
        _mean([r["energy_used"] for r in r2]), hb=False)
    row("Avg hazard hits (all)",
        _mean([r["hazard_hits"] for r in r1]),
        _mean([r["hazard_hits"] for r in r2]), hb=False)

    print(f"  {'─'*32}  {'─'*L}  {'─'*L}")
    row("Avg steps (success only)",
        _mean([r["steps"]       for r in ok1]),
        _mean([r["steps"]       for r in ok2]), hb=False)
    row("Avg energy left (success)",
        _mean([r["energy_left"] for r in ok1]),
        _mean([r["energy_left"] for r in ok2]))
    row("Avg hazard hits (success)",
        _mean([r["hazard_hits"] for r in ok1]),
        _mean([r["hazard_hits"] for r in ok2]), hb=False)

    print(f"  {'─'*32}  {'─'*L}  {'─'*L}")
    row("Avg A* efficiency",
        f"{float(np.mean(eff1)):.1f}%" if eff1 else "—",
        f"{float(np.mean(eff2)):.1f}%" if eff2 else "—")
    row("Median A* efficiency",
        f"{float(np.median(eff1)):.1f}%" if eff1 else "—",
        f"{float(np.median(eff2)):.1f}%" if eff2 else "—")

    def _signed(v: float) -> str:
        return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

    row("Avg extra steps vs A*",
        _signed(float(np.mean(xtr1))) if xtr1 else "—",
        _signed(float(np.mean(xtr2))) if xtr2 else "—", hb=False)
    row("Max extra steps (worst wander)",
        f"+{max(xtr1)}" if xtr1 else "—",
        f"+{max(xtr2)}" if xtr2 else "—", hb=False)

    print(f"\n  Mission outcome breakdown (same terrains + positions):")
    print(f"    Both succeed             : {_pct(both_ok,   n)}")
    print(f"    Only {label1:<20} : {_pct(only_m1, n)}")
    print(f"    Only {label2:<20} : {_pct(only_m2, n)}")
    print(f"    Both fail                : {_pct(both_fail, n)}")

    gap1 = sum(
        1 for a, b in zip(r1, r2)
        if a["success"] and b["success"]
        and a.get("efficiency_pct") and b.get("efficiency_pct")
        and a["efficiency_pct"] - b["efficiency_pct"] > 10
    )
    gap2 = sum(
        1 for a, b in zip(r1, r2)
        if a["success"] and b["success"]
        and a.get("efficiency_pct") and b.get("efficiency_pct")
        and b["efficiency_pct"] - a["efficiency_pct"] > 10
    )
    both_ok_count = sum(
        1 for a, b in zip(r1, r2)
        if a["success"] and b["success"]
        and a.get("efficiency_pct") and b.get("efficiency_pct")
    )
    if both_ok_count > 0:
        print(f"\n  Among {both_ok_count} missions both succeeded "
              f"(efficiency gap > 10 pct pts):")
        print(f"    {label1} more efficient : {gap1}")
        print(f"    {label2} more efficient : {gap2}")
        print(f"    Roughly equal           : {both_ok_count - gap1 - gap2}")

    print(f"\n  ← marks the better value in each row")
    print(f"\n{'='*72}")


# ─────────────────────────────────────────────────────────────────────────────
# DUAL CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv_dual(
    r1: list[dict], r2: list[dict],
    path: str, label1: str, label2: str,
):
    """One CSV row per mission, columns for both models side by side."""
    shared_fields = [
        "mission_id", "lat", "lon", "start", "goal",
        "cheb_dist", "opt_reachable", "opt_energy", "opt_steps",
    ]
    per_model = [
        "success", "fail_mode", "steps",
        "energy_used", "energy_left", "hazard_hits",
        "efficiency_pct", "extra_steps",
    ]
    header = (shared_fields
              + [f"{label1}_{f}" for f in per_model]
              + [f"{label2}_{f}" for f in per_model])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (a, b) in enumerate(zip(r1, r2), 1):
            shared = [i, a["lat"], a["lon"], a["start"], a["goal"],
                      a["cheb_dist"], a["opt_reachable"],
                      a["opt_energy"], a["opt_steps"]]
            row1 = [a.get(f) for f in per_model]
            row2 = [b.get(f) for f in per_model]
            w.writerow(shared + row1 + row2)

    print(f"\n  CSV saved [comparison] → {path}  ({len(r1)} rows, both models)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MOLA random benchmark with A* comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV naming (when --csv is not explicitly set):
  single model  →  benchmark_<model>.csv
  two models    →  comparison_<model1>_vs_<model2>.csv

Pass --csv '' to disable CSV export entirely.
        """,
    )
    parser.add_argument("--model",       default="mars_rover_ppo_best",
                        help="Primary model file without .zip")
    parser.add_argument("--normalizer",  default=None,
                        help="VecNormalize .pkl for primary model "
                             "(auto-detected if omitted)")
    parser.add_argument("--model2",      default=None,
                        help="Second model for old-vs-new comparison "
                             "(no .zip). Activates comparison mode.")
    parser.add_argument("--normalizer2", default=None,
                        help="VecNormalize .pkl for second model "
                             "(pass 'none' to force raw obs)")
    parser.add_argument("--missions",   type=int, default=1000,
                        help="Number of random MOLA missions (default 1000)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--size",       type=int, default=20,
                        help="Terrain patch size (default 20)")
    parser.add_argument("--lat-range",  nargs=2, type=float,
                        default=[-60.0, 60.0], metavar=("MIN", "MAX"),
                        help="Latitude range for random sampling")
    parser.add_argument("--min-dist",   type=int, default=14,
                        help="Min Chebyshev distance between start and goal "
                             "(default 14 — long routes)")
    parser.add_argument("--max-dist",   type=int, default=None,
                        help="Max Chebyshev distance (default: grid diagonal)")
    parser.add_argument(
        "--csv",
        default=_CSV_UNSET,
        metavar="FILE",
        help=(
            "CSV output path. Auto-named when omitted: "
            "benchmark_<model>.csv (single) or "
            "comparison_<m1>_vs_<m2>.csv (dual). "
            "Pass '' to disable."
        ),
    )
    args = parser.parse_args()

    # ── Require MOLA ─────────────────────────────────────────────────────
    if MolaTerrain is None:
        print("ERROR: mola_tutorial.py not found — cannot import MolaTerrain")
        sys.exit(1)
    if not _MOLA_PATH.exists():
        print("ERROR: mola_dem.tif not found.")
        print("  Download it: python mola_tutorial.py --download")
        sys.exit(1)

    rng      = np.random.default_rng(args.seed)
    size     = args.size
    max_dist = args.max_dist or (size - 1)
    lat_min, lat_max = args.lat_range

    # ── Resolve CSV path now that we know both model names ────────────────
    csv_path = _resolve_csv_path(args.csv, args.model, args.model2)

    # ── Determine run mode ────────────────────────────────────────────────
    comparison_mode = args.model2 is not None

    # ── Load primary model ────────────────────────────────────────────────
    print(f"\n  Mode          : "
          f"{'comparison — old vs new' if comparison_mode else 'single model benchmark'}")
    print(f"\n  Model 1       : {args.model}")
    try:
        model1 = PPO.load(args.model)
    except Exception as e:
        print(f"  ERROR loading model 1: {e}")
        sys.exit(1)

    norm1 = args.normalizer
    if norm1 is None:
        for p in [Path(args.model).parent / "vec_normalize.pkl",
                  Path("vec_normalize.pkl")]:
            if p.exists():
                norm1 = str(p)
                break

    env1       = MarsRoverEnv(size=size)
    predictor1 = Predictor(model1, env1, norm1)

    # ── Load second model (optional) ─────────────────────────────────────
    predictor2 = None
    env2       = None
    if comparison_mode:
        print(f"\n  Model 2       : {args.model2}")
        try:
            model2 = PPO.load(args.model2)
        except Exception as e:
            print(f"  ERROR loading model 2: {e}")
            sys.exit(1)

        if args.normalizer2 and args.normalizer2.lower() == "none":
            norm2 = None
        elif args.normalizer2:
            norm2 = args.normalizer2
        else:
            norm2 = None
            for p in [Path(args.model2).parent / "vec_normalize.pkl"]:
                if p.exists():
                    norm2 = str(p)
                    break

        env2       = MarsRoverEnv(size=size)
        predictor2 = Predictor(model2, env2, norm2)

    mola = MolaTerrain()

    print(f"\n  Terrain       : MOLA real Mars — unique location per mission  "
          f"(lat {lat_min}° – {lat_max}°)")
    print(f"  Missions      : {args.missions}")
    print(f"  Patch size    : {size}×{size}  ({size * 0.463:.1f} km²)")
    print(f"  Start/goal    : random, Chebyshev dist {args.min_dist}–{max_dist}"
          f"  (long-distance default)")
    print(f"  Seed          : {args.seed}")
    print(f"  CSV output    : {csv_path if csv_path else '(disabled)'}")
    if comparison_mode:
        print(f"  Comparison    : head-to-head on identical terrain + positions")

    # ── Run missions ──────────────────────────────────────────────────────
    results1:    list[dict] = []
    results2:    list[dict] = []
    skipped      = 0
    t_start      = time.time()
    dot_every    = max(1, args.missions // 20)
    seen_pixels: set[tuple[int, int]] = set()

    print(f"\n{'─'*72}")

    i = 0
    while i < args.missions:
        lat = float(rng.uniform(lat_min, lat_max))
        lon = float(rng.uniform(-180.0, 180.0))

        try:
            terrain = mola.get_terrain_patch(lat, lon, size)
        except Exception:
            skipped += 1
            continue

        try:
            pixel = mola._lat_lon_to_pixel(lat, lon)
        except Exception:
            pixel = (round(lat, 2), round(lon, 2))

        if pixel in seen_pixels:
            skipped += 1
            continue
        seen_pixels.add(pixel)

        found = False
        for _ in range(500):
            sr = int(rng.integers(0, size))
            sc = int(rng.integers(0, size))
            gr = int(rng.integers(0, size))
            gc = int(rng.integers(0, size))
            d  = max(abs(gr - sr), abs(gc - sc))
            if args.min_dist <= d <= max_dist and (sr, sc) != (gr, gc):
                found = True
                break

        if not found:
            skipped += 1
            continue

        start = (sr, sc)
        goal  = (gr, gc)

        _, opt_e = astar(terrain, start, goal)
        if np.isinf(opt_e):
            skipped += 1
            continue

        r1 = run_one(predictor1, env1, start, goal, terrain, lat, lon)
        r1["mission_id"] = i + 1
        results1.append(r1)

        if predictor2 is not None:
            r2 = run_one(predictor2, env2, start, goal, terrain, lat, lon)
            r2["mission_id"] = i + 1
            results2.append(r2)

        i += 1

        if i % dot_every == 0 or i == args.missions:
            n_ok1   = sum(x["success"] for x in results1)
            elapsed = time.time() - t_start
            eta     = elapsed / i * (args.missions - i) if i < args.missions else 0
            eff1    = np.mean([x["efficiency_pct"] for x in results1
                               if x["efficiency_pct"] is not None])
            line    = (f"  [{i:>4}/{args.missions}]  "
                       f"M1 success={n_ok1/i*100:5.1f}%  "
                       f"M1 eff={eff1:5.1f}%")
            if predictor2:
                n_ok2 = sum(x["success"] for x in results2)
                eff2  = np.mean([x["efficiency_pct"] for x in results2
                                 if x["efficiency_pct"] is not None])
                line += (f"  |  M2 success={n_ok2/i*100:5.1f}%  "
                         f"M2 eff={eff2:5.1f}%")
            line += f"  elapsed={elapsed:4.0f}s  eta={eta:4.0f}s"
            print(line, flush=True)

    mola.close()

    if skipped:
        print(f"\n  Skipped {skipped} patches "
              f"(duplicate / unreachable / dist constraint)")
    print(f"  Unique MOLA locations used : {len(seen_pixels)}")

    # ── Individual summaries ──────────────────────────────────────────────
    m1_label = Path(args.model).stem
    print_summary(results1, env1, label=m1_label)

    if comparison_mode:
        m2_label = Path(args.model2).stem
        print_summary(results2, env2, label=m2_label)
        print_head_to_head(results1, results2, env1, m1_label, m2_label)

    # ── CSV export ────────────────────────────────────────────────────────
    if csv_path:
        if comparison_mode:
            export_csv_dual(results1, results2, csv_path, m1_label, m2_label)
        else:
            export_csv(results1, csv_path, label=m1_label)


if __name__ == "__main__":
    main()