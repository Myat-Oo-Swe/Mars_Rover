"""
visualize_mission_3d.py  —  Mars Rover Mission Control + MOLA 3D Terrain
=========================================================================
New in this revision
--------------------
ASTAR-1  astar_optimal_path() — A* search with the exact same 8-direction
         move set, energy costs, and diagonal distance scaling as the rover.
         Treats the terrain as fully observable (oracle view).
         Heuristic: min(dx,dy)×√2 + |dx−dy|×1 — tight and admissible.
ASTAR-2  run_mission() now returns optimal_path, optimal_energy, and
         efficiency_pct alongside the rover's own trajectory.
ASTAR-3  build_plotly_html() accepts both paths — rover in orange (#ff6b35),
         optimal in cyan (#38d9a9) — as separate 3D scatter traces.
ASTAR-4  2-D canvas draws both paths simultaneously with distinct colours.
ASTAR-5  New "Path Comparison" panel shows optimal energy, rover energy,
         efficiency %, extra steps taken, and whether the optimal path
         was even reachable.
ASTAR-6  Legend updated with optimal path colour entry.

Previous changes retained
-------------------------
VEC-1/2/3/4/5  VecNormalize inference, auto-detect, norm badge, terminated
               flag for success, MolaTerrain import from mola_tutorial.
BUG-1  random_patch unpacking fix (terrain, lat, lon) + separate elev call.
"""

from __future__ import annotations

import argparse, json, threading, webbrowser, sys
import heapq
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.mars_rover_env import MarsRoverEnv

try:
    from mola_tutorial import MolaTerrain
    MOLA_PATH = Path(__file__).parent / "mola_dem.tif"
except ImportError:
    MolaTerrain = None
    MOLA_PATH   = None

TERRAIN_NAMES = ["Safe", "Sand", "Rock", "Slope", "Hazard"]

# Exact same tables as MarsRoverEnv
_ENERGY_COST = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4}
_MOVES = [
    (-1,  0, 1.0),   # up
    ( 1,  0, 1.0),   # down
    ( 0, -1, 1.0),   # left
    ( 0,  1, 1.0),   # right
    (-1, -1, 1.4142135623730951),  # up-left
    (-1,  1, 1.4142135623730951),  # up-right
    ( 1, -1, 1.4142135623730951),  # down-left
    ( 1,  1, 1.4142135623730951),  # down-right
]


# ─────────────────────────────────────────────────────────────────────────────
# A* OPTIMAL PATH  [ASTAR-1]
# ─────────────────────────────────────────────────────────────────────────────

def astar_optimal_path(
    terrain: np.ndarray,
    start:   tuple[int, int],
    goal:    tuple[int, int],
) -> tuple[list[list[int]], float]:
    """
    A* search on the terrain grid using the exact rover cost model.

    Cost model
    ----------
    Moving to a cell costs  ENERGY_COST[terrain[r,c]] × dist
    where dist = 1.0 for cardinal moves, √2 for diagonal moves.
    This is identical to MarsRoverEnv.step().

    Heuristic
    ---------
    The tightest admissible heuristic for this 8-direction cost model:
      h(r, c) = min(dx, dy) × √2  +  |dx − dy| × 1
    where dx = |goal_r − r|, dy = |goal_c − c|.
    This assumes every remaining cell has the cheapest terrain (SAFE, cost 1).
    It is admissible because no cell can cost less than 1 per unit distance.

    Returns
    -------
    (path, total_energy_cost)
    path is a list of [row, col] waypoints from start to goal (inclusive).
    If no path exists (fully blocked), returns ([], inf).
    """
    size  = terrain.shape[0]
    sr, sc = start
    gr, gc = goal

    def heuristic(r: int, c: int) -> float:
        dx = abs(gr - r)
        dy = abs(gc - c)
        return min(dx, dy) * 1.4142135623730951 + abs(dx - dy) * 1.0

    # g_cost[r][c] = best known energy cost to reach (r, c)
    g_cost = np.full((size, size), np.inf, dtype=np.float64)
    g_cost[sr, sc] = 0.0

    # came_from[r][c] = parent cell
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}

    # Priority queue: (f_cost, g_cost, row, col)
    # g_cost as tiebreaker keeps the heap stable and slightly favours shorter paths
    heap: list[tuple[float, float, int, int]] = [
        (heuristic(sr, sc), 0.0, sr, sc)
    ]

    while heap:
        f, g, r, c = heapq.heappop(heap)

        # Skip stale entries
        if g > g_cost[r, c] + 1e-9:
            continue

        if (r, c) == (gr, gc):
            break

        for dr, dc, dist in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue

            terrain_type = int(terrain[nr, nc])
            step_cost    = _ENERGY_COST[terrain_type] * dist
            new_g        = g + step_cost

            if new_g < g_cost[nr, nc] - 1e-9:
                g_cost[nr, nc]       = new_g
                came_from[(nr, nc)]  = (r, c)
                heapq.heappush(heap, (new_g + heuristic(nr, nc), new_g, nr, nc))

    # Reconstruct path
    if np.isinf(g_cost[gr, gc]):
        return [], float("inf")   # no path exists

    path   = []
    node: tuple[int, int] | None = (gr, gc)
    while node is not None:
        path.append(list(node))
        node = came_from.get(node)
    path.reverse()

    return path, float(g_cost[gr, gc])


# ─────────────────────────────────────────────────────────────────────────────
# VECNORMALIZE INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class NormalizedPredictor:
    def __init__(self, env: MarsRoverEnv, normalizer_path: str | None):
        self.env             = env
        self._vec_env        = None
        self._vec_normalize  = None
        self._has_normalizer = False

        if normalizer_path and Path(normalizer_path).exists():
            try:
                self._vec_env = DummyVecEnv([lambda: env])
                self._vec_normalize = VecNormalize.load(
                    normalizer_path, self._vec_env
                )
                self._vec_normalize.training    = False
                self._vec_normalize.norm_reward = False
                self._has_normalizer = True
                print(f"  VecNormalize loaded from {normalizer_path}")
            except Exception as exc:
                print(f"  VecNormalize load failed ({exc}) — using raw obs")
        else:
            status = "not found" if normalizer_path else "not specified"
            print(f"  Normalizer {status} — using raw obs")

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._has_normalizer:
            return obs
        return self._vec_normalize.normalize_obs(obs.reshape(1, -1)).reshape(-1)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return _GLOBAL_MODEL.predict(
            self.normalize_obs(obs), deterministic=deterministic
        )

    def close(self):
        if self._vec_normalize:
            self._vec_normalize.close()


_GLOBAL_MODEL: PPO | None = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def run_mission(
    predictor:     NormalizedPredictor,
    env:           MarsRoverEnv,
    start:         tuple,
    goal:          tuple,
    saved_terrain: np.ndarray | None = None,
) -> dict:
    """
    Run one deterministic rover episode AND the A* optimal search.
    Returns both trajectories plus a comparison block.  [ASTAR-2]
    """
    obs, _ = env.reset(options={"start_pos": start, "goal_pos": goal})

    if saved_terrain is not None:
        env.terrain = np.asarray(saved_terrain, dtype=np.int8)
        obs = env._get_obs()

    # ── A* on the full terrain [ASTAR-1] ─────────────────────────────────
    opt_path, opt_energy = astar_optimal_path(env.terrain, start, goal)

    # ── Rover episode ─────────────────────────────────────────────────────
    path         = [list(start)]
    rewards      = []
    energies     = [float(env.energy)]
    actions_log  = []
    total_reward = 0.0
    ACTION_NAMES = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘"]

    done = False
    while not done:
        action, _ = predictor.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        path.append(list(env.pos))
        rewards.append(round(float(reward), 3))
        energies.append(round(float(env.energy), 2))
        actions_log.append(ACTION_NAMES[int(action)])
        done = terminated or truncated

    rover_energy = round(float(env.max_energy - env.energy), 2)

    # Efficiency: how close was the rover to optimal? 100% = matched A*.
    if opt_energy > 0 and not np.isinf(opt_energy):
        efficiency_pct = round(min(100.0, opt_energy / rover_energy * 100), 1) if rover_energy > 0 else 100.0
    else:
        efficiency_pct = None

    return {
        "terrain":        env.terrain.tolist(),
        # rover trajectory
        "path":           path,
        "start":          list(start),
        "goal":           list(goal),
        "size":           env.size,
        "success":        bool(terminated),
        "steps":          int(env.steps),
        "total_reward":   round(total_reward, 2),
        "energy_used":    rover_energy,
        "energy_left":    round(float(env.energy), 2),
        "max_energy":     float(env.max_energy),
        "hazard_hits":    int(env.hazard_hits),
        "rewards":        rewards,
        "energies":       energies,
        "actions":        actions_log,
        # optimal trajectory  [ASTAR-2]
        "optimal_path":        opt_path,
        "optimal_energy":      round(opt_energy, 2) if not np.isinf(opt_energy) else None,
        "optimal_steps":       len(opt_path) - 1 if opt_path else None,
        "optimal_reachable":   len(opt_path) > 0,
        "efficiency_pct":      efficiency_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3D HTML  [ASTAR-3]
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_html(elev, terrain, lat, lon,
                      path=None, optimal_path=None):
    """
    Render an interactive 3D terrain surface with optional mission path
    (orange) and optimal path (cyan) as separate Scatter3D traces.
    """
    size    = elev.shape[0]
    cell_km = 0.463
    x = (np.arange(size) * cell_km).tolist()
    y = (np.arange(size) * cell_km).tolist()

    PLOTLY_COLORS = [
        [0.00, "#1a3320"], [0.20, "#1a3320"],
        [0.20, "#c8a84b"], [0.40, "#c8a84b"],
        [0.40, "#6e6e8a"], [0.60, "#6e6e8a"],
        [0.60, "#8b3a3a"], [0.80, "#8b3a3a"],
        [0.80, "#cc2200"], [1.00, "#cc2200"],
    ]

    counts    = [(terrain == i).sum() for i in range(5)]
    total     = terrain.size
    breakdown = "  |  ".join(
        f"{TERRAIN_NAMES[i]}: {counts[i] / total * 100:.0f}%" for i in range(5)
    )

    extra_traces = ""

    # ── Optimal path (cyan) ───────────────────────────────────────────────
    if optimal_path and len(optimal_path) > 1:
        opx = [p[1] * cell_km for p in optimal_path]
        opy = [p[0] * cell_km for p in optimal_path]
        opz = [float(elev[p[0], p[1]]) + 6 for p in optimal_path]
        extra_traces += f"""
        {{
            type:'scatter3d', x:{opx}, y:{opy}, z:{opz},
            mode:'lines',
            line:{{color:'#38d9a9', width:5}},
            name:'Optimal (A*)',
            hovertemplate:'Optimal — x: %{{x:.1f}} km<br>y: %{{y:.1f}} km<br>elev: %{{z:.0f}} m<extra>Optimal</extra>'
        }},"""

    # ── Rover path (orange) ───────────────────────────────────────────────
    if path and len(path) > 1:
        px_l = [p[1] * cell_km for p in path]
        py_l = [p[0] * cell_km for p in path]
        pz_l = [float(elev[p[0], p[1]]) + 3 for p in path]
        extra_traces += f"""
        {{
            type:'scatter3d', x:{px_l}, y:{py_l}, z:{pz_l},
            mode:'lines+markers',
            line:{{color:'#ff6b35', width:5}},
            marker:{{size:2, color:'#ff6b35', opacity:0.7}},
            name:'Rover (PPO)',
            hovertemplate:'Rover — x: %{{x:.1f}} km<br>y: %{{y:.1f}} km<br>elev: %{{z:.0f}} m<extra>Rover</extra>'
        }},"""

        sp, gp = path[0], path[-1]
        extra_traces += f"""
        {{
            type:'scatter3d',
            x:[{sp[1]*cell_km},{gp[1]*cell_km}],
            y:[{sp[0]*cell_km},{gp[0]*cell_km}],
            z:[{float(elev[sp[0],sp[1]])+12},{float(elev[gp[0],gp[1]])+12}],
            mode:'markers+text',
            marker:{{size:8, color:['#51cf66','#ffc247']}},
            text:['START','GOAL'],
            textposition:'top center',
            textfont:{{color:'#eef2f7', size:11, family:'Share Tech Mono'}},
            name:'Waypoints', hoverinfo:'skip'
        }},"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>MOLA 3D — lat={lat:.2f} lon={lon:.2f}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{{margin:0;background:#080b10;font-family:'Share Tech Mono',monospace;}}
  #plot{{width:100vw;height:100vh;}}
  #info{{position:fixed;top:14px;left:50%;transform:translateX(-50%);
        background:rgba(13,17,23,.9);border:1px solid #1c2535;
        padding:8px 20px;border-radius:4px;color:#c9d4e0;
        font-size:11px;letter-spacing:.1em;text-align:center;z-index:10;pointer-events:none;}}
  #info b{{color:#ff6b35;font-size:13px;}}
  .leg-o{{color:#ff6b35;font-weight:bold;}}
  .leg-a{{color:#38d9a9;font-weight:bold;}}
</style></head>
<body>
<div id="info">
  <b>MOLA MARS TERRAIN — 3D</b><br>
  lat={lat:.2f}°  lon={lon:.2f}°  |  {size}×{size}  |  {size*cell_km:.1f} km²<br>
  {breakdown}<br>
  <span class="leg-o">■ Rover (PPO)</span>  &nbsp;  <span class="leg-a">■ Optimal (A*)</span>
</div>
<div id="plot"></div>
<script>
const data=[
  {{type:'surface',z:{elev.tolist()},x:{x},y:{y},
   surfacecolor:{terrain.astype(float).tolist()},
   colorscale:{PLOTLY_COLORS},cmin:0,cmax:4,
   colorbar:{{title:{{text:'Terrain',font:{{color:'#ff6b35',size:13,family:'Share Tech Mono'}}}},
     tickvals:[0.4,1.2,2.0,2.8,3.6],
     ticktext:['Safe','Sand','Rock','Slope','Hazard'],
     tickfont:{{color:'#c9d4e0',size:11,family:'Share Tech Mono'}},
     thickness:18,len:0.55,bgcolor:'rgba(13,17,23,0.85)',bordercolor:'#2a3a4a',x:1.01}},
   lighting:{{ambient:0.5,diffuse:0.8,roughness:0.5,specular:0.2,fresnel:0.1}},
   lightposition:{{x:2,y:2,z:3}},
   hovertemplate:'x: %{{x:.1f}} km<br>y: %{{y:.1f}} km<br>elevation: %{{z:.0f}} m<extra></extra>',
   showscale:true}},
  {extra_traces}
];
const layout={{
  paper_bgcolor:'#080b10',plot_bgcolor:'#080b10',
  margin:{{l:0,r:0,t:0,b:0}},
  legend:{{bgcolor:'rgba(13,17,23,0.85)',bordercolor:'#1c2535',borderwidth:1,
           font:{{color:'#c9d4e0',family:'Share Tech Mono',size:11}},
           x:0.01,y:0.99,xanchor:'left',yanchor:'top'}},
  scene:{{
    xaxis:{{title:{{text:'km',font:{{color:'#6b7f94'}}}},gridcolor:'#1c2535',
           tickfont:{{color:'#6b7f94',family:'Share Tech Mono'}},backgroundcolor:'#0d1117',showbackground:true}},
    yaxis:{{title:{{text:'km',font:{{color:'#6b7f94'}}}},gridcolor:'#1c2535',
           tickfont:{{color:'#6b7f94',family:'Share Tech Mono'}},backgroundcolor:'#0d1117',showbackground:true}},
    zaxis:{{title:{{text:'elevation (m)',font:{{color:'#6b7f94'}}}},gridcolor:'#1c2535',
           tickfont:{{color:'#6b7f94',family:'Share Tech Mono'}},backgroundcolor:'#080b10',showbackground:true}},
    bgcolor:'#080b10',camera:{{eye:{{x:1.6,y:-1.6,z:1.0}}}},
    aspectmode:'manual',aspectratio:{{x:1,y:1,z:0.4}}}},
  font:{{color:'#c9d4e0',family:'Share Tech Mono'}}}};
Plotly.newPlot('plot',data,layout,{{responsive:true,displaylogo:false,
  modeBarButtonsToRemove:['toImage']}});
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP SERVER
# ─────────────────────────────────────────────────────────────────────────────

def make_handler(predictor, env, mola, fixed_lat, fixed_lon):
    saved = {"terrain": None, "elev": None, "lat": None, "lon": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def _send(self, code, ctype, body):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(b))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = urlparse(self.path).path

            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", HTML)

            elif path == "/terrain" or path.startswith("/terrain/"):
                req_lat, req_lon = None, None
                if path.startswith("/terrain/"):
                    parts = path.split("/")
                    try:
                        req_lat = float(parts[2])
                        req_lon = float(parts[3])
                    except (IndexError, ValueError):
                        pass

                if mola is not None:
                    if req_lat is not None:
                        elev, terrain_arr = mola.get_patch(req_lat, req_lon, env.size)
                        lat, lon = req_lat, req_lon
                    elif fixed_lat is not None:
                        elev, terrain_arr = mola.get_patch(fixed_lat, fixed_lon, env.size)
                        lat, lon = fixed_lat, fixed_lon
                    else:
                        # [BUG-1]  random_patch returns (terrain, lat, lon)
                        terrain_arr, lat, lon = mola.random_patch(env.size)
                        elev = mola.get_elevation_patch(lat, lon, env.size)

                    saved["elev"]    = elev
                    saved["lat"]     = lat
                    saved["lon"]     = lon
                    saved["terrain"] = terrain_arr.copy()
                    env.reset(options={
                        "terrain":   terrain_arr,
                        "start_pos": (0, 0),
                        "goal_pos":  (env.size - 1, env.size - 1),
                    })
                else:
                    env.reset()
                    saved["terrain"] = env.terrain.copy()
                    saved["elev"]    = None
                    saved["lat"]     = None
                    saved["lon"]     = None

                resp = {
                    "terrain":    (saved["terrain"].tolist()
                                   if hasattr(saved["terrain"], "tolist")
                                   else saved["terrain"]),
                    "size":       env.size,
                    "lat":        saved["lat"],
                    "lon":        saved["lon"],
                    "elev":       (saved["elev"].tolist()
                                   if saved["elev"] is not None else None),
                    "normalized": predictor._has_normalizer,
                }
                self._send(200, "application/json",
                           json.dumps(resp, cls=NumpyEncoder))

            elif path == "/view3d":
                if saved["elev"] is None or saved["terrain"] is None:
                    self._send(404, "text/plain",
                               "No terrain loaded — click NEW MAP first.")
                    return
                html = build_plotly_html(
                    saved["elev"], saved["terrain"],
                    saved["lat"] or 0.0, saved["lon"] or 0.0,
                )
                self._send(200, "text/html; charset=utf-8", html)

            else:
                self._send(404, "text/plain", "not found")

        def do_POST(self):
            path   = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))

            if path == "/view3d_mission":
                if saved["elev"] is None:
                    self._send(404, "text/plain", "No MOLA terrain loaded.")
                    return
                # [ASTAR-3]  Pass both paths to 3D renderer
                html = build_plotly_html(
                    saved["elev"], saved["terrain"],
                    saved["lat"] or 0.0, saved["lon"] or 0.0,
                    path         = body.get("path", []),
                    optimal_path = body.get("optimal_path", []),
                )
                self._send(200, "text/html; charset=utf-8", html)
            else:
                start = tuple(body["start"])
                goal  = tuple(body["goal"])
                data  = run_mission(predictor, env, start, goal,
                                    saved["terrain"])
                self._send(200, "application/json",
                           json.dumps(data, cls=NumpyEncoder))

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    return Handler


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mars Rover — Mission Control 3D</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700;800&display=swap');
:root{
  --bg:#080b10;--surface:#0d1117;--card:#111822;--border:#1c2535;
  --accent:#ff6b35;--gold:#ffc247;--teal:#38d9a9;--blue:#4dabf7;
  --red:#ff6b6b;--dim:#3d4f63;--muted:#6b7f94;--text:#c9d4e0;
  --white:#eef2f7;--success:#51cf66;--fail:#ff6b6b;
  --opt:#38d9a9;
  --t0:#1a3320;--t0h:#2d5c38;--t1:#3b2c0d;--t1h:#6b5020;
  --t2:#1e1e30;--t2h:#343452;--t3:#2e1118;--t3h:#5c2535;
  --t4:#380808;--t4h:#6b1515;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Barlow Condensed',sans-serif;font-size:15px;min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(255,107,53,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(255,107,53,.02) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto 1fr;min-height:100vh;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--border);background:rgba(13,17,23,.95);backdrop-filter:blur(8px);gap:12px;flex-wrap:wrap;}
.logo{display:flex;align-items:baseline;gap:10px;}
.logo-mark{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.25em;color:var(--accent);opacity:.75;}
.logo h1{font-size:21px;font-weight:800;letter-spacing:.06em;color:var(--white);}
.topbar-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.mode-toggle{display:flex;background:var(--surface);border:1px solid var(--border);border-radius:4px;overflow:hidden;}
.mode-btn{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600;letter-spacing:.08em;padding:6px 13px;cursor:pointer;background:transparent;border:none;color:var(--muted);transition:background .15s,color .15s;}
.mode-btn.active{background:var(--accent);color:#fff;}
.mode-btn.active.goal-active{background:var(--gold);color:#111;}
.btn{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;letter-spacing:.1em;padding:7px 16px;border-radius:4px;cursor:pointer;text-transform:uppercase;transition:opacity .15s,transform .1s;border:none;}
.btn-run{background:var(--accent);color:#fff;}
.btn-run:hover{opacity:.88;}.btn-run:active{transform:scale(.97);}.btn-run:disabled{opacity:.35;cursor:not-allowed;}
.btn-regen{background:transparent;border:1px solid var(--border);color:var(--muted);}
.btn-regen:hover{border-color:var(--teal);color:var(--teal);}
.btn-3d{background:transparent;border:1px solid #4dabf7;color:#4dabf7;}
.btn-3d:hover{background:rgba(77,171,247,.1);}
.btn-3d-mission{background:transparent;border:1px solid var(--gold);color:var(--gold);display:none;}
.btn-3d-mission:hover{background:rgba(255,194,71,.1);}
.loc-badge{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.1em;color:var(--dim);padding:6px 10px;background:var(--surface);border:1px solid var(--border);border-radius:4px;}
.loc-badge span{color:var(--teal);}
.norm-badge{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.1em;padding:5px 10px;border-radius:4px;border:1px solid;}
.norm-badge.on{background:rgba(81,207,102,.08);color:var(--success);border-color:rgba(81,207,102,.3);}
.norm-badge.off{background:rgba(255,194,71,.06);color:var(--gold);border-color:rgba(255,194,71,.25);}
.loc-select{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.08em;padding:6px 10px;background:var(--surface);border:1px solid var(--border);border-radius:4px;color:var(--text);cursor:pointer;outline:none;appearance:none;-webkit-appearance:none;padding-right:24px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%236b7f94'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 8px center;}
.loc-select:hover{border-color:var(--teal);}.loc-select:focus{border-color:var(--accent);}.loc-select option{background:var(--surface);}
.spinner{display:none;width:15px;height:15px;border-radius:50%;border:2px solid rgba(255,107,53,.2);border-top-color:var(--accent);animation:spin .6s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.main{display:grid;grid-template-columns:auto 1fr;gap:18px;padding:18px 22px;align-items:start;}
@media(max-width:860px){.main{grid-template-columns:1fr;}}
.map-col{display:flex;flex-direction:column;gap:10px;}
.map-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:14px;}
.card-title{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px;}
.card-title::after{content:'';flex:1;height:1px;background:var(--border);}
#map-canvas{display:block;cursor:crosshair;image-rendering:pixelated;border:1px solid var(--border);border-radius:3px;}
#map-canvas:hover{box-shadow:0 0 0 1px rgba(255,107,53,.35);}
.map-hint{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.1em;color:var(--dim);text-align:center;padding:5px 0 2px;}
.map-hint span{color:var(--accent);}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;padding-top:9px;border-top:1px solid var(--border);margin-top:9px;}
.leg{display:flex;align-items:center;gap:4px;font-size:12px;color:var(--muted);}
.leg-sw{width:9px;height:9px;border-radius:2px;flex-shrink:0;}
.coords-bar{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.08em;display:flex;gap:14px;padding:7px 11px;background:var(--surface);border:1px solid var(--border);border-radius:4px;flex-wrap:wrap;}
.coord-item{display:flex;gap:5px;align-items:center;}
.coord-lbl{color:var(--dim);}.coord-val{color:var(--white);font-weight:600;}
.data-col{display:flex;flex-direction:column;gap:12px;}
.status-banner{padding:9px 14px;border-radius:4px;text-align:center;font-size:15px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;display:none;}
.status-banner.success{background:rgba(81,207,102,.08);color:var(--success);border:1px solid rgba(81,207,102,.25);display:block;}
.status-banner.fail{background:rgba(255,107,107,.08);color:var(--fail);border:1px solid rgba(255,107,107,.25);display:block;}
.stat-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:9px;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:5px;padding:11px 13px;position:relative;overflow:hidden;}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat.s-orange::before{background:var(--accent);}.stat.s-teal::before{background:var(--teal);}
.stat.s-gold::before{background:var(--gold);}.stat.s-blue::before{background:var(--blue);}
.stat.s-red::before{background:var(--red);}.stat.s-white::before{background:var(--muted);}
.stat.s-opt::before{background:var(--opt);}
.stat-lbl{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;margin-bottom:4px;}
.stat-val{font-size:22px;font-weight:800;color:var(--white);line-height:1;}
.stat-val.c-orange{color:var(--accent);}.stat-val.c-teal{color:var(--teal);}.stat-val.c-gold{color:var(--gold);}
.stat-val.c-blue{color:var(--blue);}.stat-val.c-red{color:var(--red);}.stat-val.c-success{color:var(--success);}
.stat-val.c-opt{color:var(--opt);}
/* comparison panel */
.cmp-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:14px;}
.cmp-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:8px;}
.cmp-col{display:flex;flex-direction:column;gap:4px;}
.cmp-head{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;padding-bottom:4px;border-bottom:1px solid var(--border);}
.cmp-head.rover{color:var(--accent);}.cmp-head.optimal{color:var(--opt);}.cmp-head.delta{color:var(--gold);}
.cmp-row{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;}
.cmp-lbl{color:var(--muted);}.cmp-val{color:var(--white);font-weight:700;}
.cmp-val.better{color:var(--success);}.cmp-val.worse{color:var(--fail);}
.eff-bar-wrap{margin-top:8px;}
.eff-label{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);display:flex;justify-content:space-between;margin-bottom:4px;}
.eff-track{height:8px;background:var(--border);border-radius:4px;overflow:hidden;}
.eff-fill{height:100%;border-radius:4px;transition:width 1s cubic-bezier(.4,0,.2,1);}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:14px;}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.chart-wrap{position:relative;height:86px;margin-top:6px;}
svg.spark{position:absolute;inset:0;width:100%;height:100%;overflow:visible;}
.placeholder{display:flex;align-items:center;justify-content:center;height:76px;color:var(--dim);font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.1em;}
.tbar{display:flex;flex-direction:column;gap:6px;margin-top:4px;}
.tbar-row{display:flex;align-items:center;gap:8px;}
.tbar-name{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--muted);width:42px;}
.tbar-track{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;}
.tbar-fill{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.4,0,.2,1);}
.tbar-pct{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);width:26px;text-align:right;}
.log-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:14px;}
.step-log{font-family:'Share Tech Mono',monospace;font-size:11px;max-height:190px;overflow-y:auto;line-height:1.9;color:var(--dim);scrollbar-width:thin;scrollbar-color:var(--border) transparent;}
.step-log::-webkit-scrollbar{width:3px;}
.step-log::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.log-row{display:flex;gap:0;}
.l-idx{color:var(--dim);min-width:38px;}.l-act{color:var(--white);min-width:26px;}
.l-rw{min-width:58px;text-align:right;}.l-en{color:var(--teal);padding-left:10px;}
</style>
</head>
<body>
<div class="app">
<div class="topbar">
  <div class="logo">
    <span class="logo-mark">MRS-1 //</span>
    <h1>MISSION CONTROL <span style="color:var(--blue);font-size:14px;font-weight:600;letter-spacing:.15em;">+ 3D</span></h1>
  </div>
  <div class="topbar-right">
    <select class="loc-select" id="loc-select" onchange="onLocationSelect(this.value)">
      <optgroup label="── Named Locations ──">
        <option value="">📍 Choose location...</option>
        <option value="18.4/77.7">Jezero Crater (Perseverance)</option>
        <option value="-5.4/137.8">Gale Crater (Curiosity)</option>
        <option value="18.6/-133.8">Olympus Mons (Volcano)</option>
        <option value="-8.0/-70.0">Valles Marineris (Canyon)</option>
        <option value="24.0/-160.0">Amazonis Planitia (Flat)</option>
        <option value="-42.0/70.0">Hellas Basin (Deepest)</option>
        <option value="18.0/30.0">Arabia Terra</option>
        <option value="10.0/-105.0">Tharsis Plateau</option>
        <option value="-50.0/-43.0">Argyre Planitia</option>
        <option value="25.0/147.0">Elysium Mons</option>
      </optgroup>
    </select>
    <button class="btn btn-regen" onclick="regenRandom()">⟳ RANDOM</button>
    <span class="loc-badge" id="loc-badge">MOLA —</span>
    <span class="norm-badge off" id="norm-badge">NORM ✗</span>
    <div class="mode-toggle">
      <button class="mode-btn active" id="btn-start" onclick="setMode('start')">SET START</button>
      <button class="mode-btn"        id="btn-goal"  onclick="setMode('goal')">SET GOAL</button>
    </div>
    <button class="btn btn-3d" onclick="open3D()">⬡ 3D TERRAIN</button>
    <button class="btn btn-3d-mission" id="btn-3d-mission" onclick="open3DMission()">⬡ 3D + PATHS</button>
    <div class="spinner" id="spinner"></div>
    <button class="btn btn-run" id="run-btn" onclick="runMission()">▶ RUN MISSION</button>
  </div>
</div>

<div class="main">
  <div class="map-col">
    <div class="map-card">
      <div class="card-title">terrain map</div>
      <canvas id="map-canvas"></canvas>
      <div class="map-hint">Click map to place <span id="hint-mode">START</span></div>
      <div class="legend">
        <div class="leg"><div class="leg-sw" style="background:var(--t0h)"></div>Safe</div>
        <div class="leg"><div class="leg-sw" style="background:var(--t1h)"></div>Sand</div>
        <div class="leg"><div class="leg-sw" style="background:var(--t2h)"></div>Rock</div>
        <div class="leg"><div class="leg-sw" style="background:var(--t3h)"></div>Slope</div>
        <div class="leg"><div class="leg-sw" style="background:var(--t4h)"></div>Hazard</div>
        <div class="leg"><div class="leg-sw" style="background:var(--accent);border-radius:50%"></div>Rover</div>
        <div class="leg"><div class="leg-sw" style="background:var(--opt);border-radius:50%"></div>Optimal</div>
        <div class="leg"><div class="leg-sw" style="background:var(--success)"></div>Start</div>
        <div class="leg"><div class="leg-sw" style="background:var(--gold)"></div>Goal</div>
      </div>
    </div>
    <div class="coords-bar">
      <div class="coord-item"><span class="coord-lbl">MODE</span><span class="coord-val" id="disp-mode" style="color:var(--success)">START</span></div>
      <div class="coord-item"><span class="coord-lbl">START</span><span class="coord-val" id="disp-start" style="color:var(--success)">—</span></div>
      <div class="coord-item"><span class="coord-lbl">GOAL</span><span class="coord-val" id="disp-goal" style="color:var(--gold)">—</span></div>
      <div class="coord-item"><span class="coord-lbl">HOVER</span><span class="coord-val" id="disp-hover">—</span></div>
    </div>
  </div>

  <div class="data-col">
    <div id="status-banner" class="status-banner"></div>

    <!-- rover stats -->
    <div class="stat-strip">
      <div class="stat s-orange"><div class="stat-lbl">Steps</div><div class="stat-val c-orange" id="st-steps">—</div></div>
      <div class="stat s-teal"><div class="stat-lbl">Reward</div><div class="stat-val c-teal" id="st-reward">—</div></div>
      <div class="stat s-gold"><div class="stat-lbl">Energy Used</div><div class="stat-val c-gold" id="st-eused">—</div></div>
      <div class="stat s-blue"><div class="stat-lbl">Energy Left</div><div class="stat-val c-blue" id="st-eleft">—</div></div>
      <div class="stat s-red"><div class="stat-lbl">Hazards</div><div class="stat-val c-red" id="st-haz">—</div></div>
      <div class="stat s-white"><div class="stat-lbl">Path Dist</div><div class="stat-val" id="st-dist">—</div></div>
    </div>

    <!-- path comparison panel [ASTAR-5] -->
    <div class="cmp-card" id="cmp-card" style="display:none">
      <div class="card-title">path comparison — rover vs optimal (a*)</div>
      <div class="cmp-grid">
        <div class="cmp-col">
          <div class="cmp-head rover">Rover (PPO)</div>
          <div class="cmp-row"><span class="cmp-lbl">Energy</span><span class="cmp-val" id="cmp-rover-energy">—</span></div>
          <div class="cmp-row"><span class="cmp-lbl">Steps</span><span class="cmp-val" id="cmp-rover-steps">—</span></div>
          <div class="cmp-row"><span class="cmp-lbl">Success</span><span class="cmp-val" id="cmp-rover-success">—</span></div>
        </div>
        <div class="cmp-col">
          <div class="cmp-head optimal">Optimal (A*)</div>
          <div class="cmp-row"><span class="cmp-lbl">Energy</span><span class="cmp-val c-opt" id="cmp-opt-energy">—</span></div>
          <div class="cmp-row"><span class="cmp-lbl">Steps</span><span class="cmp-val c-opt" id="cmp-opt-steps">—</span></div>
          <div class="cmp-row"><span class="cmp-lbl">Reachable</span><span class="cmp-val" id="cmp-opt-reach">—</span></div>
        </div>
        <div class="cmp-col">
          <div class="cmp-head delta">Delta</div>
          <div class="cmp-row"><span class="cmp-lbl">Extra energy</span><span class="cmp-val" id="cmp-delta-energy">—</span></div>
          <div class="cmp-row"><span class="cmp-lbl">Extra steps</span><span class="cmp-val" id="cmp-delta-steps">—</span></div>
          <div class="cmp-row"><span class="cmp-lbl">Efficiency</span><span class="cmp-val" id="cmp-eff">—</span></div>
        </div>
      </div>
      <div class="eff-bar-wrap">
        <div class="eff-label">
          <span>Rover efficiency vs optimal</span>
          <span id="eff-pct-label">—</span>
        </div>
        <div class="eff-track">
          <div class="eff-fill" id="eff-fill" style="width:0%"></div>
        </div>
      </div>
    </div>

    <div class="chart-row">
      <div class="chart-card"><div class="card-title">energy</div>
        <div class="chart-wrap"><svg class="spark" id="svg-energy"><text x="50%" y="50%" text-anchor="middle" fill="#3d4f63" font-family="Share Tech Mono" font-size="10">awaiting mission</text></svg></div>
      </div>
      <div class="chart-card"><div class="card-title">reward / step</div>
        <div class="chart-wrap"><svg class="spark" id="svg-reward"><text x="50%" y="50%" text-anchor="middle" fill="#3d4f63" font-family="Share Tech Mono" font-size="10">awaiting mission</text></svg></div>
      </div>
    </div>
    <div class="chart-card"><div class="card-title">terrain traversed</div>
      <div class="tbar" id="tbar-container"><div class="placeholder">awaiting mission</div></div>
    </div>
    <div class="log-card"><div class="card-title">step log</div>
      <div class="step-log" id="step-log"><div class="placeholder">awaiting mission</div></div>
    </div>
  </div>
</div>

<script>
const T_BASE=['#1a3320','#3b2c0d','#1e1e30','#2e1118','#380808'];
const T_HOVER=['#2d5c38','#6b5020','#343452','#5c2535','#6b1515'];
const T_NAMES=['Safe','Sand','Rock','Slope','Hazard'];
const T_BARS=['#51cf66','#ffc247','#748ffc','#ff8787','#ff6b6b'];

let terrain=null, gridSize=20, CELL=24;
let mode='start', startPos=null, goalPos=null;
let missionPath=[], optimalPath=[], lastMissionData=null;
let hoveredCell=null, currentLat=null, currentLon=null;

const canvas=document.getElementById('map-canvas');
const ctx=canvas.getContext('2d');

fetch('/terrain').then(r=>r.json()).then(d=>{
  terrain=d.terrain; gridSize=d.size; currentLat=d.lat; currentLon=d.lon;
  updateLocBadge(); updateNormBadge(d.normalized); initCanvas(); drawMap();
});

function initCanvas(){CELL=Math.min(Math.floor(520/gridSize),26);canvas.width=CELL*gridSize;canvas.height=CELL*gridSize;}
function updateLocBadge(){
  const el=document.getElementById('loc-badge');
  el.innerHTML=currentLat!==null?`MOLA <span>${currentLat.toFixed(2)}° ${currentLon.toFixed(2)}°</span>`:'MOLA <span>random env</span>';
}
function updateNormBadge(norm){
  const el=document.getElementById('norm-badge');
  el.textContent=norm?'NORM ✓':'NORM ✗'; el.className='norm-badge '+(norm?'on':'off');
}
function _resetState(){
  missionPath=[]; optimalPath=[]; startPos=null; goalPos=null; lastMissionData=null;
  ['disp-start','disp-goal'].forEach(id=>document.getElementById(id).textContent='—');
  document.getElementById('status-banner').className='status-banner';
  document.getElementById('btn-3d-mission').style.display='none';
  document.getElementById('cmp-card').style.display='none';
}
function _loadTerrain(url){
  _resetState();
  fetch(url).then(r=>r.json()).then(d=>{
    terrain=d.terrain; gridSize=d.size; currentLat=d.lat; currentLon=d.lon;
    updateLocBadge(); updateNormBadge(d.normalized); initCanvas(); drawMap();
  });
}
function regenRandom(){document.getElementById('loc-select').value=''; _loadTerrain('/terrain');}
function onLocationSelect(val){if(!val)return; const[lat,lon]=val.split('/'); _loadTerrain(`/terrain/${lat}/${lon}`);}

function open3D(){window.open('/view3d','_blank');}
function open3DMission(){
  if(!lastMissionData) return;
  fetch('/view3d_mission',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      path:         lastMissionData.path,
      optimal_path: lastMissionData.optimal_path || []
    })
  }).then(r=>r.text()).then(html=>{
    const w=window.open('','_blank'); w.document.write(html); w.document.close();
  });
}

// ── DRAW — both paths on the same canvas [ASTAR-4] ──────────────────────────
function drawPath(points, color, alpha_start, alpha_end, dotR){
  if(!points || points.length < 2) return;
  ctx.save();
  ctx.shadowColor=color+'aa'; ctx.shadowBlur=6;
  ctx.beginPath();
  ctx.moveTo(points[0][1]*CELL+CELL/2, points[0][0]*CELL+CELL/2);
  points.forEach(p=>ctx.lineTo(p[1]*CELL+CELL/2, p[0]*CELL+CELL/2));
  ctx.strokeStyle=color+'aa'; ctx.lineWidth=Math.max(1.5,CELL*.11); ctx.lineJoin='round'; ctx.stroke();
  ctx.restore();
  points.forEach((p,i)=>{
    ctx.beginPath();
    ctx.arc(p[1]*CELL+CELL/2, p[0]*CELL+CELL/2, Math.max(1.2,CELL*dotR), 0, Math.PI*2);
    const a = alpha_start + (alpha_end - alpha_start)*(i/points.length);
    ctx.fillStyle=color+Math.round(a*255).toString(16).padStart(2,'0');
    ctx.fill();
  });
}

function drawMap(){
  if(!terrain) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let r=0;r<gridSize;r++) for(let c=0;c<gridSize;c++){
    const t=terrain[r][c], hov=hoveredCell&&hoveredCell[0]===r&&hoveredCell[1]===c;
    ctx.fillStyle=hov?T_HOVER[t]:T_BASE[t]; ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
  }
  ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=0.5;
  for(let i=0;i<=gridSize;i++){
    ctx.beginPath();ctx.moveTo(i*CELL,0);ctx.lineTo(i*CELL,canvas.height);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,i*CELL);ctx.lineTo(canvas.width,i*CELL);ctx.stroke();
  }
  // optimal path (cyan, drawn first so rover overlays it)
  drawPath(optimalPath, '#38d9a9', 0.15, 0.80, 0.13);
  // rover path (orange, on top)
  drawPath(missionPath, '#ff6b35', 0.10, 0.75, 0.17);

  if(startPos) drawMarker(startPos[0],startPos[1],'#51cf66','S','#000');
  if(goalPos)  drawMarker(goalPos[0], goalPos[1], '#ffc247','G','#111');
  if(missionPath.length){
    const fp=missionPath[missionPath.length-1];
    if(!goalPos||fp[0]!==goalPos[0]||fp[1]!==goalPos[1])
      drawMarker(fp[0],fp[1],'#ff6b35','R','#fff');
  }
}
function drawMarker(r,c,bg,label,fg='#000'){
  ctx.fillStyle=bg; ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
  if(CELL>=12){ctx.fillStyle=fg;ctx.font=`bold ${Math.max(9,CELL*.52)}px Barlow Condensed`;
  ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(label,c*CELL+CELL/2,r*CELL+CELL/2+1);}
}

// ── mouse ────────────────────────────────────────────────────────────────────
function cellFrom(e){
  const rect=canvas.getBoundingClientRect(),sx=canvas.width/rect.width,sy=canvas.height/rect.height;
  return[Math.floor((e.clientY-rect.top)*sy/CELL),Math.floor((e.clientX-rect.left)*sx/CELL)];
}
canvas.addEventListener('mousemove',e=>{
  const[r,c]=cellFrom(e); if(r>=0&&r<gridSize&&c>=0&&c<gridSize){
    hoveredCell=[r,c]; document.getElementById('disp-hover').textContent=`(${r},${c})`; drawMap();}
});
canvas.addEventListener('mouseleave',()=>{hoveredCell=null;document.getElementById('disp-hover').textContent='—';drawMap();});
canvas.addEventListener('click',e=>{
  const[r,c]=cellFrom(e); if(r<0||r>=gridSize||c<0||c>=gridSize) return;
  if(mode==='start'){startPos=[r,c];document.getElementById('disp-start').textContent=`(${r},${c})`;setMode('goal');}
  else{goalPos=[r,c];document.getElementById('disp-goal').textContent=`(${r},${c})`;setMode('start');}
  drawMap();
});
function setMode(m){
  mode=m;
  document.getElementById('btn-start').className='mode-btn'+(m==='start'?' active':'');
  document.getElementById('btn-goal').className='mode-btn'+(m==='goal'?' active goal-active':'');
  const hint=m==='start'?'START':'GOAL';
  document.getElementById('hint-mode').textContent=hint;
  document.getElementById('hint-mode').style.color=m==='start'?'var(--success)':'var(--gold)';
  document.getElementById('disp-mode').textContent=hint;
  document.getElementById('disp-mode').style.color=m==='start'?'var(--success)':'var(--gold)';
}

// ── run mission ───────────────────────────────────────────────────────────────
async function runMission(){
  if(!startPos||!goalPos){alert('Place both START and GOAL first.');return;}
  const btn=document.getElementById('run-btn'),spin=document.getElementById('spinner');
  btn.disabled=true; spin.style.display='block';
  document.getElementById('status-banner').className='status-banner';
  try{
    const res=await fetch('/',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({start:startPos,goal:goalPos})});
    const d=await res.json();
    terrain=d.terrain; missionPath=d.path; optimalPath=d.optimal_path||[];
    lastMissionData=d;
    drawMap(); renderResults(d);
    document.getElementById('btn-3d-mission').style.display='block';
  }catch(err){alert('Server error: '+err);}
  finally{btn.disabled=false;spin.style.display='none';}
}

// ── render results ────────────────────────────────────────────────────────────
function renderResults(d){
  const banner=document.getElementById('status-banner');
  banner.textContent=d.success?'✓  MISSION SUCCESS':'✗  MISSION FAILED — energy or step limit reached';
  banner.className='status-banner '+(d.success?'success':'fail');

  document.getElementById('st-steps').textContent=d.steps;
  document.getElementById('st-eused').textContent=d.energy_used+' / '+d.max_energy;
  document.getElementById('st-eleft').textContent=d.energy_left;
  document.getElementById('st-haz').textContent=d.hazard_hits;
  let pl=0;
  for(let i=1;i<d.path.length;i++){const dx=d.path[i][0]-d.path[i-1][0],dy=d.path[i][1]-d.path[i-1][1];pl+=Math.sqrt(dx*dx+dy*dy);}
  document.getElementById('st-dist').textContent=pl.toFixed(1);
  const rEl=document.getElementById('st-reward');
  rEl.textContent=(d.total_reward>=0?'+':'')+d.total_reward;
  rEl.className='stat-val '+(d.total_reward>=100?'c-success':d.total_reward<-20?'c-red':'c-teal');

  // comparison panel [ASTAR-5]
  const cmpCard=document.getElementById('cmp-card');
  if(d.optimal_reachable!==undefined){
    cmpCard.style.display='block';
    document.getElementById('cmp-rover-energy').textContent=d.energy_used;
    document.getElementById('cmp-rover-steps').textContent=d.steps;
    document.getElementById('cmp-rover-success').textContent=d.success?'✓':'✗';

    const optE = d.optimal_energy!==null ? d.optimal_energy : '∞';
    const optS = d.optimal_steps!==null  ? d.optimal_steps  : '—';
    document.getElementById('cmp-opt-energy').textContent=optE;
    document.getElementById('cmp-opt-steps').textContent=optS;
    document.getElementById('cmp-opt-reach').textContent=d.optimal_reachable?'✓':'✗';

    if(d.optimal_energy!==null && d.optimal_reachable){
      const dE=(d.energy_used - d.optimal_energy).toFixed(2);
      const dS=(d.steps - d.optimal_steps);
      const eEl=document.getElementById('cmp-delta-energy');
      const sEl=document.getElementById('cmp-delta-steps');
      eEl.textContent=(dE>=0?'+':'')+dE;
      eEl.className='cmp-val '+(dE<=0?'better':'worse');
      sEl.textContent=(dS>=0?'+':'')+dS;
      sEl.className='cmp-val '+(dS<=0?'better':'worse');
    } else {
      document.getElementById('cmp-delta-energy').textContent='—';
      document.getElementById('cmp-delta-steps').textContent='—';
    }

    // efficiency bar
    const eff = d.efficiency_pct;
    const effEl=document.getElementById('cmp-eff');
    const effFill=document.getElementById('eff-fill');
    const effLabel=document.getElementById('eff-pct-label');
    if(eff!==null){
      effEl.textContent=eff+'%';
      effEl.className='cmp-val '+(eff>=90?'better':eff>=60?'':'worse');
      effLabel.textContent=eff+'%';
      const barColor = eff>=90?'#51cf66':eff>=60?'#ffc247':'#ff6b6b';
      effFill.style.background=barColor;
      requestAnimationFrame(()=>effFill.style.width=Math.min(100,eff)+'%');
    } else {
      effEl.textContent='—'; effLabel.textContent='—'; effFill.style.width='0%';
    }
  }

  sparkline('svg-energy',d.energies,'#38d9a9','rgba(56,217,169,0.12)',0,d.max_energy);
  sparkline('svg-reward',d.rewards,'#ff6b35','rgba(255,107,53,0.12)',null,null);

  const counts=[0,0,0,0,0];
  d.path.forEach(p=>counts[d.terrain[p[0]][p[1]]]++);
  const total=d.path.length||1;
  const tc=document.getElementById('tbar-container'); tc.innerHTML='';
  counts.forEach((c,i)=>{
    const pct=(c/total*100).toFixed(0);
    tc.innerHTML+=`<div class="tbar-row"><div class="tbar-name">${T_NAMES[i]}</div><div class="tbar-track"><div class="tbar-fill" style="width:0%;background:${T_BARS[i]}" data-w="${pct}"></div></div><div class="tbar-pct">${c}</div></div>`;
  });
  requestAnimationFrame(()=>document.querySelectorAll('.tbar-fill').forEach(el=>el.style.width=el.dataset.w+'%'));

  let html='';
  for(let i=0;i<d.actions.length;i++){
    const rw=d.rewards[i]; const rwc=rw>0.5?'#51cf66':rw<-2?'#ff6b6b':'#ffc247';
    html+=`<div class="log-row"><span class="l-idx">#${String(i+1).padStart(3,'0')}</span><span class="l-act">${d.actions[i]}</span><span class="l-rw" style="color:${rwc}">${rw>=0?'+':''}${rw}</span><span class="l-en">⚡${d.energies[i+1]}</span></div>`;
  }
  const log=document.getElementById('step-log'); log.innerHTML=html; log.scrollTop=log.scrollHeight;
}

function sparkline(svgId,data,stroke,fill,yMin,yMax){
  const el=document.getElementById(svgId); const W=el.parentElement.clientWidth||300,H=86;
  el.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const mn=yMin!==null?yMin:Math.min(...data),mx=yMax!==null?yMax:Math.max(...data);
  const range=(mx-mn)||1;
  const px=i=>(i/(data.length-1||1))*(W-24)+12; const py=v=>H-10-((v-mn)/range)*(H-22);
  let svg=`<defs><linearGradient id="grad-${svgId}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="${stroke}" stop-opacity=".5"/>
    <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
  </linearGradient>
  <filter id="glow-${svgId}"><feGaussianBlur stdDeviation="2.5" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>`;
  if(mn<0&&mx>0){const z=py(0);svg+=`<line x1="12" y1="${z}" x2="${W-12}" y2="${z}" stroke="rgba(255,255,255,0.1)" stroke-width="1" stroke-dasharray="4 3"/>`;}
  let area=`M ${px(0)} ${H-10}`; data.forEach((v,i)=>area+=` L ${px(i)} ${py(v)}`); area+=` L ${px(data.length-1)} ${H-10} Z`;
  svg+=`<path d="${area}" fill="url(#grad-${svgId})"/>`;
  const line=data.map((v,i)=>`${i===0?'M':'L'} ${px(i)} ${py(v)}`).join(' ');
  svg+=`<path d="${line}" fill="none" stroke="${stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow-${svgId})"/>`;
  svg+=`<text x="4" y="${H-8}" fill="#3d4f63" font-size="9" font-family="Share Tech Mono">${mn.toFixed(0)}</text>`;
  svg+=`<text x="4" y="11" fill="#3d4f63" font-size="9" font-family="Share Tech Mono">${mx.toFixed(0)}</text>`;
  const minI=data.indexOf(Math.min(...data)),maxI=data.indexOf(Math.max(...data));
  [[minI,'#ff6b6b'],[maxI,stroke]].forEach(([idx,c])=>{
    const cx=px(idx),cy=py(data[idx]);
    svg+=`<circle cx="${cx}" cy="${cy}" r="4" fill="${c}" opacity=".9" filter="url(#glow-${svgId})"/>`;
    svg+=`<circle cx="${cx}" cy="${cy}" r="2" fill="#fff" opacity=".8"/>`;
  });
  el.innerHTML=svg;
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _find_normalizer(model_arg: str) -> str | None:
    for p in [Path(model_arg).parent / "vec_normalize.pkl",
              Path("vec_normalize.pkl")]:
        if p.exists():
            return str(p)
    return None


def main():
    global _GLOBAL_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="mars_rover_ppo")
    parser.add_argument("--normalizer", default=None)
    parser.add_argument("--port",  type=int,   default=8765)
    parser.add_argument("--lat",   type=float, default=None)
    parser.add_argument("--lon",   type=float, default=None)
    parser.add_argument("--size",  type=int,   default=20)
    args = parser.parse_args()

    print(f"\n  Loading model: {args.model}")
    _GLOBAL_MODEL = PPO.load(args.model)

    norm_path = args.normalizer or _find_normalizer(args.model)
    env       = MarsRoverEnv(size=args.size)
    predictor = NormalizedPredictor(env, norm_path)

    mola        = None
    terrain_src = "random synthetic"
    if MolaTerrain is not None and MOLA_PATH and MOLA_PATH.exists():
        try:
            mola        = MolaTerrain()
            terrain_src = "MOLA real Mars terrain"
            if args.lat is not None:
                terrain_src = f"MOLA fixed lat={args.lat} lon={args.lon}"
        except Exception as exc:
            print(f"  MOLA load failed ({exc}) — using random terrain")
    else:
        print("  mola_dem.tif not found — using random terrain")
        print("  To use real Mars data: python mola_tutorial.py --download")

    fixed_lat = args.lat
    fixed_lon = args.lon if args.lat is not None else None

    handler = make_handler(predictor, env, mola, fixed_lat, fixed_lon)
    server  = HTTPServer(("127.0.0.1", args.port), handler)
    url     = f"http://127.0.0.1:{args.port}"

    print(f"\n  Mars Rover Mission Control 3D")
    print(f"  ─────────────────────────────")
    print(f"  URL        : {url}")
    print(f"  Terrain    : {terrain_src}")
    print(f"  Normalizer : "
          f"{'loaded — ' + norm_path if predictor._has_normalizer else 'not found (raw obs)'}")
    print(f"  A* search  : enabled (same cost model as rover)")
    if mola:
        print(f"  3D view    : {url}/view3d  (opens via button)")
    print(f"  Ctrl+C to stop.\n")

    threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        predictor.close()
        if mola:
            mola.close()


if __name__ == "__main__":
    main()