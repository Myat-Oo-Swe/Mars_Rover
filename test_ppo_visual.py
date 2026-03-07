"""
visualize_mission.py  —  Interactive Mars Rover Mission Control
==============================================================
Starts a local HTTP server and opens the viewer in your browser.
Click any cell on the map to set Start (green) or Goal (gold),
then hit RUN MISSION to execute the PPO model live.

Usage
-----
  python visualize_mission.py
  python visualize_mission.py --model mars_rover_ppo_best
  python visualize_mission.py --port 8765
"""

import argparse
import json
import threading
import webbrowser
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from env.mars_rover_env import MarsRoverEnv
from stable_baselines3 import PPO


# ─────────────────────────────────────────────────────────────────────────────
# MISSION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def run_mission(model, env, start, goal, saved_terrain=None):
    obs, _ = env.reset(options={"start_pos": start, "goal_pos": goal})
    # Restore the terrain the user was looking at so the mission runs on the
    # same map that was displayed — without this, reset() randomises terrain again.
    if saved_terrain is not None:
        env.terrain = saved_terrain.copy()
        # Re-fetch obs with the correct terrain in the local view
        obs = env._get_obs()
    terrain      = env.terrain.tolist()
    path         = [list(start)]
    rewards      = []
    energies     = [float(env.energy)]
    actions      = []
    total_reward = 0.0
    ACTION_NAMES = ["↑","↓","←","→","↖","↗","↙","↘"]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        path.append(list(env.pos))
        rewards.append(round(float(reward), 3))
        energies.append(round(float(env.energy), 2))
        actions.append(ACTION_NAMES[int(action)])
        done = terminated or truncated

    return {
        "terrain":      terrain,
        "path":         path,
        "start":        list(start),
        "goal":         list(goal),
        "size":         env.size,
        "success":      bool(np.array_equal(env.pos, env.goal)),
        "steps":        int(env.steps),
        "total_reward": round(total_reward, 2),
        "energy_used":  round(float(env.max_energy - env.energy), 2),
        "energy_left":  round(float(env.energy), 2),
        "max_energy":   float(env.max_energy),
        "hazard_hits":  int(env.hazard_hits),
        "rewards":      rewards,
        "energies":     energies,
        "actions":      actions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP SERVER
# ─────────────────────────────────────────────────────────────────────────────

def make_handler(model, env):
    # Holds the terrain that was last sent to the browser so every mission
    # POST runs on exactly the same map the user is looking at.
    saved = {"terrain": None}

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
            elif path == "/terrain":
                env.reset()
                saved["terrain"] = env.terrain.copy()   # snapshot for later missions
                self._send(200, "application/json",
                           json.dumps({"terrain": env.terrain.tolist(),
                                       "size": env.size}, cls=NumpyEncoder))
            else:
                self._send(404, "text/plain", "not found")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            start  = tuple(body["start"])
            goal   = tuple(body["goal"])
            data   = run_mission(model, env, start, goal, saved["terrain"])
            self._send(200, "application/json", json.dumps(data, cls=NumpyEncoder))

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    return Handler


# ─────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mars Rover — Mission Control</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700;800&display=swap');

:root {
  --bg:      #080b10;
  --surface: #0d1117;
  --card:    #111822;
  --border:  #1c2535;
  --accent:  #ff6b35;
  --gold:    #ffc247;
  --teal:    #38d9a9;
  --blue:    #4dabf7;
  --red:     #ff6b6b;
  --dim:     #3d4f63;
  --muted:   #6b7f94;
  --text:    #c9d4e0;
  --white:   #eef2f7;
  --success: #51cf66;
  --fail:    #ff6b6b;
  --t0: #1a3320; --t0h: #256335;
  --t1: #3b2c0d; --t1h: #5c4515;
  --t2: #1e1e30; --t2h: #2e2e4a;
  --t3: #2e1118; --t3h: #4a1c28;
  --t4: #380808; --t4h: #5c0f0f;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Barlow Condensed',sans-serif;font-size:15px;min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(255,107,53,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,107,53,.025) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto 1fr;min-height:100vh;}

/* TOPBAR */
.topbar{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--border);background:rgba(13,17,23,.92);backdrop-filter:blur(8px);gap:16px;flex-wrap:wrap;}
.logo{display:flex;align-items:baseline;gap:10px;}
.logo-mark{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.25em;color:var(--accent);opacity:.8;}
.logo h1{font-size:22px;font-weight:800;letter-spacing:.06em;color:var(--white);}
.topbar-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.mode-toggle{display:flex;background:var(--surface);border:1px solid var(--border);border-radius:4px;overflow:hidden;}
.mode-btn{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600;letter-spacing:.08em;padding:6px 14px;cursor:pointer;background:transparent;border:none;color:var(--muted);transition:background .15s,color .15s;}
.mode-btn.active{background:var(--accent);color:#fff;}
.mode-btn.active.goal-active{background:var(--gold);color:#111;}
.run-btn{font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;letter-spacing:.12em;padding:8px 22px;background:var(--accent);color:#fff;border:none;border-radius:4px;cursor:pointer;transition:opacity .15s,transform .1s;text-transform:uppercase;}
.run-btn:hover{opacity:.88;}
.run-btn:active{transform:scale(.97);}
.run-btn:disabled{opacity:.35;cursor:not-allowed;}
.regen-btn{font-family:'Share Tech Mono',monospace;font-size:11px;padding:8px 14px;background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:4px;cursor:pointer;letter-spacing:.1em;transition:border-color .15s,color .15s;}
.regen-btn:hover{border-color:var(--teal);color:var(--teal);}

/* LAYOUT */
.main{display:grid;grid-template-columns:auto 1fr;gap:20px;padding:20px 24px;align-items:start;}
@media(max-width:860px){.main{grid-template-columns:1fr;}}

/* MAP COLUMN */
.map-col{display:flex;flex-direction:column;gap:12px;}
.map-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:16px;}
.card-title{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.card-title::after{content:'';flex:1;height:1px;background:var(--border);}
#map-canvas{display:block;cursor:crosshair;image-rendering:pixelated;border:1px solid var(--border);border-radius:3px;transition:box-shadow .2s;}
#map-canvas:hover{box-shadow:0 0 0 1px rgba(255,107,53,.4);}
.map-hint{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.12em;color:var(--dim);text-align:center;padding:6px 0 2px;}
.map-hint span{color:var(--accent);}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;padding-top:10px;border-top:1px solid var(--border);margin-top:10px;}
.leg{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted);}
.leg-sw{width:10px;height:10px;border-radius:2px;flex-shrink:0;}
.coords-bar{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.1em;display:flex;gap:18px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px;flex-wrap:wrap;}
.coord-item{display:flex;gap:6px;align-items:center;}
.coord-lbl{color:var(--dim);}
.coord-val{color:var(--white);font-weight:600;}

/* DATA COLUMN */
.data-col{display:flex;flex-direction:column;gap:14px;}
.status-banner{padding:10px 16px;border-radius:4px;text-align:center;font-size:16px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;display:none;}
.status-banner.success{background:rgba(81,207,102,.1);color:var(--success);border:1px solid rgba(81,207,102,.3);display:block;}
.status-banner.fail{background:rgba(255,107,107,.1);color:var(--fail);border:1px solid rgba(255,107,107,.3);display:block;}
.stat-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:5px;padding:12px 14px;position:relative;overflow:hidden;}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat.s-orange::before{background:var(--accent);}
.stat.s-teal::before{background:var(--teal);}
.stat.s-gold::before{background:var(--gold);}
.stat.s-blue::before{background:var(--blue);}
.stat.s-red::before{background:var(--red);}
.stat.s-white::before{background:var(--muted);}
.stat-lbl{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;margin-bottom:4px;}
.stat-val{font-size:24px;font-weight:800;color:var(--white);line-height:1;}
.stat-val.c-orange{color:var(--accent);}
.stat-val.c-teal{color:var(--teal);}
.stat-val.c-gold{color:var(--gold);}
.stat-val.c-blue{color:var(--blue);}
.stat-val.c-red{color:var(--red);}
.stat-val.c-success{color:var(--success);}

/* CHARTS */
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:16px;}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.chart-wrap{position:relative;height:90px;margin-top:6px;}
svg.spark{position:absolute;inset:0;width:100%;height:100%;overflow:visible;}
.placeholder{display:flex;align-items:center;justify-content:center;height:80px;color:var(--dim);font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:.1em;}

/* TERRAIN BARS */
.tbar{display:flex;flex-direction:column;gap:7px;margin-top:4px;}
.tbar-row{display:flex;align-items:center;gap:8px;}
.tbar-name{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--muted);width:44px;}
.tbar-track{flex:1;height:7px;background:var(--border);border-radius:3px;overflow:hidden;}
.tbar-fill{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.4,0,.2,1);}
.tbar-pct{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);width:28px;text-align:right;}

/* LOG */
.log-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:16px;}
.step-log{font-family:'Share Tech Mono',monospace;font-size:11px;max-height:200px;overflow-y:auto;line-height:1.9;color:var(--dim);scrollbar-width:thin;scrollbar-color:var(--border) transparent;}
.step-log::-webkit-scrollbar{width:3px;}
.step-log::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.log-row{display:flex;gap:0;}
.l-idx{color:var(--dim);min-width:38px;}
.l-act{color:var(--white);min-width:26px;}
.l-rw{min-width:58px;text-align:right;}
.l-en{color:var(--teal);padding-left:10px;}

.spinner{display:none;width:16px;height:16px;border-radius:50%;border:2px solid rgba(255,107,53,.2);border-top-color:var(--accent);animation:spin .6s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="app">

<div class="topbar">
  <div class="logo">
    <span class="logo-mark">MRS-1 //</span>
    <h1>MISSION CONTROL</h1>
  </div>
  <div class="topbar-right">
    <div class="mode-toggle">
      <button class="mode-btn active" id="btn-start" onclick="setMode('start')">SET START</button>
      <button class="mode-btn"        id="btn-goal"  onclick="setMode('goal')">SET GOAL</button>
    </div>
    <button class="regen-btn" onclick="regenTerrain()">⟳ NEW MAP</button>
    <div class="spinner" id="spinner"></div>
    <button class="run-btn" id="run-btn" onclick="runMission()">▶ RUN MISSION</button>
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
        <div class="leg"><div class="leg-sw" style="background:var(--accent);border-radius:50%"></div>Path</div>
        <div class="leg"><div class="leg-sw" style="background:var(--success)"></div>Start</div>
        <div class="leg"><div class="leg-sw" style="background:var(--gold)"></div>Goal</div>
      </div>
    </div>
    <div class="coords-bar">
      <div class="coord-item"><span class="coord-lbl">MODE</span><span class="coord-val" id="disp-mode" style="color:var(--success)">START</span></div>
      <div class="coord-item"><span class="coord-lbl">START</span><span class="coord-val" id="disp-start" style="color:var(--success)">—</span></div>
      <div class="coord-item"><span class="coord-lbl">GOAL</span><span class="coord-val"  id="disp-goal"  style="color:var(--gold)">—</span></div>
      <div class="coord-item"><span class="coord-lbl">HOVER</span><span class="coord-val" id="disp-hover">—</span></div>
    </div>
  </div>

  <div class="data-col">
    <div id="status-banner" class="status-banner"></div>

    <div class="stat-strip">
      <div class="stat s-orange"><div class="stat-lbl">Steps</div>      <div class="stat-val c-orange" id="st-steps">—</div></div>
      <div class="stat s-teal">  <div class="stat-lbl">Reward</div>     <div class="stat-val c-teal"   id="st-reward">—</div></div>
      <div class="stat s-gold">  <div class="stat-lbl">Energy Used</div><div class="stat-val c-gold"   id="st-eused">—</div></div>
      <div class="stat s-blue">  <div class="stat-lbl">Energy Left</div><div class="stat-val c-blue"   id="st-eleft">—</div></div>
      <div class="stat s-red">   <div class="stat-lbl">Hazards</div>    <div class="stat-val c-red"    id="st-haz">—</div></div>
      <div class="stat s-white"> <div class="stat-lbl">Path Dist</div>  <div class="stat-val"          id="st-dist">—</div></div>
    </div>

    <div class="chart-row">
      <div class="chart-card">
        <div class="card-title">energy</div>
        <div class="chart-wrap"><svg class="spark" id="svg-energy"><text x="50%" y="50%" text-anchor="middle" fill="#3d4f63" font-family="Share Tech Mono" font-size="10">awaiting mission</text></svg></div>
      </div>
      <div class="chart-card">
        <div class="card-title">reward / step</div>
        <div class="chart-wrap"><svg class="spark" id="svg-reward"><text x="50%" y="50%" text-anchor="middle" fill="#3d4f63" font-family="Share Tech Mono" font-size="10">awaiting mission</text></svg></div>
      </div>
    </div>

    <div class="chart-card">
      <div class="card-title">terrain traversed</div>
      <div class="tbar" id="tbar-container"><div class="placeholder">awaiting mission</div></div>
    </div>

    <div class="log-card">
      <div class="card-title">step log</div>
      <div class="step-log" id="step-log"><div class="placeholder">awaiting mission</div></div>
    </div>
  </div>
</div>

<script>
const T_BASE  = ['#1a3320','#3b2c0d','#1e1e30','#2e1118','#380808'];
const T_HOVER = ['#2d5c38','#6b5020','#343452','#5c2535','#6b1515'];
const T_NAMES = ['Safe','Sand','Rock','Slope','Hazard'];
const T_BARS  = ['#51cf66','#ffc247','#748ffc','#ff8787','#ff6b6b'];

let terrain=null, gridSize=20, CELL=24;
let mode='start', startPos=null, goalPos=null, missionPath=[];
let hoveredCell=null;

const canvas=document.getElementById('map-canvas');
const ctx=canvas.getContext('2d');

// ── INIT ──────────────────────────────────────────────────────────────────────
fetch('/terrain').then(r=>r.json()).then(d=>{terrain=d.terrain;gridSize=d.size;initCanvas();drawMap();});

function initCanvas(){
  CELL=Math.min(Math.floor(520/gridSize),26);
  canvas.width=CELL*gridSize; canvas.height=CELL*gridSize;
}

function regenTerrain(){
  missionPath=[];startPos=null;goalPos=null;
  ['disp-start','disp-goal'].forEach(id=>document.getElementById(id).textContent='—');
  document.getElementById('status-banner').className='status-banner';
  fetch('/terrain').then(r=>r.json()).then(d=>{terrain=d.terrain;gridSize=d.size;initCanvas();drawMap();});
}

// ── DRAW ──────────────────────────────────────────────────────────────────────
function drawMap(){
  if(!terrain) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // terrain cells
  for(let r=0;r<gridSize;r++){
    for(let c=0;c<gridSize;c++){
      const t=terrain[r][c];
      const hov=hoveredCell&&hoveredCell[0]===r&&hoveredCell[1]===c;
      ctx.fillStyle=hov?T_HOVER[t]:T_BASE[t];
      ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
    }
  }

  // grid
  ctx.strokeStyle='rgba(255,255,255,0.045)';ctx.lineWidth=0.5;
  for(let i=0;i<=gridSize;i++){
    ctx.beginPath();ctx.moveTo(i*CELL,0);ctx.lineTo(i*CELL,canvas.height);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,i*CELL);ctx.lineTo(canvas.width,i*CELL);ctx.stroke();
  }

  // path
  if(missionPath.length>1){
    // glow
    ctx.save();
    ctx.shadowColor='rgba(255,107,53,.7)';ctx.shadowBlur=8;
    ctx.beginPath();
    ctx.moveTo(missionPath[0][1]*CELL+CELL/2,missionPath[0][0]*CELL+CELL/2);
    missionPath.forEach(p=>ctx.lineTo(p[1]*CELL+CELL/2,p[0]*CELL+CELL/2));
    ctx.strokeStyle='rgba(255,107,53,.65)';ctx.lineWidth=Math.max(1.5,CELL*.13);ctx.lineJoin='round';ctx.stroke();
    ctx.restore();
    // dots
    missionPath.forEach((p,i)=>{
      const a=0.08+0.72*(i/missionPath.length);
      ctx.beginPath();ctx.arc(p[1]*CELL+CELL/2,p[0]*CELL+CELL/2,Math.max(1.5,CELL*.17),0,Math.PI*2);
      ctx.fillStyle=`rgba(255,107,53,${a})`;ctx.fill();
    });
  }

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
  if(CELL>=12){
    ctx.fillStyle=fg;
    ctx.font=`bold ${Math.max(9,CELL*.52)}px Barlow Condensed`;
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(label,c*CELL+CELL/2,r*CELL+CELL/2+1);
  }
}

// ── MOUSE ─────────────────────────────────────────────────────────────────────
function cellFrom(e){
  const rect=canvas.getBoundingClientRect();
  const sx=canvas.width/rect.width, sy=canvas.height/rect.height;
  return [Math.floor((e.clientY-rect.top)*sy/CELL),Math.floor((e.clientX-rect.left)*sx/CELL)];
}
canvas.addEventListener('mousemove',e=>{
  const [r,c]=cellFrom(e);
  if(r>=0&&r<gridSize&&c>=0&&c<gridSize){
    hoveredCell=[r,c];
    document.getElementById('disp-hover').textContent=`(${r},${c})`;
    drawMap();
  }
});
canvas.addEventListener('mouseleave',()=>{hoveredCell=null;document.getElementById('disp-hover').textContent='—';drawMap();});
canvas.addEventListener('click',e=>{
  const [r,c]=cellFrom(e);
  if(r<0||r>=gridSize||c<0||c>=gridSize) return;
  if(mode==='start'){ startPos=[r,c]; document.getElementById('disp-start').textContent=`(${r},${c})`; setMode('goal'); }
  else              { goalPos=[r,c];  document.getElementById('disp-goal').textContent=`(${r},${c})`;  setMode('start'); }
  drawMap();
});

function setMode(m){
  mode=m;
  document.getElementById('btn-start').className='mode-btn'+(m==='start'?' active':'');
  document.getElementById('btn-goal').className ='mode-btn'+(m==='goal'?' active goal-active':'');
  const hint=m==='start'?'START':'GOAL';
  document.getElementById('hint-mode').textContent=hint;
  document.getElementById('hint-mode').style.color=m==='start'?'var(--success)':'var(--gold)';
  document.getElementById('disp-mode').textContent=hint;
  document.getElementById('disp-mode').style.color=m==='start'?'var(--success)':'var(--gold)';
}

// ── RUN ───────────────────────────────────────────────────────────────────────
async function runMission(){
  if(!startPos||!goalPos){alert('Place both START and GOAL on the map first.');return;}
  const btn=document.getElementById('run-btn'), spin=document.getElementById('spinner');
  btn.disabled=true; spin.style.display='block';
  document.getElementById('status-banner').className='status-banner';
  try{
    const res=await fetch('/',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start:startPos,goal:goalPos})});
    const d=await res.json();
    // update terrain with the one used for this mission
    terrain=d.terrain;
    missionPath=d.path;
    drawMap();
    renderResults(d);
  }catch(err){alert('Server error: '+err);}
  finally{btn.disabled=false;spin.style.display='none';}
}

// ── RESULTS ───────────────────────────────────────────────────────────────────
function renderResults(d){
  const banner=document.getElementById('status-banner');
  banner.textContent=d.success?'✓  MISSION SUCCESS':'✗  MISSION FAILED — energy or step limit reached';
  banner.className='status-banner '+(d.success?'success':'fail');

  document.getElementById('st-steps').textContent = d.steps;
  document.getElementById('st-eused').textContent = d.energy_used+' / '+d.max_energy;
  document.getElementById('st-eleft').textContent = d.energy_left;
  document.getElementById('st-haz').textContent   = d.hazard_hits;
  let pl=0;
  for(let i=1;i<d.path.length;i++){const dx=d.path[i][0]-d.path[i-1][0],dy=d.path[i][1]-d.path[i-1][1];pl+=Math.sqrt(dx*dx+dy*dy);}
  document.getElementById('st-dist').textContent=pl.toFixed(1);
  const rEl=document.getElementById('st-reward');
  rEl.textContent=(d.total_reward>=0?'+':'')+d.total_reward;
  rEl.className='stat-val '+(d.total_reward>=100?'c-success':d.total_reward<-20?'c-red':'c-teal');

  sparkline('svg-energy',d.energies,'#38d9a9','rgba(56,217,169,0.12)',0,d.max_energy);
  sparkline('svg-reward',d.rewards, '#ff6b35','rgba(255,107,53,0.12)',null,null);

  // terrain bars
  const counts=[0,0,0,0,0];
  d.path.forEach(p=>counts[d.terrain[p[0]][p[1]]]++);
  const total=d.path.length||1;
  const tc=document.getElementById('tbar-container');
  tc.innerHTML='';
  counts.forEach((c,i)=>{
    const pct=(c/total*100).toFixed(0);
    tc.innerHTML+=`<div class="tbar-row"><div class="tbar-name">${T_NAMES[i]}</div><div class="tbar-track"><div class="tbar-fill" style="width:0%;background:${T_BARS[i]}" data-w="${pct}"></div></div><div class="tbar-pct">${c}</div></div>`;
  });
  requestAnimationFrame(()=>document.querySelectorAll('.tbar-fill').forEach(el=>el.style.width=el.dataset.w+'%'));

  // step log
  let html='';
  for(let i=0;i<d.actions.length;i++){
    const rw=d.rewards[i];
    const rwc=rw>0.5?'#51cf66':rw<-2?'#ff6b6b':'#ffc247';
    html+=`<div class="log-row"><span class="l-idx">#${String(i+1).padStart(3,'0')}</span><span class="l-act">${d.actions[i]}</span><span class="l-rw" style="color:${rwc}">${rw>=0?'+':''}${rw}</span><span class="l-en">⚡${d.energies[i+1]}</span></div>`;
  }
  const log=document.getElementById('step-log');
  log.innerHTML=html; log.scrollTop=log.scrollHeight;
}

// ── SPARKLINE ─────────────────────────────────────────────────────────────────
function sparkline(svgId,data,stroke,fill,yMin,yMax){
  const el=document.getElementById(svgId);
  const W=el.parentElement.clientWidth||300, H=90;
  el.setAttribute('viewBox',`0 0 ${W} ${H}`);

  const mn=yMin!==null?yMin:Math.min(...data);
  const mx=yMax!==null?yMax:Math.max(...data);
  const range=(mx-mn)||1;
  const px=i=>(i/(data.length-1||1))*(W-24)+12;
  const py=v=>H-10-((v-mn)/range)*(H-22);

  let svg=`<defs>
    <linearGradient id="grad-${svgId}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${stroke}" stop-opacity=".5"/>
      <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
    </linearGradient>
    <filter id="glow-${svgId}">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;

  // zero line
  if(mn<0&&mx>0){
    const z=py(0);
    svg+=`<line x1="12" y1="${z}" x2="${W-12}" y2="${z}" stroke="rgba(255,255,255,0.1)" stroke-width="1" stroke-dasharray="4 3"/>`;
  }

  // filled area with gradient
  let area=`M ${px(0)} ${H-10}`;
  data.forEach((v,i)=>area+=` L ${px(i)} ${py(v)}`);
  area+=` L ${px(data.length-1)} ${H-10} Z`;
  svg+=`<path d="${area}" fill="url(#grad-${svgId})"/>`;

  // main line with glow
  const line=data.map((v,i)=>`${i===0?'M':'L'} ${px(i)} ${py(v)}`).join(' ');
  svg+=`<path d="${line}" fill="none" stroke="${stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow-${svgId})"/>`;

  // axis ticks
  svg+=`<text x="4" y="${H-8}" fill="#3d4f63" font-size="9" font-family="Share Tech Mono">${mn.toFixed(0)}</text>`;
  svg+=`<text x="4" y="11" fill="#3d4f63" font-size="9" font-family="Share Tech Mono">${mx.toFixed(0)}</text>`;

  // highlight min + max
  const minI=data.indexOf(Math.min(...data)), maxI=data.indexOf(Math.max(...data));
  [[minI,'#ff6b6b'],[maxI,stroke]].forEach(([idx,c])=>{
    const cx=px(idx), cy=py(data[idx]);
    svg+=`<circle cx="${cx}" cy="${cy}" r="4" fill="${c}" opacity=".9" filter="url(#glow-${svgId})"/>`;
    svg+=`<circle cx="${cx}" cy="${cy}" r="2" fill="#fff" opacity=".8"/>`;
  });

  el.innerHTML=svg;
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mars_rover_ppo")
    parser.add_argument("--port",  type=int, default=8765)
    args = parser.parse_args()

    env   = MarsRoverEnv()
    model = PPO.load(args.model)

    server = HTTPServer(("127.0.0.1", args.port), make_handler(model, env))
    url = f"http://127.0.0.1:{args.port}"
    print(f"  Mars Rover Mission Control  →  {url}")
    print(f"  Press Ctrl+C to stop.\n")

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()