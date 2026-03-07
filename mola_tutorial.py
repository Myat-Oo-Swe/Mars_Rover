"""
mola_tutorial.py  —  MOLA Mars terrain tutorial (fixed for metre-based CRS)

Usage
-----
  python mola_tutorial.py --download
  python mola_tutorial.py --info
  python mola_tutorial.py --patch --lat 18.4 --lon 77.7 --show   # Jezero Crater
  python mola_tutorial.py --patch --lat -8   --lon -70  --show   # Valles Marineris
  python mola_tutorial.py --patch --show                          # random location
  python mola_tutorial.py --benchmark
"""

import argparse, sys, random
import numpy as np
from pathlib import Path

MOLA_PATH = Path(__file__).parent / "mola_dem.tif"
MOLA_URL  = ("https://planetarymaps.usgs.gov/mosaic/"
             "Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")

# ── Terrain thresholds — all values are fractions of [0, 1] ─────────────────
# Slope and roughness are each normalised to their own min/max per patch,
# so 0.0 = flattest/smoothest cell in the patch, 1.0 = steepest/roughest.
# Tune these between 0 and 1 — easy to reason about.
SLOPE_SAND    = 0.10   # bottom quarter of slope → safe
SLOPE_SLOPE   = 0.20   # above 20% of max slope → slope class
SLOPE_HAZARD  = 0.85   # above 85% of max slope → hazard
ROUGH_SAND    = 0.20   # bottom 30% of roughness → sand
ROUGH_ROCK    = 0.60   # 30-60% roughness → rock  |  above 60% → hazard

TERRAIN_NAMES  = ["Safe", "Sand", "Rock", "Slope", "Hazard"]
TERRAIN_COLORS = ["#1a3320", "#3b2c0d", "#1e1e30", "#2e1118", "#380808"]


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_mola():
    if MOLA_PATH.exists():
        print(f"  mola_dem.tif already exists ({MOLA_PATH.stat().st_size/1e6:.0f} MB) — skipping.")
        return

    import shutil, subprocess

    if not shutil.which("aria2c"):
        print("  aria2c not found. Install with:  sudo apt install aria2")
        print("  Then re-run:  python mola_tutorial.py --download")
        sys.exit(1)

    print("  Downloading ~2 GB from USGS using aria2c (16 parallel connections)...")
    print(f"  Destination: {MOLA_PATH}\n")

    cmd = [
        "aria2c",
        "--split=16",              # 16 chunks
        "--max-connection-per-server=16",  # 16 connections per server
        "--min-split-size=10M",    # minimum chunk size
        "--continue=true",         # resume if interrupted
        "--file-allocation=none",  # faster start (no pre-allocation)
        "--console-log-level=notice",
        "--summary-interval=5",    # progress update every 5 seconds
        f"--dir={MOLA_PATH.parent}",
        f"--out={MOLA_PATH.name}",
        MOLA_URL,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n  Download failed. Try running manually:")
        print(f"  aria2c -x16 -s16 -c -o mola_dem.tif \"{MOLA_URL}\"")
        sys.exit(1)

    print(f"\n  Done. Saved to {MOLA_PATH}  ({MOLA_PATH.stat().st_size/1e6:.0f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# INSPECT
# ─────────────────────────────────────────────────────────────────────────────

def inspect_mola():
    _require_file()
    import rasterio, rasterio.windows as rw
    with rasterio.open(MOLA_PATH) as src:
        print("\n===== MOLA FILE INFO =====")
        print(f"  Driver     : {src.driver}")
        print(f"  Dimensions : {src.width} x {src.height} pixels")
        print(f"  CRS        : {src.crs.to_epsg() or 'custom (metres)'}")
        # resolution is in metres for this projected file
        res_m = abs(src.transform.a)
        print(f"  Resolution : {res_m:.1f} m / pixel")
        print(f"  File size  : {MOLA_PATH.stat().st_size/1e9:.2f} GB")
        centre_window = rw.Window(src.width//2-500, src.height//2-500, 1000, 1000)
        sample = src.read(1, window=centre_window).astype(np.float32)
        if src.nodata: sample[sample == src.nodata] = np.nan
        print(f"\n  Elevation sample (centre 1000x1000 px):")
        print(f"    min  = {np.nanmin(sample):.0f} m")
        print(f"    max  = {np.nanmax(sample):.0f} m")
        print(f"    mean = {np.nanmean(sample):.0f} m")
        print("==========================\n")


# ─────────────────────────────────────────────────────────────────────────────
# MOLA TERRAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MolaTerrain:
    """
    Extract terrain patches from the MOLA GeoTIFF.

    Example
    -------
    mola = MolaTerrain()
    patch = mola.get_terrain_patch(lat=18.4, lon=77.7, size=20)
    # returns (20,20) int8 array with values 0-4
    """

    def __init__(self, path=None):
        self.path = Path(path or MOLA_PATH)
        if not self.path.exists():
            raise FileNotFoundError(f"Run --download first. File not found: {self.path}")
        try:
            import rasterio
            self._rasterio = rasterio
        except ImportError:
            raise ImportError("pip install rasterio")
        self._src = None

    def _open(self):
        if self._src is None or self._src.closed:
            self._src = self._rasterio.open(self.path)
        return self._src

    def _lat_lon_to_pixel(self, lat, lon):
        """
        Convert geographic (lat, lon) degrees to pixel (row, col).

        FIX: This file uses a projected CRS in metres, NOT degrees.
        We must reproject from WGS84 degrees → Mars projected metres first.
        """
        src = self._open()
        # Manual equirectangular projection for Mars sphere.
        # pyproj raises a celestial-body mismatch error when going Earth↔Mars,
        # so we do the maths directly — spherical equirectangular geometry.
        R = 3_396_190.0          # Mars mean radius in metres (IAU 2000)
        x = np.radians(lon) * R  # easting
        y = np.radians(lat) * R  # northing

        col, row = ~src.transform * (x, y)
        return int(row), int(col)

    def get_elevation_patch(self, lat: float, lon: float, size: int = 20) -> np.ndarray:
        """Extract a (size x size) float32 elevation patch centred on (lat, lon)."""
        import rasterio.windows as rw
        src  = self._open()
        row, col = self._lat_lon_to_pixel(lat, lon)
        half = size // 2
        r0   = max(0, min(row - half, src.height - size))
        c0   = max(0, min(col - half, src.width  - size))
        elev = src.read(1, window=rw.Window(c0, r0, size, size)).astype(np.float32)
        nd   = src.nodata
        if nd is not None:
            mask = elev == nd
            if mask.any():
                elev[mask] = np.median(elev[~mask]) if (~mask).any() else 0.0
        return elev

    def elevation_to_terrain(self, elev: np.ndarray) -> np.ndarray:
        """Convert float elevation patch → int8 terrain class grid (0-4)."""
        from scipy.ndimage import uniform_filter

        # Normalise elevation to [0,1]
        norm  = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)

        # Slope = gradient magnitude, normalised to [0, 1] within this patch
        dy, dx    = np.gradient(norm)
        slope_raw = np.sqrt(dx**2 + dy**2)
        slope     = (slope_raw - slope_raw.min()) / (slope_raw.max() - slope_raw.min() + 1e-6)

        # Roughness = local std dev in 3x3 window, normalised to [0, 1]
        m         = uniform_filter(norm.astype(np.float64), size=3)
        m2        = uniform_filter((norm**2).astype(np.float64), size=3)
        rough_raw = np.sqrt(np.maximum(m2 - m**2, 0.0))
        rough     = (rough_raw - rough_raw.min()) / (rough_raw.max() - rough_raw.min() + 1e-6)

        # Classification — Hazard defined first as a hard boundary,
        # then remaining classes fill in below it.
        terrain = np.zeros(elev.shape, dtype=np.int8)

        # Hazard (4): too steep OR too rough — checked first, nothing overrides it
        terrain[
            (slope >= SLOPE_HAZARD) |
            (rough >= ROUGH_ROCK)
        ] = 4

        # Slope (3): steep but not hazardous, smooth surface
        terrain[
            (slope >= SLOPE_SLOPE) &
            (slope <  SLOPE_HAZARD) &
            (rough <  ROUGH_ROCK)
        ] = 3

        # Rock (2): low slope like sand but medium roughness
        terrain[
            (slope <  SLOPE_SLOPE) &
            (rough >= ROUGH_SAND) &
            (rough <  ROUGH_ROCK)
        ] = 2

        # Sand (1): low slope, low roughness
        terrain[
            (slope >= SLOPE_SAND) &
            (slope <  SLOPE_SLOPE) &
            (rough <  ROUGH_SAND)
        ] = 1

        # Safe (0): very flat and smooth — default (already zeros)

        return terrain
    def get_terrain_patch(self, lat: float, lon: float, size: int = 20) -> np.ndarray:
        """One-shot: elevation extraction + terrain classification."""
        return self.elevation_to_terrain(self.get_elevation_patch(lat, lon, size))

    def random_patch(self, size: int = 20, lat_range=(-60, 60)):
        """Return (terrain, lat, lon) for a random location."""
        lat = random.uniform(*lat_range)
        lon = random.uniform(-180, 180)
        return self.get_terrain_patch(lat, lon, size), lat, lon

    def close(self):
        if self._src and not self._src.closed:
            self._src.close()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISE A PATCH
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_show(lat=None, lon=None, size=20, show=True, save=False):
    _require_file()
    mola = MolaTerrain()

    if lat is None or lon is None:
        terrain, lat, lon = mola.random_patch(size=size)
    else:
        terrain = mola.get_terrain_patch(lat, lon, size)

    elev   = mola.get_elevation_patch(lat, lon, size)
    counts = [(terrain == i).sum() for i in range(5)]
    total  = terrain.size

    print(f"\n===== MOLA PATCH =====")
    print(f"  Location  : lat={lat:.3f}°,  lon={lon:.3f}°")
    print(f"  Grid size : {size}x{size}  ({size*0.463:.1f} km x {size*0.463:.1f} km)")
    print(f"  Elevation : min={elev.min():.0f} m,  max={elev.max():.0f} m,  "
          f"range={elev.max()-elev.min():.0f} m")
    print(f"\n  Terrain breakdown:")
    for i, (name, count) in enumerate(zip(TERRAIN_NAMES, counts)):
        bar = "█" * int(count / total * 30)
        print(f"    {i} {name:<8} {bar} {count} ({count/total*100:.0f}%)")
    print("======================\n")

    if not (show or save):
        return terrain

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("pip install matplotlib"); return terrain

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0c0f")
    for ax in axes:
        ax.set_facecolor("#0a0c0f")
        ax.tick_params(colors="#4a5568")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c2535")

    # Panel 1 — elevation
    im = axes[0].imshow(elev, cmap="RdYlGn_r", interpolation="nearest")
    axes[0].set_title(f"Elevation (m)\nlat={lat:.2f}°  lon={lon:.2f}°",
                      color="#c9d4e0", fontsize=10)
    plt.colorbar(im, ax=axes[0])

    # Panel 2 — terrain classes
    cmap = ListedColormap(TERRAIN_COLORS)
    axes[1].imshow(terrain, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[1].set_title("Terrain Classes", color="#c9d4e0", fontsize=10)
    axes[1].legend(
        handles=[mpatches.Patch(color=TERRAIN_COLORS[i], label=f"{i} {TERRAIN_NAMES[i]}")
                 for i in range(5)],
        fontsize=7, facecolor="#111822", labelcolor="#c9d4e0", edgecolor="#1c2535"
    )

    # Panel 3 — distribution
    bars = axes[2].bar(TERRAIN_NAMES, [c/total*100 for c in counts],
                       color=TERRAIN_COLORS, edgecolor="#1c2535", width=0.6)
    axes[2].set_ylabel("% of cells", color="#c9d4e0")
    axes[2].set_title("Distribution", color="#c9d4e0", fontsize=10)
    axes[2].set_facecolor("#0d1117")
    for bar, pct in zip(bars, [c/total*100 for c in counts]):
        if pct > 2:
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f"{pct:.0f}%", ha="center", va="bottom",
                         color="#c9d4e0", fontsize=8)

    plt.suptitle(f"MOLA  |  {size}x{size} grid  |  {size*0.463:.1f} km²",
                 color="#ff6b35", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        out = f"mola_patch_{lat:.1f}_{lon:.1f}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0a0c0f")
        print(f"  Saved: {out}")
    if show:
        plt.show()

    return terrain


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(n=10, size=20):
    _require_file()
    try:
        from env.mars_rover_env import MarsRoverEnv
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"  Missing: {e}"); return

    model_path = next((p for p in ["mars_rover_ppo_best.zip","mars_rover_ppo.zip"]
                       if Path(p).exists()), None)
    if not model_path:
        print("  No model found."); return

    mola    = MolaTerrain()
    env     = MarsRoverEnv(size=size)
    model   = PPO.load(model_path.replace(".zip",""))
    results = []

    LOCATIONS = [
        ("Jezero Crater",     18.4,   77.7),
        ("Gale Crater",       -5.4,  137.8),
        ("Olympus Mons",      18.6, -133.8),
        ("Valles Marineris",  -8.0,  -70.0),
        ("Amazonis Planitia", 24.0, -160.0),
        ("Hellas Basin",     -42.0,   70.0),
        ("Arabia Terra",      18.0,   30.0),
        ("Tharsis Plateau",   10.0, -105.0),
        ("Argyre Planitia",  -50.0,  -43.0),
        ("Elysium Mons",      25.0,  147.0),
    ]

    print(f"\n===== MOLA BENCHMARK ({n} missions) =====\n")
    print(f"  {'Location':<22} {'Result':<10} {'Steps':<8} {'Energy':<10} {'Hazards'}")
    print(f"  {'-'*58}")

    for name, lat, lon in LOCATIONS[:n]:
        terrain = mola.get_terrain_patch(lat, lon, size)
        obs, _  = env.reset(options={"start_pos":(0,0),"goal_pos":(size-1,size-1),"terrain":terrain})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        success = bool(np.array_equal(env.pos, env.goal))
        results.append(success)
        print(f"  {'✓' if success else '✗'} {name:<20} "
              f"{'SUCCESS' if success else 'FAILED':<10} {env.steps:<8} "
              f"{env.energy:.1f}/100   {env.hazard_hits}")

    print(f"\n  Success rate: {sum(results)/len(results)*100:.0f}%  ({sum(results)}/{len(results)})\n")



# ─────────────────────────────────────────────────────────────────────────────
# 3D VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def show_3d(lat=None, lon=None, size=20):
    """Save an interactive 3D surface as HTML and open it in the browser."""
    _require_file()
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  pip install plotly")
        sys.exit(1)

    import webbrowser, tempfile, os

    mola = MolaTerrain()
    if lat is None or lon is None:
        _, lat, lon = mola.random_patch(size=size)

    elev    = mola.get_elevation_patch(lat, lon, size)
    terrain = mola.get_terrain_patch(lat, lon, size)

    PLOTLY_COLORS = [
        [0.00, "#1a3320"], [0.20, "#1a3320"],  # Safe
        [0.20, "#c8a84b"], [0.40, "#c8a84b"],  # Sand
        [0.40, "#6e6e8a"], [0.60, "#6e6e8a"],  # Rock
        [0.60, "#8b3a3a"], [0.80, "#8b3a3a"],  # Slope
        [0.80, "#cc2200"], [1.00, "#cc2200"],  # Hazard
    ]

    cell_km = 0.463
    x = np.arange(size) * cell_km
    y = np.arange(size) * cell_km

    counts    = [(terrain == i).sum() for i in range(5)]
    total     = terrain.size
    breakdown = "  |  ".join(
        f"{TERRAIN_NAMES[i]}: {counts[i]/total*100:.0f}%" for i in range(5)
    )

    fig = go.Figure(go.Surface(
        z=elev,
        x=x,
        y=y,
        surfacecolor=terrain.astype(float),
        colorscale=PLOTLY_COLORS,
        cmin=0, cmax=4,
        colorbar=dict(
            title=dict(text="Terrain", font=dict(color="#ff6b35", size=13)),
            tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
            ticktext=["Safe", "Sand", "Rock", "Slope", "Hazard"],
            tickfont=dict(color="#c9d4e0", size=11),
            thickness=18,
            len=0.6,
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="#2a3a4a",
        ),
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.5, specular=0.2),
        lightposition=dict(x=2, y=2, z=3),
        hovertemplate="x: %{x:.1f} km<br>y: %{y:.1f} km<br>elev: %{z:.0f} m<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>MOLA Mars Terrain — 3D</b>   "
                f"lat={lat:.2f}°  lon={lon:.2f}°  |  {size}×{size}  |  {size*cell_km:.1f} km²<br>"
                f"<sup>{breakdown}</sup>"
            ),
            font=dict(color="#eef2f7", size=14),
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title=dict(text="km", font=dict(color="#6b7f94")),
                       gridcolor="#1c2535", tickfont=dict(color="#6b7f94"),
                       backgroundcolor="#0d1117"),
            yaxis=dict(title=dict(text="km", font=dict(color="#6b7f94")),
                       gridcolor="#1c2535", tickfont=dict(color="#6b7f94"),
                       backgroundcolor="#0d1117"),
            zaxis=dict(title=dict(text="elevation (m)", font=dict(color="#6b7f94")),
                       gridcolor="#1c2535", tickfont=dict(color="#6b7f94"),
                       backgroundcolor="#080b10"),
            bgcolor="#080b10",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
        ),
        paper_bgcolor="#080b10",
        font=dict(color="#c9d4e0"),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # Save to a temp HTML file and open in browser
    out = os.path.join(tempfile.gettempdir(), f"mola_3d_{lat:.2f}_{lon:.2f}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  Saved: {out}")
    print(f"  Opening in browser...")
    webbrowser.open(f"file://{out}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS / CLI
# ─────────────────────────────────────────────────────────────────────────────

def _require_file():
    if not MOLA_PATH.exists():
        print("mola_dem.tif not found. Run:  python mola_tutorial.py --download")
        sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download",  action="store_true")
    p.add_argument("--info",      action="store_true")
    p.add_argument("--patch",     action="store_true")
    p.add_argument("--3d",        action="store_true", dest="show3d")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--lat",  type=float, default=None)
    p.add_argument("--lon",  type=float, default=None)
    p.add_argument("--size", type=int,   default=20)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    if   args.download:  download_mola()
    elif args.info:      inspect_mola()
    elif args.patch:     extract_and_show(args.lat, args.lon, args.size, args.show, args.save)
    elif args.show3d:    show_3d(args.lat, args.lon, args.size)
    elif args.benchmark: benchmark(size=args.size)
    else:
        print("Usage:")
        print("  python mola_tutorial.py --download")
        print("  python mola_tutorial.py --info")
        print("  python mola_tutorial.py --patch --lat 18.4 --lon 77.7 --show")
        print("  python mola_tutorial.py --3d    --lat 18.4 --lon 77.7")
        print("  python mola_tutorial.py --3d    --lat -8   --lon -70 --size 40")
        print("  python mola_tutorial.py --benchmark")

if __name__ == "__main__":
    main()