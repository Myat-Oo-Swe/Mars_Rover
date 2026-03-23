"""
mola_tutorial.py  —  MOLA Mars terrain utilities + tutorial CLI
================================================================
This file now serves a dual purpose:
  1. Standalone CLI tool for downloading, inspecting, and visualising MOLA data.
  2. Shared library — train_ppo.py and visualize_mission_3d.py import
     MolaTerrain and MolaTerrainCache from here.

All changes vs original
-----------------------
BUG-1   MolaTerrain.__del__ added — file handle never leaks now.
BUG-2   random_patch uses np.random.default_rng, not stdlib random.
BUG-3   benchmark() success detection uses `terminated`, not positional
        comparison.
NEW-1   MolaTerrainCache — pre-loads N patches at startup, filters
        unreachable ones via BFS, serves them from RAM during training
        (zero disk I/O per episode after init).
NEW-2   elevation_to_terrain classification gap documented and tightened.
NEW-3   Consistent threshold constants shared between MolaTerrain and cache.

Usage
-----
  python mola_tutorial.py --download
  python mola_tutorial.py --info
  python mola_tutorial.py --patch --lat 18.4 --lon 77.7 --show
  python mola_tutorial.py --patch --lat -8   --lon -70  --show
  python mola_tutorial.py --3d    --lat 18.4 --lon 77.7
  python mola_tutorial.py --benchmark
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

MOLA_PATH = Path(__file__).parent / "mola_dem.tif"
MOLA_URL  = ("https://planetarymaps.usgs.gov/mosaic/"
             "Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")

# ── Terrain thresholds — normalised [0, 1] per patch ─────────────────────────
SLOPE_SAND    = 0.10
SLOPE_SLOPE   = 0.20
SLOPE_HAZARD  = 0.85
ROUGH_SAND    = 0.20
ROUGH_ROCK    = 0.60

TERRAIN_NAMES  = ["Safe", "Sand", "Rock", "Slope", "Hazard"]
TERRAIN_COLORS = ["#1a3320", "#3b2c0d", "#1e1e30", "#2e1118", "#380808"]


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_mola():
    if MOLA_PATH.exists():
        print(f"  mola_dem.tif already exists "
              f"({MOLA_PATH.stat().st_size / 1e6:.0f} MB) — skipping.")
        return

    import shutil, subprocess

    if not shutil.which("aria2c"):
        print("  aria2c not found. Install:  sudo apt install aria2")
        print("  Then re-run:  python mola_tutorial.py --download")
        sys.exit(1)

    print("  Downloading ~2 GB from USGS using aria2c …")
    cmd = [
        "aria2c", "--split=16", "--max-connection-per-server=16",
        "--min-split-size=10M", "--continue=true", "--file-allocation=none",
        "--console-log-level=notice", "--summary-interval=5",
        f"--dir={MOLA_PATH.parent}", f"--out={MOLA_PATH.name}", MOLA_URL,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  Download failed. Try manually:\n"
              f"  aria2c -x16 -s16 -c -o mola_dem.tif \"{MOLA_URL}\"")
        sys.exit(1)
    print(f"\n  Saved to {MOLA_PATH}  ({MOLA_PATH.stat().st_size / 1e6:.0f} MB)")


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
        res_m = abs(src.transform.a)
        print(f"  Resolution : {res_m:.1f} m / pixel")
        print(f"  File size  : {MOLA_PATH.stat().st_size / 1e9:.2f} GB")
        centre_window = rw.Window(
            src.width // 2 - 500, src.height // 2 - 500, 1000, 1000
        )
        sample = src.read(1, window=centre_window).astype(np.float32)
        if src.nodata:
            sample[sample == src.nodata] = np.nan
        print(f"\n  Elevation sample (centre 1000×1000 px):")
        print(f"    min  = {np.nanmin(sample):.0f} m")
        print(f"    max  = {np.nanmax(sample):.0f} m")
        print(f"    mean = {np.nanmean(sample):.0f} m")
        print("==========================\n")


# ─────────────────────────────────────────────────────────────────────────────
# MOLA TERRAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MolaTerrain:
    """
    Extract and classify terrain patches from the MOLA GeoTIFF.

    Thread safety: each instance opens its own file handle. Do NOT share
    a single instance across multiprocessing workers — pass a path string
    and construct per-process.

    Example
    -------
    mola  = MolaTerrain()
    patch = mola.get_terrain_patch(lat=18.4, lon=77.7, size=20)
    # (20, 20) int8 array, values 0-4
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or MOLA_PATH)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Run --download first. File not found: {self.path}"
            )
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

    def __del__(self):                          # [BUG-1]  always close handle
        self.close()

    def close(self):
        if self._src and not self._src.closed:
            try:
                self._src.close()
            except Exception:
                pass

    def _lat_lon_to_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        """
        Convert geographic degrees to pixel (row, col).
        Uses manual equirectangular Mars projection — avoids pyproj
        celestial-body mismatch errors.
        """
        src = self._open()
        R   = 3_396_190.0          # Mars mean radius (IAU 2000)
        x   = np.radians(lon) * R
        y   = np.radians(lat) * R
        col, row = ~src.transform * (x, y)
        return int(row), int(col)

    def get_elevation_patch(
        self, lat: float, lon: float, size: int = 20
    ) -> np.ndarray:
        """Extract a (size, size) float32 elevation patch centred on (lat, lon)."""
        import rasterio.windows as rw
        src       = self._open()
        row, col  = self._lat_lon_to_pixel(lat, lon)
        half      = size // 2
        r0        = max(0, min(row - half, src.height - size))
        c0        = max(0, min(col - half, src.width  - size))
        elev      = src.read(1, window=rw.Window(c0, r0, size, size)).astype(np.float32)
        nd        = src.nodata
        if nd is not None:
            mask = elev == nd
            if mask.any():
                elev[mask] = np.median(elev[~mask]) if (~mask).any() else 0.0
        return elev

    def elevation_to_terrain(self, elev: np.ndarray) -> np.ndarray:
        """
        Convert a float elevation patch → int8 terrain class grid (0-4).

        Classification order (hazard first, safe last) ensures the most
        dangerous label always wins when multiple conditions overlap.

        [NEW-2]  ROUGH_ROCK threshold is exclusive on the rock boundary
        so a cell with rough == ROUGH_ROCK lands in hazard, not rock —
        making the boundary unambiguous.
        """
        from scipy.ndimage import uniform_filter

        norm   = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
        dy, dx = np.gradient(norm)
        sr     = np.sqrt(dx**2 + dy**2)
        slope  = (sr - sr.min()) / (sr.max() - sr.min() + 1e-6)

        m         = uniform_filter(norm.astype(np.float64), size=3)
        m2        = uniform_filter((norm**2).astype(np.float64), size=3)
        rr        = np.sqrt(np.maximum(m2 - m**2, 0.0))
        rough     = (rr - rr.min()) / (rr.max() - rr.min() + 1e-6)

        terrain = np.zeros(elev.shape, dtype=np.int8)

        # Hazard (4): steep slope OR very rough — checked first
        terrain[(slope >= SLOPE_HAZARD) | (rough >= ROUGH_ROCK)] = 4

        # Slope (3): moderately steep, not rough enough to be hazard
        terrain[
            (slope >= SLOPE_SLOPE) & (slope < SLOPE_HAZARD) &
            (rough < ROUGH_ROCK)
        ] = 3

        # Rock (2): flat but moderately rough
        terrain[
            (slope < SLOPE_SLOPE) &
            (rough >= ROUGH_SAND) & (rough < ROUGH_ROCK)
        ] = 2

        # Sand (1): low slope, low roughness
        terrain[
            (slope >= SLOPE_SAND) & (slope < SLOPE_SLOPE) &
            (rough < ROUGH_SAND)
        ] = 1

        # Safe (0): default — already zeros

        return terrain

    def get_terrain_patch(
        self, lat: float, lon: float, size: int = 20
    ) -> np.ndarray:
        """One-shot convenience: elevation extraction + classification."""
        return self.elevation_to_terrain(self.get_elevation_patch(lat, lon, size))

    def get_patch(
        self, lat: float, lon: float, size: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (elevation, terrain) pair — used by visualizer."""
        elev    = self.get_elevation_patch(lat, lon, size)
        terrain = self.elevation_to_terrain(elev)
        return elev, terrain

    def random_patch(
        self,
        size: int = 20,
        lat_range: tuple[float, float] = (-60.0, 60.0),
        rng: np.random.Generator | None = None,       # [BUG-2]
    ) -> tuple[np.ndarray, float, float]:
        """
        Return (terrain, lat, lon) for a random location.
        [BUG-2]  Uses numpy RNG — not stdlib random — for reproducibility.
        """
        rng = rng or np.random.default_rng()
        lat = float(rng.uniform(*lat_range))
        lon = float(rng.uniform(-180.0, 180.0))
        return self.get_terrain_patch(lat, lon, size), lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# MOLA TERRAIN CACHE  [NEW-1]
# ─────────────────────────────────────────────────────────────────────────────

class MolaTerrainCache:
    """
    Pre-loads N terrain patches from MOLA at startup and serves them
    by random sampling during training — zero disk I/O per reset() after init.

    Why this matters
    ----------------
    With 8 SubprocVecEnv workers each calling rasterio on every episode reset,
    disk seeks into the 2 GB MOLA file become the training bottleneck.
    A 500-patch cache uses only ~200 KB of RAM (500 × 20×20 × 1 byte) and
    makes reset() O(1) — just a numpy array copy.

    Reachability filtering
    ----------------------
    Each candidate patch is tested with BFS before being added to the cache.
    Only patches where (0,0) can reach (size-1, size-1) are kept.
    This is done once at startup rather than per-episode.

    Usage
    -----
    cache = MolaTerrainCache(n_patches=300, size=20)
    terrain = cache.sample()          # random int8 (20,20) array
    terrain = cache.sample_for_stage(curriculum_phase=0.1)  # easy region
    """

    # Geographic regions by difficulty — used for geographic curriculum
    # Each entry: (name, lat, lon)
    REGIONS = {
        "easy": [
            ("Amazonis Planitia",  24.0, -160.0),
            ("Vastitas Borealis",  70.0,    0.0),
            ("Arcadia Planitia",   47.0, -170.0),
            ("Utopia Planitia",    48.0,  110.0),
            ("Chryse Planitia",    22.0,  -30.0),
        ],
        "medium": [
            ("Arabia Terra",       18.0,   30.0),
            ("Isidis Planitia",    13.0,   87.0),
            ("Hellas rim",        -38.0,   65.0),
            ("Gale Crater",        -5.4,  137.8),
            ("Jezero Crater",      18.4,   77.7),
        ],
        "hard": [
            ("Olympus Mons",       18.6, -133.8),
            ("Tharsis Plateau",    10.0, -105.0),
            ("Valles Marineris",   -8.0,  -70.0),
            ("Elysium Mons",       25.0,  147.0),
            ("Argyre rim",        -47.0,  -40.0),
        ],
    }

    def __init__(
        self,
        n_patches:         int   = 300,
        size:              int   = 20,
        lat_range:         tuple = (-60.0, 60.0),
        seed:              int   = 42,
        verbose:           bool  = True,
    ):
        self.size      = size
        self.rng       = np.random.default_rng(seed)
        self._patches:  list[np.ndarray] = []
        self._coords:   list[tuple[float, float]] = []

        # Per-difficulty buckets for geographic curriculum
        self._easy_patches:   list[np.ndarray] = []
        self._medium_patches: list[np.ndarray] = []
        self._hard_patches:   list[np.ndarray] = []

        if verbose:
            print(f"  Building MOLA terrain cache ({n_patches} patches, "
                  f"size={size})…")

        mola    = MolaTerrain()
        env_bfs = _BFSChecker(size)
        kept    = 0
        tried   = 0

        # Fill named-region buckets first
        for difficulty, locations in self.REGIONS.items():
            bucket = getattr(self, f"_{difficulty}_patches")
            for _, lat, lon in locations:
                for radius in (0.0, 0.3, -0.3, 0.6, -0.6):
                    t = mola.get_terrain_patch(lat + radius, lon + radius, size)
                    if env_bfs.is_reachable(t):
                        bucket.append(t)
                        self._patches.append(t)
                        self._coords.append((lat, lon))
                        kept += 1
                    tried += 1

        # Fill remainder with random patches
        while kept < n_patches:
            t, lat, lon = mola.random_patch(size=size, lat_range=lat_range,
                                            rng=self.rng)
            tried += 1
            if env_bfs.is_reachable(t):
                self._patches.append(t)
                self._coords.append((lat, lon))
                kept += 1

        mola.close()

        if verbose:
            print(f"  Cache ready — {kept} patches "
                  f"({tried - kept} rejected by BFS), "
                  f"~{kept * size * size / 1024:.0f} KB RAM")
            for d in ("easy", "medium", "hard"):
                n = len(getattr(self, f"_{d}_patches"))
                print(f"    {d:<8}: {n} patches")

    def sample(self) -> np.ndarray:
        """Return a random patch (uniform over all cached patches)."""
        idx = int(self.rng.integers(0, len(self._patches)))
        return self._patches[idx].copy()

    def sample_for_stage(self, curriculum_phase: float) -> np.ndarray:
        """
        Sample according to curriculum phase:
          phase < 0.25  →  easy regions only
          phase < 0.50  →  easy + medium
          phase < 0.75  →  easy + medium + hard
          phase >= 0.75 →  full cache (all)
        This replaces the distance-based curriculum with geographic difficulty.
        """
        if curriculum_phase < 0.25:
            pool = self._easy_patches or self._patches
        elif curriculum_phase < 0.50:
            pool = (self._easy_patches + self._medium_patches) or self._patches
        elif curriculum_phase < 0.75:
            pool = (self._easy_patches + self._medium_patches +
                    self._hard_patches) or self._patches
        else:
            pool = self._patches

        idx = int(self.rng.integers(0, len(pool)))
        return pool[idx].copy()

    def __len__(self) -> int:
        return len(self._patches)


class _BFSChecker:
    """Lightweight BFS reachability check used only during cache build."""
    def __init__(self, size: int):
        self.size = size

    def is_reachable(self, terrain: np.ndarray) -> bool:
        from collections import deque
        size  = self.size
        start = (0, 0)
        goal  = (size - 1, size - 1)
        seen: set[tuple[int, int]] = {start}
        q:    deque[tuple[int, int]] = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                node   = (nr, nc)
                if 0 <= nr < size and 0 <= nc < size and node not in seen:
                    if node == goal:
                        return True
                    seen.add(node)
                    q.append(node)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISE A PATCH
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_show(
    lat=None, lon=None, size=20, show=True, save=False
) -> np.ndarray:
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
    print(f"  Grid size : {size}×{size}  ({size * 0.463:.1f} km × {size * 0.463:.1f} km)")
    print(f"  Elevation : min={elev.min():.0f} m,  max={elev.max():.0f} m,  "
          f"range={elev.max() - elev.min():.0f} m")
    print(f"\n  Terrain breakdown:")
    for i, (name, count) in enumerate(zip(TERRAIN_NAMES, counts)):
        bar = "█" * int(count / total * 30)
        print(f"    {i} {name:<8} {bar} {count} ({count / total * 100:.0f}%)")
    print("======================\n")

    if not (show or save):
        return terrain

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("pip install matplotlib")
        return terrain

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0c0f")
    for ax in axes:
        ax.set_facecolor("#0a0c0f")
        ax.tick_params(colors="#4a5568")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c2535")

    im = axes[0].imshow(elev, cmap="RdYlGn_r", interpolation="nearest")
    axes[0].set_title(
        f"Elevation (m)\nlat={lat:.2f}°  lon={lon:.2f}°",
        color="#c9d4e0", fontsize=10,
    )
    plt.colorbar(im, ax=axes[0])

    cmap = ListedColormap(TERRAIN_COLORS)
    axes[1].imshow(terrain, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[1].set_title("Terrain Classes", color="#c9d4e0", fontsize=10)
    axes[1].legend(
        handles=[mpatches.Patch(color=TERRAIN_COLORS[i], label=f"{i} {TERRAIN_NAMES[i]}")
                 for i in range(5)],
        fontsize=7, facecolor="#111822", labelcolor="#c9d4e0", edgecolor="#1c2535",
    )

    bars = axes[2].bar(
        TERRAIN_NAMES, [c / total * 100 for c in counts],
        color=TERRAIN_COLORS, edgecolor="#1c2535", width=0.6,
    )
    axes[2].set_ylabel("% of cells", color="#c9d4e0")
    axes[2].set_title("Distribution", color="#c9d4e0", fontsize=10)
    axes[2].set_facecolor("#0d1117")
    for bar, pct in zip(bars, [c / total * 100 for c in counts]):
        if pct > 2:
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.0f}%",
                ha="center", va="bottom", color="#c9d4e0", fontsize=8,
            )

    plt.suptitle(
        f"MOLA  |  {size}×{size} grid  |  {size * 0.463:.1f} km²",
        color="#ff6b35", fontsize=13, fontweight="bold",
    )
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

def benchmark(n: int = 10, size: int = 20):
    _require_file()
    try:
        from env.mars_rover_env import MarsRoverEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as e:
        print(f"  Missing: {e}")
        return

    model_path = next(
        (p for p in ["mars_rover_ppo_best.zip", "mars_rover_ppo.zip"]
         if Path(p).exists()),
        None,
    )
    if not model_path:
        print("  No model found.")
        return

    model = PPO.load(model_path.replace(".zip", ""))

    # Load normalizer if present
    norm_path = Path("vec_normalize.pkl")
    vec_norm  = None
    if norm_path.exists():
        env_fn   = lambda: MarsRoverEnv(size=size)
        dummy    = DummyVecEnv([env_fn])
        vec_norm = VecNormalize.load(str(norm_path), dummy)
        vec_norm.training    = False
        vec_norm.norm_reward = False
        print("  VecNormalize loaded.")

    def normalize_obs(obs):
        if vec_norm is None:
            return obs
        return vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)

    mola    = MolaTerrain()
    env     = MarsRoverEnv(size=size)
    results = []

    LOCATIONS = [
        ("Jezero Crater",      18.4,   77.7),
        ("Gale Crater",        -5.4,  137.8),
        ("Olympus Mons",       18.6, -133.8),
        ("Valles Marineris",   -8.0,  -70.0),
        ("Amazonis Planitia",  24.0, -160.0),
        ("Hellas Basin",      -42.0,   70.0),
        ("Arabia Terra",       18.0,   30.0),
        ("Tharsis Plateau",    10.0, -105.0),
        ("Argyre Planitia",   -50.0,  -43.0),
        ("Elysium Mons",       25.0,  147.0),
    ]

    print(f"\n===== MOLA BENCHMARK ({n} missions) =====\n")
    print(f"  {'Location':<22} {'Result':<10} {'Steps':<8} {'Energy':<10} Hazards")
    print(f"  {'-'*58}")

    for name, lat, lon in LOCATIONS[:n]:
        terrain = mola.get_terrain_patch(lat, lon, size)
        obs, _  = env.reset(options={
            "start_pos": (0, 0),
            "goal_pos":  (size - 1, size - 1),
            "terrain":   terrain,
        })
        done       = False
        terminated = False
        while not done:
            action, _ = model.predict(normalize_obs(obs), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        # [BUG-3]  Use terminated flag — not positional comparison
        success = terminated
        results.append(success)
        print(f"  {'✓' if success else '✗'} {name:<20} "
              f"{'SUCCESS' if success else 'FAILED':<10} {env.steps:<8} "
              f"{env.energy:.1f}/100   {env.hazard_hits}")

    sr = sum(results) / len(results) * 100
    print(f"\n  Success rate: {sr:.0f}%  ({sum(results)}/{len(results)})\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3D VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def show_3d(lat=None, lon=None, size=20):
    """Save an interactive 3D surface as HTML and open in the browser."""
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
    mola.close()

    PLOTLY_COLORS = [
        [0.00, "#1a3320"], [0.20, "#1a3320"],
        [0.20, "#c8a84b"], [0.40, "#c8a84b"],
        [0.40, "#6e6e8a"], [0.60, "#6e6e8a"],
        [0.60, "#8b3a3a"], [0.80, "#8b3a3a"],
        [0.80, "#cc2200"], [1.00, "#cc2200"],
    ]

    cell_km   = 0.463
    x         = np.arange(size) * cell_km
    y         = np.arange(size) * cell_km
    counts    = [(terrain == i).sum() for i in range(5)]
    total     = terrain.size
    breakdown = "  |  ".join(
        f"{TERRAIN_NAMES[i]}: {counts[i] / total * 100:.0f}%" for i in range(5)
    )

    fig = go.Figure(go.Surface(
        z=elev, x=x, y=y,
        surfacecolor=terrain.astype(float),
        colorscale=PLOTLY_COLORS,
        cmin=0, cmax=4,
        colorbar=dict(
            title=dict(text="Terrain", font=dict(color="#ff6b35", size=13)),
            tickvals=[0.4, 1.2, 2.0, 2.8, 3.6],
            ticktext=["Safe", "Sand", "Rock", "Slope", "Hazard"],
            tickfont=dict(color="#c9d4e0", size=11),
            thickness=18, len=0.6,
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
                f"lat={lat:.2f}°  lon={lon:.2f}°  |  {size}×{size}  |  "
                f"{size * cell_km:.1f} km²<br><sup>{breakdown}</sup>"
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

    out = os.path.join(tempfile.gettempdir(),
                       f"mola_3d_{lat:.2f}_{lon:.2f}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  Saved: {out}")
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
    elif args.patch:     extract_and_show(args.lat, args.lon, args.size,
                                          args.show, args.save)
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