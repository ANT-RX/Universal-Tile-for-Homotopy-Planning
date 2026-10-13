"""
mosaic_no_hppm.py — Assembles a mosaic by requesting tiles from the
diffusion model, applying ONLY the GEOMETRIC filters (margin around the
ports/center, collapse detection, cluster limit) -- WITHOUT running
HPPM to verify that the result is actually navigable.

*** WARNING: a mosaic assembled this way has NO guarantee of being
solvable. It is not useful for evaluating the model or for generating
training data -- it is only for visual inspection / scale testing (for
example, to see how dense a mosaic can get, beyond
MAX_OBSTACULOS=1000 of the C engine, which is not even invoked here). ***

Why it is much faster than assembler.py: without HPPM there is no need to
verify 4*P routes per candidate (that is where almost all the time cost of
the real assembler is) -- here the first candidate that passes the geometric
filters (normally the 1st or 2nd, the GEOMETRIC acceptance rate is
much higher than the real universality rate) is used directly. There is also no
repair per cell: without verification there is no signal that something failed.

Usage:
    python3 -m src.pipeline.mosaic_no_hppm --model-dir ckpt_tiles_limpio_v1 \
        --rows 10 --columns 10 --target-obstacles 5000 \
        --ports-per-side 3 --port-margin 0.04 \
        --output-dir artifacts/mosaico_5000_sin_hppm
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.domain_scaling import MIN_RADIUS, TILE_SIDE
from src.common.compat import load_checkpoint
from src.common.domain_scaling import max_radius as max_radius_L
from src.common.domain_scaling import (DEFAULT_CONFIDENCE_RADIUS, DEFAULT_DDIM_STEPS, DEFAULT_GUIDANCE, DEFAULT_MAX_CLUSTER, DEFAULT_OVERLAP_MARGIN, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_SEED)
from src.common.maps_io import save_maps
from src.dataset.universal_generator import ports_and_center
from src.dataset.verify_maps import MIN_OBS_GENERATOR, detect_collapse
from src.model.set_diffusion import (_largest_cluster, _universal_filter, build_model,
                                     cosine_alphas_cumprod, sample_batch, select_device)

# TILE_SIDE is imported from src.common.domain_scaling: single definition shared
# with src.pipeline.assembler.

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE  # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN        # --port-margin
MAX_CLUSTER = DEFAULT_MAX_CLUSTER        # --max-cluster
OVERLAP_MARGIN = DEFAULT_OVERLAP_MARGIN  # --overlap-margin
GUIDANCE = DEFAULT_GUIDANCE              # --guidance
ETA = 0.0                                # --eta
GENERATION_BATCH = 8                     # --generation-batch
MAX_GENERATION_ATTEMPTS = 200            # --max-generation-attempts
CONFIDENCE_RADIUS = DEFAULT_CONFIDENCE_RADIUS  # --confidence-radius
PLOT = True                              # --plot / --no-plot


def generate_tile_no_hppm(model, ab, cfg, n_target, protected, args, device, gen):
    """Asks the model for tiles until one passes the GEOMETRIC filters
    (without HPPM). The geometric acceptance rate is much higher than the
    real universality rate (there is no check that the corridor actually
    exists) -- normally the first candidate or the first few are enough."""
    n_max = cfg["n_max"]
    n_requested = int(np.clip(round(n_target), MIN_OBS_GENERATOR, n_max))
    max_radius_i = max_radius_L(n_requested, TILE_SIDE)

    attempts = 0
    while attempts < MAX_GENERATION_ATTEMPTS:
        b = min(GENERATION_BATCH, MAX_GENERATION_ATTEMPTS - attempts)
        nb = torch.full((b,), n_requested, dtype=torch.long, device=device)
        Lb = torch.full((b,), TILE_SIDE, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            raw = sample_batch(model, ab, nb, Lb, GUIDANCE, args.ddim_steps, ETA,
                               gen, device, cfg["success_only"], n_max).float().cpu().numpy()
        attempts += len(raw)
        for o in raw:
            o = o[:n_requested]
            f = _universal_filter(o, protected, PORT_MARGIN * TILE_SIDE, MIN_RADIUS,
                                  max_radius_i, n_max, TILE_SIDE)
            if len(f) < MIN_OBS_GENERATOR or detect_collapse(f):
                continue
            if MAX_CLUSTER is not None and _largest_cluster(f, OVERLAP_MARGIN) > MAX_CLUSTER:
                continue
            return f
    return None


def push_obstacles(obs, point, confidence_radius):
    """Geometry only: pushes obstacles that invade the confidence radius
    around start/goal. Same as in assembler.py, unchanged."""
    if len(obs) == 0:
        return obs
    obs = obs.copy()
    px, py = point
    ox, oy, r = obs[:, 0], obs[:, 1], obs[:, 2]
    d = np.hypot(ox - px, oy - py)
    needed = r + confidence_radius
    invade = d < needed
    if invade.any():
        ang = np.where(d[invade] < 1e-9, 0.0, np.arctan2((oy - py)[invade], (ox - px)[invade]))
        obs[invade, 0] = px + np.cos(ang) * (needed[invade] + 1e-6)
        obs[invade, 1] = py + np.sin(ang) * (needed[invade] + 1e-6)
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--columns", type=int, required=True)
    p.add_argument("--target-obstacles", type=int, default=None,
                   help="without this, each cell requests a random n from the checkpoint distribution")
    p.add_argument("--ddim-steps", type=int, default=DEFAULT_DDIM_STEPS)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", default="artifacts/mosaico_sin_hppm")
    args = p.parse_args()

    device = select_device(args.device)
    ckpt_path = args.model or os.path.join(args.model_dir, "diffusion_model.pt")
    ck = load_checkpoint(ckpt_path, map_location=device)
    cfg = ck["config"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    ab = cosine_alphas_cumprod(cfg["T"]).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"Model {ckpt_path} (step {ck.get('step')}) | n_max={cfg['n_max']} | guidance={GUIDANCE}")
    print("*** WITHOUT HPPM VERIFICATION -- geometric filters only. No guarantee of being navigable. ***")

    ports, center = ports_and_center(TILE_SIDE, PORTS_PER_SIDE)
    protected = list(ports.values()) + [center]

    L_total_x, L_total_y = args.columns * TILE_SIDE, args.rows * TILE_SIDE
    start, goal = (0.0, 0.0), (L_total_x, L_total_y)
    total_cells = args.rows * args.columns
    print(f"Mosaic {args.rows}x{args.columns} ({L_total_x:.1f} x {L_total_y:.1f}) | "
          f"{total_cells} cells | target={args.target_obstacles}")

    rng_np = np.random.default_rng(args.seed)
    hist = np.array(cfg["hist_n_success"], dtype=float)
    hist_p = hist / hist.sum()

    remaining_target = args.target_obstacles
    remaining_cells = total_cells
    t0 = time.time()
    tile_by_cell = {}
    for row in range(args.rows):
        for col in range(args.columns):
            n_target = (remaining_target / max(1, remaining_cells)) if args.target_obstacles is not None \
                else float(rng_np.choice(len(hist), p=hist_p))
            tile = generate_tile_no_hppm(model, ab, cfg, n_target, protected, args, device, gen)
            if tile is None:
                raise SystemExit(f"Cell ({row},{col}): no candidate passed the geometric "
                                 f"filters in {MAX_GENERATION_ATTEMPTS} attempts (target n "
                                 f"={n_target:.0f}). Unlikely without HPPM -- check "
                                 "--port-margin/--max-cluster.")
            tile_by_cell[(row, col)] = tile
            if args.target_obstacles is not None:
                remaining_target -= len(tile)
                remaining_cells -= 1
    dt = time.time() - t0

    global_obs = []
    for (row, col), tile in tile_by_cell.items():
        dx, dy = col * TILE_SIDE, row * TILE_SIDE
        for x, y, r in tile:
            global_obs.append([x + dx, y + dy, r])
    obs = np.array(global_obs, dtype=np.float32)
    print(f"{len(obs)} obstacles generated in {dt:.1f}s ({len(obs)/max(dt,1e-9):.1f} obstacles/s)")

    for point in (start, goal):
        obs = push_obstacles(obs, point, CONFIDENCE_RADIUS)

    os.makedirs(args.output_dir, exist_ok=True)
    save_maps(args.output_dir, [0], [obs], n_max=len(obs) + 1)
    with open(os.path.join(args.output_dir, "mosaic_report.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": args.rows, "columns": args.columns, "n_obstacles": len(obs),
                   "target_obstacles": args.target_obstacles, "verificado_con_hppm": False,
                   "checkpoint": ckpt_path, "time_sec": round(dt, 1)}, f, indent=2)
    print(f"Saved to {args.output_dir}/ (verificado_con_hppm=False)")

    if PLOT:
        fig, ax = plt.subplots(figsize=(max(6, L_total_x * 1.2), max(6, L_total_y * 1.2)))
        ax.set_xlim(-0.1, L_total_x + 0.1)
        ax.set_ylim(-0.1, L_total_y + 0.1)
        ax.set_aspect("equal")
        for c in range(args.columns + 1):
            ax.axvline(c * TILE_SIDE, color="0.85", linewidth=0.6, zorder=1)
        for f in range(args.rows + 1):
            ax.axhline(f * TILE_SIDE, color="0.85", linewidth=0.6, zorder=1)
        for x, y, r in obs:
            ax.add_patch(plt.Circle((x, y), r, facecolor="#1565c0", edgecolor="none", alpha=0.5, zorder=2))
        ax.plot(*start, "s", color="black", markersize=8, zorder=5)
        ax.plot(*goal, "*", color="black", markersize=14, zorder=5)
        ax.set_title(f"Mosaic {args.rows}x{args.columns} | {len(obs)} obstacles | "
                     "NOT VERIFIED (geometric filters only)", fontsize=10, color="#c62828")
        fig.tight_layout()
        out = os.path.join(args.output_dir, "mosaico.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Image saved to {out}")


if __name__ == "__main__":
    main()
