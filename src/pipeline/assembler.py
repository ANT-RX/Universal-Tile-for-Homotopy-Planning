"""
assembler.py — Assembles a mosaic of universal tiles by generating each tile
LIVE from the trained diffusion model (set_diffusion.py), not from a
pre-generated catalog; it reuses set_diffusion.py's build_model, sample_batch,
_universal_filter, _largest_cluster and _verify_universal_candidate rather than
reimplementing them.

Why live and not from a catalog:
  - A catalog from universal_generator.py is already "cheating" with respect to
    the real goal (testing whether the MODEL can generate useful mosaics): a
    mosaic assembled with tiles from the training dataset says nothing about
    whether the model generalizes.
  - set_diffusion.py's sample() saves every candidate that passed the previous
    filters, not only the universal ones, so it cannot be reused as a catalog;
    generating live means every tile placed was ALREADY verified universal.

Cost: each requested tile may need several candidates before finding a
universal one (measured success rate: ~27%, clean dataset, guidance=1),
generated in small batches (--generation-batch) and verified in PARALLEL
(--workers) against the 4*P ports. A batch of 8 has ~91% chance of at least
1 usable one; --max-generation-attempts caps a cell.

The assembled mosaic goes through clearance control at the start/goal, full
HPPM verification and repair by cell replacement if it fails.

Scale note: ROBOT_RADIUS/step_size/etc. do NOT scale with the size of the
mosaic -- they are physical properties of the robot, calibrated for the tile
(L=1.0). The mosaic is larger in absolute coordinates (tiles pasted as-is), but
the robot keeps the same relative size as in a single tile.

CAUTION -- hard engine limit: HPPMDinamico3.c / hppm_v2 has
MAX_OBSTACULOS=1000 (fixed-size matrix, it silently ignores any
obstacle beyond that). A mosaic that exceeds it fails with an error BEFORE
sending anything to HPPM (see verify_global) -- this limit is not raised
here because it would require recompiling the engine (out of scope of this
file).

Usage:
    python3 -m src.pipeline.assembler \
        --model-dir ckpt_tiles_limpio_v1 --rows 6 --columns 6 \
        --target-obstacles 800 --hppm-exe artifacts/hppm_v2 \
        --ports-per-side 3 --port-margin 0.04 \
        --output-dir artifacts/mosaico_6x6_vivo --workers 30
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.domain_scaling import (ENGINE_MAX_OBSTACLES, MIN_RADIUS, MOSAIC_VERIFICATION_SCOPE,
                                       TILE_SIDE,
                                       scaled_constants)
from src.common.compat import load_checkpoint
from src.common.domain_scaling import max_radius as max_radius_L
from src.common.domain_scaling import (DEFAULT_CONFIDENCE_RADIUS, DEFAULT_DDIM_STEPS, DEFAULT_GUIDANCE, DEFAULT_M1, DEFAULT_M2, DEFAULT_MAX_CLUSTER, DEFAULT_OVERLAP_MARGIN, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_SEED)
from src.common.maps_io import save_maps
from src.dataset.repaired_generator import culprit_obstacle
from src.dataset.universal_generator import ports_and_center
from src.dataset.verify_maps import MIN_OBS_GENERATOR, compute_repulsion, detect_collapse, verify_one_map
from src.model.set_diffusion import (_largest_cluster, _universal_filter, _verify_universal_candidate,
                                     build_model, cosine_alphas_cumprod, sample_batch, select_device)

# TILE_SIDE and ENGINE_MAX_OBSTACLES are defined in src.common.domain_scaling:
# the tile side is shared with src.pipeline.mosaic_no_hppm, and the engine limit
# has to stay in sync with `#define MAX_OBSTACULOS` in engine/HPPMDinamico3.c
# (the C engine ignores extra obstacles in silence, so a higher Python guard
# would let a truncated verification pass as a success).

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
CONFIDENCE_RADIUS = DEFAULT_CONFIDENCE_RADIUS  # --confidence-radius
MAX_CELL_RETRIES = 20                          # --max-cell-retries
TIMEOUT = 15.0                                 # --timeout
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE        # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN              # --port-margin
MAX_CLUSTER = DEFAULT_MAX_CLUSTER              # --max-cluster
OVERLAP_MARGIN = DEFAULT_OVERLAP_MARGIN        # --overlap-margin
GUIDANCE = DEFAULT_GUIDANCE                    # --guidance
ETA = 0.0                                      # --eta
GENERATION_BATCH = 8                           # --generation-batch
MAX_GENERATION_ATTEMPTS = 200                  # --max-generation-attempts
START = None                                   # --start (None = corner (0,0))
GOAL = None                                    # --goal (None = opposite corner of the mosaic)
PLOT = True                                    # --plot / --no-plot


# ==========================================================================
# Live generation of a single universal tile
# ==========================================================================
def generate_tile_live(model, ab, cfg, n_target, protected, ports, center, const,
                       args, device, gen, executor):
    """Asks the model for tiles with n close to n_target (clamped to
    [MIN_OBS_GENERATOR, n_max of the checkpoint]) until finding one that passes
    the geometric filter + verification of the 4*P routes. Batches of
    --generation-batch candidates, verified in parallel with `executor`
    (created only once in main(), reused for all cells).
    Returns the tile (n,3) or None if --max-generation-attempts was exhausted
    without finding a universal one."""
    n_max = cfg["n_max"]
    n_requested = int(np.clip(round(n_target), MIN_OBS_GENERATOR, n_max))
    max_radius_i = max_radius_L(n_requested, TILE_SIDE)

    evaluated = 0
    while evaluated < MAX_GENERATION_ATTEMPTS:
        b = min(GENERATION_BATCH, MAX_GENERATION_ATTEMPTS - evaluated)
        nb = torch.full((b,), n_requested, dtype=torch.long, device=device)
        Lb = torch.full((b,), TILE_SIDE, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            raw = sample_batch(model, ab, nb, Lb, GUIDANCE, args.ddim_steps, ETA,
                               gen, device, cfg["success_only"], n_max).float().cpu().numpy()
        evaluated += len(raw)

        filtered = []
        for o in raw:
            o = o[:n_requested]  # trim the zero padding (raw has shape (batch, n_max, 3),
                                 # all candidates in this batch share n_requested)
            f = _universal_filter(o, protected, PORT_MARGIN * TILE_SIDE, MIN_RADIUS,
                                  max_radius_i, n_max, TILE_SIDE)
            if len(f) < MIN_OBS_GENERATOR or detect_collapse(f):
                filtered.append(None)
                continue
            if MAX_CLUSTER is not None and _largest_cluster(f, OVERLAP_MARGIN) > MAX_CLUSTER:
                filtered.append(None)
                continue
            filtered.append(f)

        # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
        tasks = [(i, f.tolist(), args.hppm_exe, const["step_size"], TIMEOUT, ports, center,
                  const["robot_radius"], const["safety_margin"], const["arrival_radius"],
                  DEFAULT_M1, DEFAULT_M2) for i, f in enumerate(filtered) if f is not None]
        if not tasks:
            continue
        for r in executor.map(_verify_universal_candidate, tasks, chunksize=4):
            if r["state"] == "success":
                return filtered[r["idx"]]
    return None


# ==========================================================================
# Grid assembly (same logic as always, cells via generate_tile_live)
# ==========================================================================
def assemble_grid(model, ab, cfg, rows, columns, protected, ports, center, const,
                  args, device, gen, executor, rng, rng_np, tile_by_cell=None, total_target=None):
    """Returns (global_obstacles (n,3), cell_of_obstacle (n,),
    tile_by_cell {(row,col): tile}). tile_by_cell: if passed (from a
    cell already replaced, already containing its new tile), it is respected --
    only the cells that are still missing are generated. If any cell is left
    without a universal tile after --max-generation-attempts, it fails with an
    error (it does not assemble a mosaic with a silent hole). rng/rng_np:
    generators (random.Random / np.random.default_rng) seeded once in main()
    and passed explicitly -- they are not created anew on each call, so that
    --seed is truly reproducible."""
    tile_by_cell = dict(tile_by_cell or {})
    cells = [(r, c) for r in range(rows) for c in range(columns)]
    already_fixed = [c for c in cells if c in tile_by_cell]
    to_fix = rng.sample([c for c in cells if c not in tile_by_cell],
                        k=len(cells) - len(already_fixed))

    if total_target is not None:
        remaining_target = total_target - sum(len(tile_by_cell[c]) for c in already_fixed)
        remaining_cells = len(to_fix)
    else:
        remaining_target = remaining_cells = None

    hist = np.array(cfg["hist_n_success"], dtype=float)
    hist_p = hist / hist.sum()

    for cell in to_fix:
        n_target = (remaining_target / max(1, remaining_cells)) if total_target is not None \
            else float(rng_np.choice(len(hist), p=hist_p))
        tile = generate_tile_live(model, ab, cfg, n_target, protected, ports, center, const,
                                  args, device, gen, executor)
        if tile is None:
            raise SystemExit(
                f"Could not generate a universal tile for cell {cell} (target n={n_target:.0f}) "
                f"in --max-generation-attempts={MAX_GENERATION_ATTEMPTS} attempts. Raise that "
                "limit, lower --guidance, or check that the checkpoint/engine are fine.")
        tile_by_cell[cell] = tile
        if total_target is not None:
            remaining_target -= len(tile)
            remaining_cells -= 1

    global_obs, cell_of_obs = [], []
    for (row, col), tile in tile_by_cell.items():
        dx, dy = col * TILE_SIDE, row * TILE_SIDE
        for x, y, r in tile:
            global_obs.append([x + dx, y + dy, r])
            cell_of_obs.append((row, col))
    return np.array(global_obs, dtype=np.float32), cell_of_obs, tile_by_cell


def push_obstacles(obs, point, confidence_radius):
    """Pushes outward any obstacle that invades confidence_radius
    around `point` -- it moves its CENTER along the line
    obstacle->point until it is just outside (it does not change the radius, it
    does not discard it). Returns a copy; it does not touch `obs` in-place.
    Vectorized."""
    obs = obs.copy()
    if len(obs) == 0:
        return obs, 0
    px, py = point
    ox, oy, r = obs[:, 0], obs[:, 1], obs[:, 2]
    d = np.hypot(ox - px, oy - py)
    needed = r + confidence_radius
    invade = d < needed
    pushed = int(invade.sum())
    if pushed:
        ang = np.where(d[invade] < 1e-9, 0.0, np.arctan2((oy - py)[invade], (ox - px)[invade]))
        obs[invade, 0] = px + np.cos(ang) * (needed[invade] + 1e-6)
        obs[invade, 1] = py + np.sin(ang) * (needed[invade] + 1e-6)
    return obs, pushed


def verify_global(obs, start, goal, hppm_exe, timeout, m1, m2, return_traj=False):
    if len(obs) > ENGINE_MAX_OBSTACLES:
        raise SystemExit(
            f"The mosaic has {len(obs)} obstacles > MAX_OBSTACULOS={ENGINE_MAX_OBSTACLES} of the "
            "C engine. The engine would silently ignore them beyond that limit -- the HPPM result "
            "would not be reliable. Lower --target-obstacles or the size of the mosaic.")
    const = scaled_constants(TILE_SIDE)
    rep = compute_repulsion(obs, start=start, goal=goal,
                            robot_radius=const["robot_radius"], safety_margin=const["safety_margin"])
    _, res = verify_one_map((0, rep, hppm_exe, const["step_size"], timeout, return_traj,
                             start, goal, const["robot_radius"], const["arrival_radius"], m1, m2))
    return res


def plot_mosaic(obs, res, start, goal, rows, columns, cell_of_obs, out_path):
    L_x, L_y = columns * TILE_SIDE, rows * TILE_SIDE
    fig, ax = plt.subplots(figsize=(max(6, L_x * 1.4), max(6, L_y * 1.4)))
    ax.set_xlim(-0.1, L_x + 0.1)
    ax.set_ylim(-0.1, L_y + 0.1)
    ax.set_aspect("equal")
    for c in range(columns + 1):
        ax.axvline(c * TILE_SIDE, color="0.85", linewidth=0.6, zorder=1)
    for f in range(rows + 1):
        ax.axhline(f * TILE_SIDE, color="0.85", linewidth=0.6, zorder=1)
    for x, y, r in obs:
        ax.add_patch(plt.Circle((x, y), r, facecolor="#1565c0", edgecolor="none", alpha=0.55, zorder=2))
    status_color = {"success": "#2e7d32"}.get(res.get("state"), "#c62828")
    tray = res.get("tray")
    if tray is not None and len(tray):
        ax.plot(tray[:, 0], tray[:, 1], "-", color=status_color, linewidth=1.8, zorder=4)
    ax.plot(*start, "s", color="black", markersize=8, zorder=5, label="start")
    ax.plot(*goal, "*", color="black", markersize=14, zorder=5, label="goal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    title = f"Mosaic {rows}x{columns} | {len(obs)} obstacles | status='{res.get('state')}'"
    if res.get("state") == "success":
        title += f" | length={res.get('length'):.2f} | steps={res.get('steps')}"
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, columns + 1))
    ax.set_yticks(range(0, rows + 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Image saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="checkpoint folder (e.g. ckpt_tiles_limpio_v1)")
    p.add_argument("--model", default=None, help="defaults to <model-dir>/diffusion_model.pt")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--columns", type=int, default=None)
    p.add_argument("--target-obstacles", type=int, default=None,
                   help="total obstacles requested for the whole mosaic. Without this, each cell "
                        "requests a random n from the checkpoint distribution.")
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--ddim-steps", type=int, default=DEFAULT_DDIM_STEPS)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help="verification of candidates in parallel (CPU, HPPM) -- not to be confused "
                        "with the GPU device")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", default="artifacts/mosaico")
    args = p.parse_args()

    if args.rows is None or args.columns is None:
        raise SystemExit("--rows and --columns are mandatory.")

    device = select_device(args.device)
    ckpt_path = args.model or os.path.join(args.model_dir, "diffusion_model.pt")
    ck = load_checkpoint(ckpt_path, map_location=device)
    cfg = ck["config"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    ab = cosine_alphas_cumprod(cfg["T"]).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"Model {ckpt_path} (step {ck.get('step')}) | n_max={cfg['n_max']} | guidance={GUIDANCE}")

    ports, center = ports_and_center(TILE_SIDE, PORTS_PER_SIDE)
    protected = list(ports.values()) + [center]
    const = scaled_constants(TILE_SIDE)

    L_total_x, L_total_y = args.columns * TILE_SIDE, args.rows * TILE_SIDE
    start = tuple(START) if START else (0.0, 0.0)
    goal = tuple(GOAL) if GOAL else (L_total_x, L_total_y)
    print(f"Mosaic {args.rows}x{args.columns} tiles ({L_total_x:.1f} x {L_total_y:.1f} "
          f"in global coordinates) | start={start} goal={goal}")

    if args.target_obstacles is not None:
        total_cells = args.rows * args.columns
        min_possible = total_cells * MIN_OBS_GENERATOR
        max_possible = total_cells * cfg["n_max"]
        if not (min_possible <= args.target_obstacles <= max_possible):
            raise SystemExit(
                f"--target-obstacles {args.target_obstacles} is not reachable with "
                f"{args.rows}x{args.columns}={total_cells} cells (possible range: "
                f"[{min_possible}, {max_possible}], checkpoint n_max={cfg['n_max']}).")
        if args.target_obstacles > ENGINE_MAX_OBSTACLES:
            raise SystemExit(
                f"--target-obstacles {args.target_obstacles} > MAX_OBSTACULOS="
                f"{ENGINE_MAX_OBSTACLES} of the engine -- it will never be able to verify. Lower it.")
        print(f"Target: {args.target_obstacles} obstacles in {total_cells} cells "
              f"(required average: {args.target_obstacles/total_cells:.1f}/cell, "
              f"maximum of the checkpoint: {cfg['n_max']}/cell)")

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        obs, cell_of_obs, tile_by_cell = assemble_grid(
            model, ab, cfg, args.rows, args.columns, protected, ports, center, const,
            args, device, gen, executor, rng, rng_np, total_target=args.target_obstacles)
        print(f"{len(obs)} obstacles in the assembled mosaic ({time.time()-t0:.1f}s generating)")

        for point, name in [(start, "start"), (goal, "goal")]:
            obs, pushed = push_obstacles(obs, point, CONFIDENCE_RADIUS)
            if pushed:
                print(f"  pushed {pushed} obstacle(s) that invaded the confidence radius "
                      f"around {name}={point}")

        for attempt in range(MAX_CELL_RETRIES + 1):
            # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
            res = verify_global(obs, start, goal, args.hppm_exe, TIMEOUT, DEFAULT_M1, DEFAULT_M2,
                                return_traj=True)
            if res["state"] == "success":
                print(f"\n*** VALID MOSAIC *** (attempt {attempt}, length={res.get('length')}, "
                      f"steps={res.get('steps')})")
                break

            print(f"  attempt {attempt}: status='{res['state']}' -- looking for the culprit tile...")
            k = culprit_obstacle(obs, res, const["robot_radius"])
            if k is None:
                print("  could not identify a culprit obstacle (empty trajectory/timeout) "
                      "-- retrying with another random cell")
                bad_cell = (rng.randrange(args.rows), rng.randrange(args.columns))
            else:
                bad_cell = cell_of_obs[k]
            n_replacement = len(tile_by_cell[bad_cell])
            print(f"  changing the tile of cell {bad_cell} (n={n_replacement}, new one from the live model)")

            # generate the replacement targeting the SAME n the tile that failed had (not random) --
            # this way it does not skew the total already achieved if there is --target-obstacles
            new_tile = generate_tile_live(model, ab, cfg, n_replacement, protected, ports, center,
                                          const, args, device, gen, executor)
            if new_tile is None:
                raise SystemExit(
                    f"Could not generate a replacement for cell {bad_cell} (target n="
                    f"{n_replacement}) in --max-generation-attempts={MAX_GENERATION_ATTEMPTS} "
                    "attempts.")
            new_tile_by_cell = dict(tile_by_cell)
            new_tile_by_cell[bad_cell] = new_tile

            obs, cell_of_obs, tile_by_cell = assemble_grid(
                model, ab, cfg, args.rows, args.columns, protected, ports, center, const,
                args, device, gen, executor, rng, rng_np,
                tile_by_cell=new_tile_by_cell, total_target=None)
            for point, _ in [(start, "start"), (goal, "goal")]:
                obs, _ = push_obstacles(obs, point, CONFIDENCE_RADIUS)
        else:
            print(f"\n*** Could not achieve a valid mosaic in {MAX_CELL_RETRIES} "
                  f"cell replacements. ***")

    os.makedirs(args.output_dir, exist_ok=True)
    save_maps(args.output_dir, [0], [obs[:, :3]], n_max=len(obs) + 1)
    with open(os.path.join(args.output_dir, "mosaic_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "rows": args.rows, "columns": args.columns, "n_obstacles": len(obs),
            "target_obstacles": args.target_obstacles,
            "start": list(start), "meta": list(goal), "final_state": res["state"],
            "checkpoint": ckpt_path, "guidance": GUIDANCE,
            "n_per_cell": {f"{r},{c}": len(t) for (r, c), t in tile_by_cell.items()},
            "total_time_sec": round(time.time() - t0, 1),
            "cell_retries_used": attempt,
            # success BEFORE any cell repair (first verification of the freshly assembled
            # mosaic) vs. AFTER >=1 cell replacement -- reported separately (reviewer 4.6)
            "solved_first_try": res["state"] == "success" and attempt == 0,
            "solved_after_repair": res["state"] == "success" and attempt > 0,
            "mosaic_verification": MOSAIC_VERIFICATION_SCOPE,
        }, f, indent=2)
    print(f"Saved to {args.output_dir}/ ({time.time()-t0:.1f}s total)")

    if PLOT:
        plot_mosaic(obs, res, start, goal, args.rows, args.columns, cell_of_obs,
                    os.path.join(args.output_dir, "mosaico.png"))


if __name__ == "__main__":
    main()
