"""
Plan B: SUCCESS maps at any density with the original generator +
local repair verified by HPPM.

Algorithm per map:
  1. Generate a map with the exact rule of 1_generar_dataset.py.
  2. Run HPPM. If it is a success, finish.
  3. If there is a collision: identify the obstacle the
     trajectory crashes into (the first one it touches). If it is a stagnation: the
     obstacle closest to the last point.
  4. Re-sample ONLY that obstacle with the same generator rule
     (uniform position, uniform radius, exclusion zone) and repeat.

Every obstacle still comes out of the generator distribution; only
configurations that HPPM does not solve are rejected. The result is 100% success
by construction, so the relevant metrics are:
  - COST: HPPM runs per success map, versus 1/rate of pure
    rejection sampling (e.g. ~17 with 80 obstacles, ~45 with 95).
  - REALISM: use the diversity/memorisation test of the paper metrics
    (``src/evaluation/paper_metrics.py``) on the output folder.

The output has the same format as verify_maps.py
(mapa_NNNNN/obstaculos.txt, results_detail.json, report.json).

Usage:
    python3 repaired_generator.py --num-obs 80 --n 2000 --hppm-exe artifacts/hppm_v2 \
        --output-dir reparado_n80_s42 --seed 42

Gibbs steps and the iteration cap are frozen module constants, not flags:
set GIBBS = 80 for less bias, or MAX_ITER = 0 for pure rejection (the exact
reference).

To assess the realism of the output folder, run the paper-metrics diversity
test (``python3 -m src.evaluation.paper_metrics --tests diversity --help``).
"""
import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.dataset.baseline_generator import (EXCLUSION_MARGIN, STRICT_OVERLAP_MARGIN, MIN_RADIUS,
                                            generate_obstacles, cluster_size_with_candidate)
from src.common.domain_scaling import (MAX_DENSITY, STEP_SIZE_BASE, compute_compatible_L,
                                       scaled_constants, route_points)
from src.common.domain_scaling import max_radius as max_radius_L
from src.common.domain_scaling import (DEFAULT_L, DEFAULT_M1, DEFAULT_M2, DEFAULT_SEED, DEFAULT_TIMEOUT)
from src.common.maps_io import save_maps
from src.dataset.verify_maps import (ROBOT_RADIUS, compute_repulsion, write_hppm_files,
                                     verify_one_map)

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
MAX_ITER = 40                            # --max-iter
GIBBS = 0                                # --gibbs
TIMEOUT = DEFAULT_TIMEOUT                # --timeout
FORMAT = "npz"                           # --format
FULL_DETAIL = False                      # --full-detail
START = [0.0, 0.0]                       # --start
GOAL = [1.0, 1.0]                        # --goal
# NOTICE: MAX_CLUSTER is None ON PURPOSE, not DEFAULT_MAX_CLUSTER (4). This is a
# general-purpose repair tool that predates the universal-tiles scheme; the production
# path for tiles is universal_generator.py, which does default to 4. Do not "fix" this to 4.
MAX_CLUSTER = None                       # --max-cluster (None = no limit, current behavior)
OVERLAP_MARGIN = STRICT_OVERLAP_MARGIN   # --overlap-margin
L = str(DEFAULT_L)                       # --L ('auto' or a fixed number)
NUM_OBS = 0                              # --num-obs (0 = random n in [--num-obs-min, --num-obs-max])
NUM_OBS_MIN = 5                          # --num-obs-min
NUM_OBS_MAX = 100                        # --num-obs-max


def sample_obstacle(rng: random.Random, n: int, existing_obs=None,
                    start=(0.0, 0.0), goal=(1.0, 1.0), L: float = 1.0,
                    max_cluster: int = None, overlap_margin: float = STRICT_OVERLAP_MARGIN,
                    max_attempts: int = 20000):
    """One obstacle with the rule of baseline_generator.generate_obstacles for
    a map of n obstacles in the domain [0,L]x[0,L], rejecting by exclusion zone
    of start/goal (scaled with L) and, if max_cluster is passed,
    also by overlap cluster size with existing_obs. Used
    to RESAMPLE a single obstacle (repair/Gibbs) -- that is why the
    attempt budget is per-obstacle (max_attempts), unlike the
    shared budget of generate_obstacles()."""
    max_radius = max_radius_L(n, L) if L != 1.0 else math.sqrt(MAX_DENSITY / (n * math.pi))
    exclusion_margin = EXCLUSION_MARGIN if L == 1.0 else (
        scaled_constants(L)["generator_exclusion_radius"] + scaled_constants(L)["start_goal_margin"])
    for _ in range(max_attempts):
        x, y = rng.uniform(0.0, L), rng.uniform(0.0, L)
        r = rng.uniform(MIN_RADIUS, max_radius)
        if math.hypot(x - start[0], y - start[1]) <= r + exclusion_margin:
            continue
        if math.hypot(x - goal[0], y - goal[1]) <= r + exclusion_margin:
            continue
        candidate = [x, y, r]
        if max_cluster is not None:
            size = cluster_size_with_candidate(existing_obs, candidate, overlap_margin)
            if size > max_cluster:
                continue
        return candidate
    raise RuntimeError(
        f"sample_obstacle: no valid placement found in {max_attempts} attempts "
        f"(n={n}, L={L}, max_cluster={max_cluster}). The map may be too dense for "
        "this cluster limit — consider lowering num-obs, raising MAX_CLUSTER, or scaling the domain further.")


def culprit_obstacle(obs, res, robot_radius: float = None):
    robot_radius = ROBOT_RADIUS if robot_radius is None else robot_radius
    if "tray" not in res:
        raise RuntimeError(
            "verify_one_map did not return the trajectory: verify_maps.py is outdated "
            "(the 6th argument return_tray is missing). Without that the repair degrades to pure rejection.")
    tray = res["tray"]
    if len(tray) == 0:
        return None
    d = np.linalg.norm(tray[:, None, :] - obs[None, :, :2], axis=-1) - obs[None, :, 2]
    if res["state"] == "collision":
        crashes = (d <= robot_radius).any(axis=1)
        if crashes.any():
            return int(np.argmin(d[int(np.argmax(crashes))]))
    return int(np.argmin(d[-1]))


def repair_one_map(task):
    (idx, seed, n_obs, max_iter, exe, timeout, gibbs,
     fixed_start, fixed_goal, max_cluster, overlap_margin, L_spec, route,
     m1, m2, n_min, n_max) = task
    rng = random.Random(seed)

    # n is resolved HERE (not in main()) because each map has its own independent
    # rng (parallelized by seed) -- with L="auto" (growing map,
    # section 3 of the design) L depends on n, and n can be random per map.
    # n_obs=0 -> random n in [n_min, n_max] (before hardcoded to (5,100),
    # necessary to populate the dataset with a variety of n, not only the points
    # of a diagnostic bucket).
    n_i = n_obs or rng.randint(n_min, n_max)
    L_i = compute_compatible_L(n_i) if L_spec == "auto" else float(L_spec)
    if route:
        start, goal = route_points(route, L_i)
    else:
        start, goal = fixed_start, fixed_goal
    const = scaled_constants(L_i)
    step, robot_radius_i, arrival_radius_i = const["step_size"], const["robot_radius"], const["arrival_radius"]

    obs = generate_obstacles(rng, n_i, start=start, goal=goal, L=L_i,
                             max_cluster=max_cluster, overlap_margin=overlap_margin)
    n_rule = n_i
    for it in range(max_iter + 1):
        _, res = verify_one_map((idx, compute_repulsion(obs, start=start, goal=goal,
                                                         robot_radius=robot_radius_i,
                                                         safety_margin=const["safety_margin"]),
                                 exe, step, timeout, True, start, goal, robot_radius_i, arrival_radius_i,
                                 m1, m2))
        if res["state"] == "success":
            accepted = 0
            for _ in range(gibbs):
                # Gibbs with a proposal from the generator rule: it is accepted if
                # it is still a success. Stationary distribution = generator
                # conditioned on success (it corrects the bias of the repair).
                k_prop = rng.randrange(len(obs))
                prop = obs.copy()
                prop[k_prop] = sample_obstacle(
                    rng, n_rule, existing_obs=np.delete(obs, k_prop, axis=0),
                    start=start, goal=goal, L=L_i,
                    max_cluster=max_cluster, overlap_margin=overlap_margin)
                _, r2 = verify_one_map((idx, compute_repulsion(prop, start=start, goal=goal,
                                                                robot_radius=robot_radius_i,
                                                                safety_margin=const["safety_margin"]),
                                        exe, step, timeout, False, start, goal, robot_radius_i, arrival_radius_i,
                                        m1, m2))
                if r2["state"] == "success":
                    obs, res, accepted = prop, r2, accepted + 1
            return {"gibbs_accepted": accepted,"idx": idx, "state": "success", "iteraciones": it, "corridas_hppm": it + 1,
                    "num_obs": len(obs), "length": res["length"], "steps": res["steps"],
                    "success_before_repair": it == 0, "obs": obs, "total_runs": it + 1 + gibbs,
                    "L": L_i, "start": list(start), "meta": list(goal), "step_size": step,
                    "radio_robot": robot_radius_i, "m1": m1, "m2": m2}
        k = culprit_obstacle(obs, res, robot_radius_i)
        if k is None or it == max_iter:
            return {"idx": idx, "state": "no_reparado", "iteraciones": it, "corridas_hppm": it + 1,
                    "num_obs": len(obs), "success_before_repair": False, "obs": None,
                    "total_runs": it + 1, "L": L_i}
        obs = obs.copy()
        obs[k] = sample_obstacle(
            rng, n_rule, existing_obs=np.delete(obs, k, axis=0),
            start=start, goal=goal, L=L_i,
            max_cluster=max_cluster, overlap_margin=overlap_margin)


def main():
    p = argparse.ArgumentParser(description="Original generator + local repair verified by HPPM")
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--n", type=int, default=2000)
    # BEFORE: max(1, cpu_count()-2) -- intended for large machines (losing 2
    # cores out of 64 is not noticeable), but on small machines (Colab, 2 cores)
    # this left ONLY 1 worker working without warning -- half of the expected
    # compute. Now it uses all available cores by default.
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4),
                   help="parallel processes. Default: all available cores.")
    p.add_argument("--output-dir", default="artifacts/mapas_reparados")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--route", choices=["diagonal", "diagonal_inverse", "horizontal", "vertical"], default=None,
                   help="route family (it recomputes start/goal according to the L of each map). If it is passed, "
                        "it ignores --start/--goal. Without this, --start/--goal are used as is (intended "
                        "for fixed L, not for 'auto', where the domain changes size per map).")
    args = p.parse_args()

    fixed_start, fixed_goal = tuple(START), tuple(GOAL)
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = random.Random(args.seed)
    # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
    tasks = [(i, seeds.getrandbits(63), NUM_OBS, MAX_ITER, args.hppm_exe,
              TIMEOUT, GIBBS, fixed_start, fixed_goal, MAX_CLUSTER,
              OVERLAP_MARGIN, L, args.route, DEFAULT_M1, DEFAULT_M2,
              NUM_OBS_MIN, NUM_OBS_MAX) for i in range(args.n)]

    t0 = time.time()
    # Previous check: one map in the main process, to fail fast and clearly
    L_test = compute_compatible_L(80) if L == "auto" else float(L)
    start_p, goal_p = route_points(args.route, L_test) if args.route else (fixed_start, fixed_goal)
    const_p = scaled_constants(L_test)
    _, test = verify_one_map((0, compute_repulsion(
                                       generate_obstacles(random.Random(0), 80, start=start_p, goal=goal_p,
                                                          L=L_test, max_cluster=MAX_CLUSTER,
                                                          overlap_margin=OVERLAP_MARGIN),
                                       start=start_p, goal=goal_p,
                                       robot_radius=const_p["robot_radius"], safety_margin=const_p["safety_margin"]),
                                   args.hppm_exe, const_p["step_size"], TIMEOUT, True,
                                   start_p, goal_p, const_p["robot_radius"], const_p["arrival_radius"],
                                   DEFAULT_M1, DEFAULT_M2))
    if "tray" not in test:
        raise SystemExit("ERROR: verify_maps.py is outdated (it does not return the trajectory). "
                         "Update it before using the repair.")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(repair_one_map, tasks, chunksize=8))
    dt = time.time() - t0

    detail, ok_idx, ok_maps = [], [], []
    for r in results:
        obs = r.pop("obs")
        if obs is not None:
            # BUG fixed: before, compute_repulsion(obs) was recomputed with
            # default start/goal (0,0)/(1,1) regardless of the real route
            # used for that map -- with ROUTE or L="auto" that wrote p_neto
            # with the sign of the wrong route. Now it uses the start/goal/
            # robot_radius that repair_one_map already returns per map.
            r_start = tuple(r.get("start", START))
            r_goal = tuple(r.get("meta", GOAL))
            r_step = r.get("step_size", STEP_SIZE_BASE)
            r_robot_radius = r.get("radio_robot")
            r_m1, r_m2 = r.get("m1", DEFAULT_M1), r.get("m2", DEFAULT_M2)
            ok_idx.append(r["idx"])
            ok_maps.append(compute_repulsion(obs, start=r_start, goal=r_goal,
                                             robot_radius=r_robot_radius))
            if FORMAT == "folders":
                d = os.path.join(args.output_dir, f"mapa_{r['idx']:05d}")
                os.makedirs(d, exist_ok=True)
                write_hppm_files(d, ok_maps[-1], r_step, r_start, r_goal, r_m1, r_m2)
        detail.append(r)
    if FORMAT == "npz":
        save_maps(args.output_dir, ok_idx, ok_maps)
    if not FULL_DETAIL:
        # with MAX_ITER = 0 and a large n, the detail of the non-repaired ones only takes up space
        detail = [r for r in detail if r["state"] == "success"]

    ok = [r for r in detail if r["state"] == "success"]
    runs = np.array([r["total_runs"] for r in detail])
    direct_rate = float(np.mean([r["success_before_repair"] for r in detail]))
    report = {
        "metodo": "generator + local repair",
        "num_obs": NUM_OBS or f"random {NUM_OBS_MIN}-{NUM_OBS_MAX}",
        "mapas_pedidos": args.n,
        "maps_success": len(ok),
        "tasa_reparados": round(len(ok) / args.n, 4),
        "success_rate_before_repair": round(direct_rate, 4),
        "max_iter": MAX_ITER,
        "gibbs_steps": GIBBS,
        "gibbs_acceptance_rate": round(float(np.mean([r["gibbs_accepted"] for r in ok])) / GIBBS, 3)
        if ok and GIBBS else None,
        "hppm_runs_per_success": round(float(runs.sum() / max(len(ok), 1)), 2),
        "hppm_runs_per_success_pure_rejection": round(1 / direct_rate, 2) if direct_rate > 0 else None,
        "median_iterations": float(np.median([r["iteraciones"] for r in ok])) if ok else None,
        "iteraciones_p90": float(np.percentile([r["iteraciones"] for r in ok], 90)) if ok else None,
        "mean_num_obs_success": round(float(np.mean([r["num_obs"] for r in ok])), 1) if ok else None,
        "mean_length": round(float(np.mean([r["length"] for r in ok])), 4) if ok else None,
        "length_std": round(float(np.std([r["length"] for r in ok])), 4) if ok else None,
        "time_sec": round(dt, 1),
        "seed": args.seed,
        "L": L,
        "ruta": args.route,
        "fixed_start_if_no_route": list(fixed_start),
        "meta_fijo_si_no_hay_ruta": list(fixed_goal),
        "k_max_cluster": MAX_CLUSTER,
        "margen_solape": OVERLAP_MARGIN,
        "m1": DEFAULT_M1,
        "m2": DEFAULT_M2,
    }
    with open(os.path.join(args.output_dir, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=1)
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
