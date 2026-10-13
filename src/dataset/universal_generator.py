"""
universal_generator.py — Tiles solvable from the 4 sides AT THE SAME TIME, not by
separate routes mixed afterwards.

Why "generating diagonal + vertical, mixing without a label
to train" is not enough: a map that was verified ONLY against diagonal has
no guarantee of also solving horizontal -- mixing two sets
verified separately gives a model that learns a mixture of styles,
not universal maps. This script filters in the generation itself: a map
is only accepted if the 4 routes succeed ON THE SAME MAP.

How to verify that cheaply: instead of checking the 6 possible pairs between
4 ports (N-S, E-W, and the 4 corners), it is checked that the 4 ports
ALL reach a shared central point -- by geometric transitivity
(N->center and S->center imply that N->center->S exists) 4
verifications are enough, not 6. And since the center is invariant under a 90
degree rotation (only the ports rotate among themselves), ONE single canonical
generation already
covers the 4 orientations -- no need to rotate anything afterwards.

It repairs (like repaired_generator.py), it does not only reject: requiring that 4*P routes
succeed AT THE SAME TIME by pure chance is impractical with pure rejection
(with P=3 -> 12 routes, even with 70% individual success per route, the
joint probability is ~1.4% -- measured in practice: 0 successes in 535k
candidates). The repair attacks ONE failure at a time: it runs the routes in
order, stops at the first one that fails, resamples the culprit obstacle of
THAT route, and verifies EVERYTHING again from the beginning (fixing one route
can break another already solved one). There is no theoretical guarantee that this
always converges, but in practice (see repaired_generator.py for the
1-route case) it usually does so in a few iterations.

Usage:
    python3 universal_generator.py --num-obs-min 1 --num-obs-max 50 --n 5000 \
        --hppm-exe artifacts/hppm_v2 --output-dir universal_n01-50 \
        --workers 30 --seed 42
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.dataset.baseline_generator import STRICT_OVERLAP_MARGIN
from src.common.domain_scaling import compute_L, scaled_constants
from src.dataset.repaired_generator import culprit_obstacle
from src.common.maps_io import save_maps
from src.dataset.verify_maps import compute_repulsion, verify_one_map
from src.common.domain_scaling import (DEFAULT_L, DEFAULT_M1, DEFAULT_M2, DEFAULT_MAX_CLUSTER, DEFAULT_PORT_MARGIN, DEFAULT_PORTS_PER_SIDE, DEFAULT_SEED, DEFAULT_TILE_N_MAX, DEFAULT_TIMEOUT)

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
MAX_ITER = 60                              # --max-iter
TIMEOUT = DEFAULT_TIMEOUT                  # --timeout
MAX_CLUSTER = DEFAULT_MAX_CLUSTER          # --max-cluster
OVERLAP_MARGIN = STRICT_OVERLAP_MARGIN     # --overlap-margin
L = str(DEFAULT_L)                         # --L ('auto' or a fixed number)
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE    # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN          # --port-margin
REPORT_EVERY_SEC = 15.0                    # --report-every-sec
NUM_OBS = 0                                # --num-obs (0 = random in [min, max] per map)
NUM_OBS_MIN = 1                            # --num-obs-min
NUM_OBS_MAX = DEFAULT_TILE_N_MAX          # --num-obs-max


def ports_and_center(L: float, P: int = 1):
    """4*P ports (P per each N,S,E,W side) and the center of the domain
    [0,L]x[0,L]. P=1: a single port per side, at the midpoint -- the minimum,
    and only comparable against maps verified the same way. P=3 (what the
    training dataset uses, and the value of PORTS_PER_SIDE): ports at
    25%/50%/75% of the side, it gives redundancy
    (if one crossing comes out congested in a generated map, there are two other
    available on that same side) at the cost of 4*P port-center
    verifications per candidate instead of 4 (see cost analysis:
    2P^2 if ALL the pairs between sides are requested, or 2P if only aligned
    pairs -- here, with the shared-center scheme, the real cost is
    linear in P: 4*P verifications, not quadratic, because each port
    is only compared against the center, not against the other ports).
    It is still invariant under a 90 degree rotation: rotating permutes the
    4 groups of P ports among themselves, the center does not move.
    NOTE: the P=1 parameter default is kept for callers that pass P explicitly;
    the frozen PORTS_PER_SIDE is 3, see domain_scaling.DEFAULT_PORTS_PER_SIDE."""
    c = L / 2
    if P == 1:
        positions = [c]
    else:
        # P equally spaced points inside the side, without touching the corners
        # (it avoids overlapping with the exclusion margin of the neighboring sides)
        positions = [L * (i + 1) / (P + 1) for i in range(P)]
    ports = {}
    for i, pos in enumerate(positions):
        ports[f"N{i}"] = (pos, L)
        ports[f"S{i}"] = (pos, 0.0)
        ports[f"E{i}"] = (L, pos)
        ports[f"O{i}"] = (0.0, pos)
    return ports, (c, c)


def _sample_one_obstacle(rng, max_radius, protected, exclusion_margin, existing_obs, uf,
                         max_cluster, overlap_margin, L, max_attempts=2000):
    """A single valid obstacle (outside the protected points, and if
    max_cluster is not None, without forming an overlap cluster that is
    too large). Used both to complete the initial map and
    to REPAIR a culprit obstacle without having to redo the whole map."""
    import math
    from src.dataset.baseline_generator import MIN_RADIUS
    for _ in range(max_attempts):
        ox, oy = rng.uniform(0.0, L), rng.uniform(0.0, L)
        r = rng.uniform(MIN_RADIUS, max_radius)
        if any(math.hypot(ox - px, oy - py) <= r + exclusion_margin for px, py in protected):
            continue
        if max_cluster is not None and existing_obs:
            neighbors = [j for j, (jx, jy, jr) in enumerate(existing_obs)
                         if math.hypot(ox - jx, oy - jy) - (r + jr) < overlap_margin]
            roots = {uf.find(j) for j in neighbors} if uf else set()
            if uf and 1 + sum(uf.size[root] for root in roots) > max_cluster:
                continue
            return [ox, oy, r], neighbors
        return [ox, oy, r], []
    return None, None


def _generate_candidate(rng, n, L, ports, center, max_cluster, overlap_margin, port_margin):
    """A candidate of obstacles with exclusion around THE 5 PROTECTED POINTS
    (4 ports + center), not only 2 (start/goal) as in
    generate_obstacles(). Reusing generate_obstacles by calling it once per
    each port-center pair and keeping the most demanding result is not
    correct (each call would generate a DIFFERENT map) -- instead,
    a single map is generated and the obstacles that invade
    any of the 5 points are filtered by hand, reusing the logic of generate_obstacles
    for radius/density but with the generalized exclusion check."""
    import math
    from src.dataset.baseline_generator import MIN_RADIUS
    from src.common.domain_scaling import max_radius as max_radius_L

    max_radius = max_radius_L(n, L)
    # BEFORE: it reused generator_exclusion_radius + start_goal_margin (0.17 at L=1),
    # calibrated for the 2-point scheme (start/goal). With 4*P+1 protected
    # points (P=3 -> 13 points), that margin is larger than the distance
    # between neighboring ports of the same side -- it leaves the map with no valid place
    # for large obstacles, especially at small n (where max_radius is larger).
    # This margin is only so that an obstacle is not born on top of a port/center;
    # the real clearance for the robot is guaranteed by HPPM in the verification
    # (ROBOT_RADIUS), this prefilter does not need to be so generous.
    exclusion_margin = port_margin
    protected = list(ports.values()) + [center]

    obs, attempts = [], 0
    cap = 3000 if max_cluster is not None else 1000
    from src.dataset.baseline_generator import _UnionFindSizes
    uf = _UnionFindSizes() if max_cluster is not None else None
    while len(obs) < n and attempts < cap:
        ox, oy = rng.uniform(0.0, L), rng.uniform(0.0, L)
        r = rng.uniform(MIN_RADIUS, max_radius)
        attempts += 1
        if any(math.hypot(ox - px, oy - py) <= r + exclusion_margin for px, py in protected):
            continue
        neighbors = []
        if uf is not None and obs:
            for j, (jx, jy, jr) in enumerate(obs):
                if math.hypot(ox - jx, oy - jy) - (r + jr) < overlap_margin:
                    neighbors.append(j)
            roots = {uf.find(j) for j in neighbors}
            if 1 + sum(uf.size[root] for root in roots) > max_cluster:
                continue
        obs.append([ox, oy, r])
        if uf is not None:
            new_idx = uf.new_node()
            for j in neighbors:
                uf.union(new_idx, j)
    return np.array(obs, dtype=np.float32)


def verify_universal(task):
    """Generates a candidate and REPAIRS it against the 4*P routes port->center
    (instead of rejecting the whole map at the first failure, as in the previous
    version). Pure rejection needs the 4*P routes successful by pure luck
    in a single shot -- with P=3 (12 routes) that is ~p^12, impractical even
    with a high individual p (see warning in the chat: 0 successes in 535k attempts
    with the rejection version). The repair attacks ONE failure at a time: in each
    iteration it runs the routes in order, it stops at the first one that fails,
    identifies the culprit obstacle of THAT route and resamples it, and it checks
    again from the beginning (fixing one route can break another already
    solved one, that is why everything is re-verified, not only the one that failed)."""
    (idx, seed, n_i, L_spec, exe, timeout, max_cluster, overlap_margin,
     m1, m2, P, port_margin, max_iter) = task
    rng = random.Random(seed)
    L_i = compute_L(n_i) if L_spec == "auto" else float(L_spec)
    const = scaled_constants(L_i)
    ports, center = ports_and_center(L_i, P)
    port_margin_i = port_margin * L_i
    protected = list(ports.values()) + [center]

    from src.common.domain_scaling import max_radius as max_radius_L
    max_radius = max_radius_L(n_i, L_i)

    obs = _generate_candidate(rng, n_i, L_i, ports, center, max_cluster, overlap_margin,
                              port_margin_i)
    if len(obs) < n_i:
        return {"idx": idx, "state": "incompleto", "num_obs": len(obs), "L": L_i}
    obs = obs.tolist()

    total_runs = 0
    for it in range(max_iter + 1):
        failure = None
        for port_name, point in ports.items():
            rep = compute_repulsion(np.array(obs), start=point, goal=center,
                                    robot_radius=const["robot_radius"],
                                    safety_margin=const["safety_margin"])
            _, res = verify_one_map((idx, rep, exe, const["step_size"], timeout, True,
                                     point, center, const["robot_radius"], const["arrival_radius"],
                                     m1, m2))
            total_runs += 1
            if res["state"] != "success":
                failure = (port_name, res)
                break
        if failure is None:
            return {"idx": idx, "state": "success", "num_obs": len(obs), "L": L_i,
                    "corridas_hppm": total_runs, "iteraciones": it, "obs": np.array(obs)}

        port_name, res = failure
        if it == max_iter:
            return {"idx": idx, "state": "no_reparado", "failure_at": port_name,
                    "num_obs": len(obs), "L": L_i, "corridas_hppm": total_runs, "obs": None}

        k = culprit_obstacle(np.array(obs), res, const["robot_radius"])
        if k is None:
            return {"idx": idx, "state": "no_reparado", "failure_at": port_name,
                    "num_obs": len(obs), "L": L_i, "corridas_hppm": total_runs, "obs": None}

        others = obs[:k] + obs[k + 1:]
        uf_dummy = None  # no cluster control in the repair (same criterion as repaired_generator.py)
        new, _ = _sample_one_obstacle(rng, max_radius, protected, port_margin_i,
                                      others, uf_dummy, None, overlap_margin, L_i)
        if new is None:
            return {"idx": idx, "state": "no_reparado", "failure_at": port_name,
                    "num_obs": len(obs), "L": L_i, "corridas_hppm": total_runs, "obs": None}
        obs[k] = new


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--n", type=int, default=5000, help="UNIVERSAL maps requested (success)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--output-dir", default="artifacts/universal")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    master_rng = random.Random(args.seed)
    t0 = time.time()
    t_last_report = t0
    ok_idx, ok_maps, detail = [], [], []
    idx = 0
    total_attempts = 0
    n_incomplete = 0  # they did not even get to run HPPM (generation did not complete the map)

    def _print_progress(forced=False):
        nonlocal t_last_report
        now = time.time()
        if not forced and now - t_last_report < REPORT_EVERY_SEC:
            return
        t_last_report = now
        elapsed = now - t0
        current_rate = len(ok_idx) / max(1, total_attempts)
        incomplete_fraction = n_incomplete / max(1, total_attempts)
        candidates_per_sec = total_attempts / max(elapsed, 1e-9)
        remaining_now = args.n - len(ok_idx)
        eta_sec = (remaining_now / (current_rate * candidates_per_sec)) if current_rate > 0 and candidates_per_sec > 0 else float("inf")
        eta_txt = f"{eta_sec/60:.0f} min" if eta_sec != float("inf") else "?"
        warning_msg = ""
        if total_attempts > 2000 and incomplete_fraction > 0.5:
            warning_msg = "  *** WARNING: >50% 'incompleto' -- generation almost never completes the map, " \
                          "check PORT_MARGIN / PORTS_PER_SIDE before continuing ***"
        print(f"  [progress] {len(ok_idx)}/{args.n} successes | {total_attempts} candidates tried "
              f"({incomplete_fraction:.0%} incomplete) | rate={current_rate:.4f} | "
              f"{candidates_per_sec:.1f} candidates/s | elapsed={elapsed/60:.1f} min | "
              f"ETA~{eta_txt}{warning_msg}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while len(ok_idx) < args.n:
            remaining = args.n - len(ok_idx)
            batch = []
            for _ in range(remaining * 3):  # it over-requests, the universal acceptance rate is lower
                n_i = NUM_OBS or master_rng.randint(NUM_OBS_MIN, NUM_OBS_MAX)
                seed_i = master_rng.getrandbits(63)
                # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
                batch.append((idx, seed_i, n_i, L, args.hppm_exe, TIMEOUT,
                              MAX_CLUSTER, OVERLAP_MARGIN, DEFAULT_M1, DEFAULT_M2,
                              PORTS_PER_SIDE, PORT_MARGIN, MAX_ITER))
                idx += 1
            for r in ex.map(verify_universal, batch, chunksize=8):
                total_attempts += 1
                if r["state"] == "success":
                    ok_idx.append(r["idx"])
                    ok_maps.append(r["obs"])
                elif r["state"] == "incompleto":
                    n_incomplete += 1
                r.pop("obs", None)
                detail.append(r)
                _print_progress()
                if len(ok_idx) >= args.n:
                    break
    _print_progress(forced=True)

    os.makedirs(args.output_dir, exist_ok=True)
    save_maps(args.output_dir, ok_idx, ok_maps)
    with open(os.path.join(args.output_dir, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    dt = time.time() - t0
    rate = len(ok_idx) / max(1, total_attempts)
    failures_by_port = {}
    for r in detail:
        if r["state"] == "no_reparado" and "failure_at" in r:
            failures_by_port[r["failure_at"]] = failures_by_port.get(r["failure_at"], 0) + 1

    report = {
        "metodo": f"universal generator (rejection, {4 * PORTS_PER_SIDE} ports->center)",
        "maps_success": len(ok_idx), "candidates_tried": total_attempts,
        "universal_acceptance_rate": round(rate, 4),
        "candidates_incomplete": n_incomplete,
        "frac_incompletos": round(n_incomplete / max(1, total_attempts), 4),
        "hppm_runs_per_success": round(sum(r.get("corridas_hppm", 0) for r in detail) / max(1, len(ok_idx)), 2),
        "failures_per_port": failures_by_port,
        "ports_per_side": PORTS_PER_SIDE,
        "margen_puertos": PORT_MARGIN,
        "time_sec": round(dt, 1), "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
