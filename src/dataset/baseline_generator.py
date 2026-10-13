"""
Baseline: success rate of the ORIGINAL GENERATOR (1_generar_dataset.py)
evaluated with exactly the same pipeline as the diffusion maps
(geometric_filter, detect_collapse, compute_repulsion 'masa', fixed step).

It is the number the generative model has to beat. If diffusion does not
beat it, generating with the original script and filtering with HPPM is better.

Usage:
    python3 baseline_generator.py --hppm-exe artifacts/hppm_v2 --n 2000 --seeds 42 123 777
"""
import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.common.domain_scaling import (DEFAULT_OVERLAP_MARGIN, MAX_DENSITY, MIN_RADIUS, ROUTES,
                                       compute_compatible_L, scaled_constants, route_points)
from src.common.domain_scaling import max_radius as max_radius_L
from src.common.domain_scaling import (DEFAULT_L, DEFAULT_M1, DEFAULT_M2, DEFAULT_TIMEOUT)
from src.dataset.verify_maps import (compute_repulsion, detect_collapse, geometric_filter,
                                     verify_one_map, MIN_OBS_GENERATOR)


BINS = [(5, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

# RADIO_ROBOT=0.15 in 1_generar_dataset.py is actually the exclusion margin
# around start/goal (GENERATOR_EXCLUSION_RADIUS in
# verify_maps.py), not a physical robot radius. At L=1 (today's domain)
# the total margin is 0.15+0.02=0.17; at L>1 it scales via scaled_constants.
EXCLUSION_MARGIN = 0.17  # only for L=1 (compatibility); see generate_obstacles for L>1
# NOTE: 0.17 IS the sum GENERATOR_EXCLUSION_RADIUS_BASE (0.15) + START_GOAL_MARGIN_BASE
# (0.02), but it is kept as a literal on purpose: the float sum evaluates to
# 0.16999999999999998, not to 0.17 bit for bit, and this margin feeds the
# generated coordinates. One ULP is irrelevant geometrically, but this constant
# is part of the generator's exact reproduction contract, so it is not rewritten.

# "Same cluster" threshold for overlap control (new, it did not exist
# in the original generator): 0.0 = the disks cross geometrically
# (strict). 0.09 = the same criterion that compute_repulsion() in
# verify_maps.py uses to group repulsion signs
# (2*ROBOT_RADIUS+SAFETY_MARGIN), more conservative.
STRICT_OVERLAP_MARGIN = DEFAULT_OVERLAP_MARGIN

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
TIMEOUT = DEFAULT_TIMEOUT               # --timeout
START = [0.0, 0.0]                      # --start
GOAL = [1.0, 1.0]                       # --goal
# NOTICE: MAX_CLUSTER is None ON PURPOSE and deliberately differs from
# DEFAULT_MAX_CLUSTER (4) used by every other generator. This script replicates the
# ORIGINAL generator, which predates the cluster-size control: it IS the baseline the
# whole project is compared against, so it must not apply a filter the original did
# not have. Do not "fix" this to DEFAULT_MAX_CLUSTER.
MAX_CLUSTER = None                      # --max-cluster (None = no limit, original behavior)
OVERLAP_MARGIN = STRICT_OVERLAP_MARGIN  # --overlap-margin ('same cluster' threshold, see generate_obstacles)
L = str(DEFAULT_L)                      # --L ("auto" or a fixed number)
NUM_OBS = 0                             # --num-obs (0 = random n in [5, 100] per map)


def cluster_size_with_candidate(existing_obs: np.ndarray, candidate,
                                overlap_margin: float = STRICT_OVERLAP_MARGIN) -> int:
    """Size of the overlap cluster that `candidate` would have if it is added
    to `existing_obs`. Two disks are in the same cluster if
        center_distance - (r_i + r_j) < overlap_margin.
    Union-Find over existing_obs + candidate -- the same connectivity criterion
    that compute_repulsion() in verify_maps.py uses to
    homogenize signs, reused here to REJECT placements before
    they happen, instead of only labeling them afterwards."""
    if existing_obs is None or len(existing_obs) == 0:
        return 1
    all_obs = np.vstack([np.asarray(existing_obs, dtype=float), candidate])
    n = len(all_obs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(all_obs[i, 0] - all_obs[j, 0], all_obs[i, 1] - all_obs[j, 1])
            gap = dist - (all_obs[i, 2] + all_obs[j, 2])
            if gap < overlap_margin:
                union(i, j)

    new_root = find(n - 1)
    return sum(1 for i in range(n) if find(i) == new_root)


class _UnionFindSizes:
    """Union-Find WITH PERSISTENT STATE across attempts, so as not to recompute
    the whole cluster (O(n^2)) for every candidate -- see generate_obstacles.
    Each node == an obstacle already ACCEPTED, in the order in which it was added."""
    __slots__ = ("parent", "size")

    def __init__(self):
        self.parent = []
        self.size = []

    def new_node(self) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.size.append(1)
        return i

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj
            self.size[rj] += self.size[ri]


def generate_obstacles(rng: random.Random, n_obs=None, n_min: int = 5, n_max: int = 100,
                        start=None, goal=None, L: float = None,
                        route: str = "diagonal",
                        max_cluster: int = None, overlap_margin: float = STRICT_OVERLAP_MARGIN,
                        max_total_attempts: int = None):
    """Faithful copy of generar_mapa() in 1_generar_dataset.py (without writing
    files), plus three extensions:
      - Domain scaling (section 3 of the design): if L is None (default),
        it behaves as before (domain [0,1]x[0,1], max_radius(n,1)).
        If L is a number, the domain becomes [0,L]x[0,L], max_radius uses
        A=L^2, and ROBOT_RADIUS/exclusion margins scale with L (via
        domain_scaling.scaled_constants). If L == "auto", it is computed
        with domain_scaling.compute_compatible_L(n) -- the "growing map": the more
        obstacles, the bigger the domain, the same relative diversity as n=95.
      - start/goal: if they are not passed explicitly, they are derived from `route`
        ("diagonal", "diagonal_inverse", "horizontal", "vertical") and L via
        domain_scaling.route_points -- so there is no need to recompute by hand the
        corners/sides every time L changes with n.
      - max_cluster: if it is not None, it also rejects a candidate when its
        overlap cluster (see cluster_size_with_candidate) exceeds
        this size. The original had NO overlap control at all
        between obstacles -- this is a new rejection, not a replica.

    Attempt budget SHARED for the whole map (not per obstacle),
    just like the original: 1000 by default. With max_cluster active and/or a
    large L, rejection is more frequent and 1000 attempts may not be enough
    to complete n -- if you see systematically short maps, raise
    max_total_attempts. n_min/n_max: range for the random draw when
    n_obs is None (default (5,100), the range of the original dataset -- before
    it was hardcoded, one could not generate with a random n outside that
    range)."""
    n = n_obs or rng.randint(n_min, n_max)

    if L is None:
        L_effective = 1.0
        max_radius = math.sqrt(MAX_DENSITY / (n * math.pi))
        exclusion_margin = EXCLUSION_MARGIN  # 0.17, behavior identical to the original
    else:
        L_effective = compute_compatible_L(n) if L == "auto" else float(L)
        max_radius = max_radius_L(n, L_effective)
        const = scaled_constants(L_effective)
        exclusion_margin = const["generator_exclusion_radius"] + const["start_goal_margin"]

    if start is None or goal is None:
        start, goal = route_points(route, L_effective)

    obs, attempts = [], 0
    # 1000 (the original) is enough without max_cluster up to n~100. With
    # max_cluster active at n=1000/L~3.25, measured empirically: 1000
    # attempts falls short (incomplete maps), 3000 always reaches
    # (10/10 in the test). If you raise max_cluster or lower n, 1000 may
    # reach again -- this is a conservative floor, not a general law.
    cap = max_total_attempts if max_total_attempts is not None else (
        3000 if max_cluster is not None else 1000)
    # Incremental Union-Find: O(1) amortized per union, instead of recomputing
    # the cluster of ALL the obstacles (O(n^2)) for every candidate. The
    # neighbor search is still O(len(obs)) per candidate (without a spatial
    # index) -- at n=1000 that is O(attempts*n), several orders of magnitude
    # faster than O(attempts*n^2), good enough in practice.
    uf = _UnionFindSizes() if max_cluster is not None else None
    while len(obs) < n and attempts < cap:
        ox, oy = rng.uniform(0.0, L_effective), rng.uniform(0.0, L_effective)
        r = rng.uniform(MIN_RADIUS, max_radius)
        attempts += 1
        if math.hypot(ox - start[0], oy - start[1]) <= r + exclusion_margin:
            continue
        if math.hypot(ox - goal[0], oy - goal[1]) <= r + exclusion_margin:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--route", choices=sorted(ROUTES), default=None,
                   help="route family (it recomputes start/goal according to the L of each map). If it is "
                        "passed, it ignores --start/--goal. Without this, --start/--goal are used as is "
                        "(intended for fixed L, not for 'auto').")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    L_arg = L  # "auto" or something parseable to float
    fixed_start, fixed_goal = tuple(START), tuple(GOAL)
    rates, per_bin = [], {}
    for seed in args.seeds:
        rng = random.Random(seed)
        tasks, discarded = [], 0
        for i in range(args.n):
            n_i = NUM_OBS or rng.randint(5, 100)
            L_i = compute_compatible_L(n_i) if L_arg == "auto" else float(L_arg)
            if args.route:
                start_i, goal_i = route_points(args.route, L_i)
            else:
                start_i, goal_i = fixed_start, fixed_goal
            const = scaled_constants(L_i)

            obs = geometric_filter(
                generate_obstacles(rng, n_i, start=start_i, goal=goal_i, L=L_i,
                                   max_cluster=MAX_CLUSTER, overlap_margin=OVERLAP_MARGIN),
                start=start_i, goal=goal_i, L=L_i,
                robot_radius=const["generator_exclusion_radius"],
                start_goal_margin=const["start_goal_margin"])
            # NOTE: detect_collapse() (nn_threshold, extent_threshold) still
            # does NOT scale with L -- at L>1 its defaults (calibrated at L=1) can
            # give false positives/negatives. Validate empirically before
            # trusting the "discarded" count at L!=1 (see warning in the chat).
            if len(obs) < MIN_OBS_GENERATOR or detect_collapse(obs):
                discarded += 1
                continue
            # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
            tasks.append((i, compute_repulsion(obs, start=start_i, goal=goal_i, sign_mode="mass",
                                               robot_radius=const["robot_radius"],
                                               safety_margin=const["safety_margin"]),
                          args.hppm_exe, const["step_size"], TIMEOUT, False, start_i, goal_i,
                          const["robot_radius"], const["arrival_radius"], DEFAULT_M1, DEFAULT_M2))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            states = [r["state"] for _, r in ex.map(verify_one_map, tasks, chunksize=32)]
        rate = states.count("success") / args.n
        rates.append(rate)
        n_obs = np.array([len(t[1]) for t in tasks])
        ok = np.array([e == "success" for e in states])
        for lo, hi in BINS:
            m = (n_obs >= lo) & (n_obs <= hi)
            if m.any():
                per_bin.setdefault(f"{lo}-{hi}", []).append(float(ok[m].mean()))
        print(json.dumps({"seed": seed, "success_rate_over_total_generated": round(rate, 4),
                          "descartados": discarded,
                          "mean_num_obs_success": round(float(n_obs[ok].mean()), 1) if ok.any() else None,
                          "count": {e: states.count(e) for e in set(states)}}))

    print(json.dumps({"BASELINE_mean": round(float(np.mean(rates)), 4),
                      "std_across_seeds": round(float(np.std(rates, ddof=1)), 4) if len(rates) > 1 else None,
                      "n_per_seed": args.n, "L": L_arg, "ruta": args.route,
                      "fixed_start_if_no_route": list(fixed_start), "meta_fijo_si_no_hay_ruta": list(fixed_goal),
                      "k_max_cluster": MAX_CLUSTER, "margen_solape": OVERLAP_MARGIN,
                      "m1": DEFAULT_M1, "m2": DEFAULT_M2,
                      "rate_by_num_obs": {k: round(float(np.mean(v)), 4) for k, v in per_bin.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
