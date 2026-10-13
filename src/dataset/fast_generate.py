"""
fast_generate.py — Generation of universal tiles WITHOUT iterative
repair: each candidate is generated ONCE (pure geometric placement,
with no HPPM check during placement) and verified ONCE
against the 4*P port->center routes (stopping at the first one that fails). If it does not
pass, it is discarded -- the culprit obstacle is not resampled and that
candidate is not retried. The goal is raw generation speed and that the
model/generator faces the real distribution of random maps,
without repair "easing the path" toward
artificially solvable configurations.

Difference from universal_generator.py: that script REPAIRS (it identifies the
culprit obstacle of the first route that fails, resamples it, retries
from the beginning, up to its MAX_ITER times) -- among other things,
`MAX_ITER = 0` in that script already does the same as this one (one generation, one
filter), but it keeps the repair code unused (import of
culprit_obstacle, the resampling branch, etc.). This version removes it
entirely: there is no possible repair even if someone passes an extra flag, and the
hot path neither loads nor references that machinery.

Expected cost (measured in this conversation, same engine/check): acceptance
rate ~0.4-0.5% (1 universal per ~200-250 candidates). Much cheaper PER
CANDIDATE than repair (without repeating Newton-Raphson over
the same map again and again), but much more expensive PER SUCCESS in total
HPPM runs (~305-349 vs ~209.6 for repair, measured). This script
prioritizes the first number, not the second -- see the warning in the chat if the
goal is to maximize tiles/hour instead of the statistical purity of the sample.

Usage:
    python3 -m src.dataset.fast_generate --num-obs-min 1 --num-obs-max 50 --n 100000 \
        --hppm-exe artifacts/hppm_v2 --output-dir artifacts/tiles_rapido_s500 \
        --workers 30 --seed 500
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.common.domain_scaling import compute_L, scaled_constants
from src.common.domain_scaling import (DEFAULT_L, DEFAULT_M1, DEFAULT_M2, DEFAULT_MAX_CLUSTER, DEFAULT_OVERLAP_MARGIN, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_SEED, DEFAULT_TILE_N_MAX, DEFAULT_TIMEOUT)
from src.common.maps_io import save_maps
from src.dataset.universal_generator import _generate_candidate, ports_and_center
from src.dataset.verify_maps import compute_repulsion, verify_one_map

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
TIMEOUT = DEFAULT_TIMEOUT                  # --timeout
MAX_CLUSTER = DEFAULT_MAX_CLUSTER          # --max-cluster
OVERLAP_MARGIN = DEFAULT_OVERLAP_MARGIN    # --overlap-margin
L = str(DEFAULT_L)                         # --L ('auto' or a fixed number)
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE    # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN          # --port-margin
REPORT_EVERY_SEC = 15.0                    # --report-every-sec
NUM_OBS = 0                                # --num-obs (0 = random in [min, max] per map)
NUM_OBS_MIN = 1                            # --num-obs-min
NUM_OBS_MAX = DEFAULT_TILE_N_MAX          # --num-obs-max


def generate_and_filter(task):
    """ONE candidate: places the obstacles once (without HPPM), runs the
    4*P routes once (stopping at the first one that fails), returns the
    result. Without repair -- there is no import of culprit_obstacle, there
    is no resampling branch, there is no retry loop over the same map."""
    (idx, seed, n_i, L_spec, exe, timeout, k_max_cluster, overlap_margin,
     m1, m2, P, port_margin) = task
    rng = random.Random(seed)
    L_i = compute_L(n_i) if L_spec == "auto" else float(L_spec)
    const = scaled_constants(L_i)
    ports, center = ports_and_center(L_i, P)
    port_margin_i = port_margin * L_i

    obs = _generate_candidate(rng, n_i, L_i, ports, center, k_max_cluster, overlap_margin,
                              port_margin_i)
    if len(obs) < n_i:
        return {"idx": idx, "state": "incompleto", "num_obs": len(obs), "L": L_i}
    obs = obs.tolist()

    runs = 0
    for port_name, point in ports.items():
        rep = compute_repulsion(np.array(obs), start=point, goal=center,
                                robot_radius=const["robot_radius"],
                                safety_margin=const["safety_margin"])
        # return_traj=False: without repair there is no need to identify the
        # culprit obstacle, so the trajectory is not needed -- it saves
        # reading tray.txt on every run.
        _, res = verify_one_map((idx, rep, exe, const["step_size"], timeout, False,
                                 point, center, const["robot_radius"], const["arrival_radius"],
                                 m1, m2))
        runs += 1
        if res["state"] != "success":
            return {"idx": idx, "state": res["state"], "failure_at": port_name,
                    "num_obs": len(obs), "L": L_i, "corridas_hppm": runs, "obs": None}

    return {"idx": idx, "state": "success", "num_obs": len(obs), "L": L_i,
            "corridas_hppm": runs, "obs": np.array(obs)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--n", type=int, default=5000, help="candidates to ATTEMPT (not successes requested: "
                                                       "without repair it cannot be guaranteed to reach "
                                                       "a number of successes in bounded time)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--output-dir", default="artifacts/rapido")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    master_rng = random.Random(args.seed)
    tasks = []
    for i in range(args.n):
        n_i = NUM_OBS or master_rng.randint(NUM_OBS_MIN, NUM_OBS_MAX)
        seed_i = master_rng.getrandbits(63)
        # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
        tasks.append((i, seed_i, n_i, L, args.hppm_exe, TIMEOUT,
                      MAX_CLUSTER, OVERLAP_MARGIN, DEFAULT_M1, DEFAULT_M2,
                      PORTS_PER_SIDE, PORT_MARGIN))

    print(f"Generating {args.n} candidates (without repair, a single final filter) with {args.workers} workers...")
    t0 = time.time()
    ok_idx, ok_maps, detail = [], [], []
    n_incomplete = 0
    t_last_report = t0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(generate_and_filter, tasks, chunksize=32)):
            obs = r.pop("obs", None)
            if r["state"] == "success":
                ok_idx.append(r["idx"])
                ok_maps.append(obs)
            elif r["state"] == "incompleto":
                n_incomplete += 1
            detail.append(r)

            now = time.time()
            if now - t_last_report >= REPORT_EVERY_SEC or i + 1 == args.n:
                t_last_report = now
                elapsed = now - t0
                rate = len(ok_idx) / max(1, i + 1)
                print(f"  [progress] {i+1}/{args.n} candidates | {len(ok_idx)} successes "
                      f"(rate={rate:.4%}) | {(i+1)/max(elapsed,1e-9):.1f} candidates/s | "
                      f"{elapsed/60:.1f} min elapsed", flush=True)
    dt = time.time() - t0

    os.makedirs(args.output_dir, exist_ok=True)
    save_maps(args.output_dir, ok_idx, ok_maps)
    with open(os.path.join(args.output_dir, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, default=float)

    total_runs = sum(r["corridas_hppm"] for r in detail if "corridas_hppm" in r)
    report = {
        "metodo": f"fast generation without repair ({4 * PORTS_PER_SIDE} ports->center)",
        "candidates_attempted": args.n,
        "maps_success": len(ok_idx),
        "tasa_aceptacion": round(len(ok_idx) / args.n, 5),
        "candidates_incomplete": n_incomplete,
        "hppm_runs_total": total_runs,
        "hppm_runs_per_success": round(total_runs / max(1, len(ok_idx)), 2),
        "candidates_per_sec": round(args.n / dt, 1),
        "ports_per_side": PORTS_PER_SIDE,
        "margen_puertos": PORT_MARGIN,
        "time_sec": round(dt, 1),
        "workers": args.workers,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
