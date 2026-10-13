"""
clean_tiles.py -- Re-verifies EACH tile of the src/dataset/universal_generator.py
batches against the current HPPM engine and collision check (hppm_v2 + the
segment-to-segment fix of verify_maps.py), and builds a datos_tiles.npz with
only the ones that are still universal.

Why it is needed: the 500k tiles were labelled with the old point-to-point
collision check (with a known false negative: it may fail to detect that the
trajectory jumps over an entire obstacle between two converged points).
Measured on a sample (check_engine.py, 300 tiles): ~6.3% of the tiles no
longer pass the 12 routes with the corrected check.

Unlike check_engine.py (which is diagnostic only, over a sample), this script
processes ALL the tiles and writes the clean .npz ready for prepare_tiles.py
-- no, better: it writes the final .npz itself (same format as
prepare_tiles.py, so the two do not have to be run).

Usage:
    python3 clean_tiles.py --pattern "artifacts/universal_n01-50_P3_lote*" \
        --hppm-exe artifacts/hppm_v2 --n-max 50 --workers 30 \
        --output artifacts/tile-obstacle-maps.npz
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.common.maps_io import load_maps
from src.common.domain_scaling import (DEFAULT_M1, DEFAULT_M2, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_TILE_N_MAX, DEFAULT_TIMEOUT)

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE  # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN        # --port-margin (accepted but never used by this script)
TIMEOUT = DEFAULT_TIMEOUT                # --timeout
REPORT_EVERY = 20000                     # --report-every
N_MAX = DEFAULT_TILE_N_MAX               # --n-max


def _verify_one_tile(args_tuple):
    """args_tuple: (global_idx, obs (n,3) list, hppm_exe, step_size, timeout,
    ports, center, robot_radius, safety_margin, arrival_radius, m1, m2).
    Stops at the first route that fails -- just like
    set_diffusion._verify_universal_candidate, reimplemented here
    standalone so as not to depend on importing torch."""
    from src.dataset.verify_maps import compute_repulsion, verify_one_map
    (idx, obs, exe, step_size, timeout, ports, center,
     robot_radius, safety_margin, arrival_radius, m1, m2) = args_tuple
    obs_arr = np.asarray(obs, dtype=np.float64)
    for point in ports.values():
        rep = compute_repulsion(obs_arr, start=point, goal=center, sign_mode="mass",
                                robot_radius=robot_radius, safety_margin=safety_margin)
        _, res = verify_one_map((idx, rep, exe, step_size, timeout, False,
                                 point, center, robot_radius, arrival_radius, m1, m2))
        if res["state"] != "success":
            return idx, False
    return idx, True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="artifacts/universal_*_lote*")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--output", default="artifacts/tile-obstacle-maps.npz")
    args = p.parse_args()

    from src.dataset.universal_generator import ports_and_center
    from src.common.domain_scaling import scaled_constants

    L = 1.0
    ports, center = ports_and_center(L, PORTS_PER_SIDE)
    const = scaled_constants(L)

    folders = sorted(
        d for d in glob.glob(os.path.join(args.base_dir, args.pattern))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "maps.npz"))
    )
    if not folders:
        raise SystemExit(f"No folders '{args.pattern}' with maps.npz found under {args.base_dir!r}.")
    print(f"{len(folders)} folders found.")

    all_obs, origin = [], []  # origin: (folder, local_idx) for the per-folder report
    for folder in folders:
        idxs, maps = load_maps(folder, columns=3)
        for local_idx, m in zip(idxs, maps):
            all_obs.append(m)
            origin.append(folder)
    N = len(all_obs)
    print(f"{N} tiles in total to re-verify (this runs the 12 routes on each one -- it may take a while).")

    tasks = [(i, all_obs[i].tolist(), args.hppm_exe, const["step_size"], TIMEOUT,
              ports, center, const["robot_radius"], const["safety_margin"],
              const["arrival_radius"], DEFAULT_M1, DEFAULT_M2) for i in range(N)]

    t0 = time.time()
    universal = np.zeros(N, dtype=bool)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for idx, ok in ex.map(_verify_one_tile, tasks, chunksize=16):
            universal[idx] = ok
            done += 1
            if done % REPORT_EVERY == 0:
                dt = time.time() - t0
                print(f"  {done}/{N} verified ({done/dt:.0f}/s) | "
                      f"{universal[:done].sum()} survive ({universal[:done].mean():.1%}) | "
                      f"{dt/60:.1f} min elapsed", flush=True)
    dt_total = time.time() - t0

    print(f"\nDone in {dt_total/60:.1f} min: {universal.sum()}/{N} tiles are still universal "
          f"({universal.mean():.1%})")

    # breakdown by source folder, to see whether the contamination is even
    import collections
    per_folder_total = collections.Counter(origin)
    per_folder_ok = collections.Counter(c for c, ok in zip(origin, universal) if ok)
    for c in folders:
        tot, ok = per_folder_total[c], per_folder_ok[c]
        print(f"  {c}: {ok}/{tot} ({ok/tot:.1%})")

    ok_idx = np.where(universal)[0]
    n_final = len(ok_idx)
    obs_out = np.zeros((n_final, N_MAX, 3), dtype=np.float32)
    n_out = np.zeros(n_final, dtype=np.int16)
    for k, i in enumerate(ok_idx):
        m = all_obs[i]
        obs_out[k, :len(m)] = m
        n_out[k] = len(m)
    success_out = np.ones(n_final, dtype=bool)
    L_out = np.ones(n_final, dtype=np.float32)

    np.savez_compressed(args.output, obs=obs_out, n=n_out, success=success_out, L=L_out)
    print(f"\nSaved {args.output}: {n_final} clean tiles "
          f"({os.path.getsize(args.output) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
