"""
prepare_tiles.py -- Consolidates the folders of src/dataset/universal_generator.py
(one per batch, each with maps.npz + results_detail.json) into a single
datos_tiles.npz to train src/model/set_diffusion.py.

Unlike datos_conjuntos.npz (old scheme, with a variable L and a route per map),
here:
  - L is ALWAYS 1.0 (the universal-tiles scheme fixes SIDE=1.0, it does not
    grow with n -- see resumen_dataset_universal.md section 2).
  - There is no 'ruta' column: a universal tile is not conditioned to a
    particular route, by design (it solves the 4*P port->center routes at the
    same time).
  - maps.npz from src/dataset/universal_generator.py ONLY contains universal
    tiles (success by construction, just like src/dataset/repaired_generator.py)
    -- success=True for everything that goes in.

Output format (datos_tiles.npz):
    obs   float32 (N, n_max, 3)   x, y, r in [0,SIDE]x[0,SIDE], SIDE=1.0
    n     int16   (N,)
    success bool  (N,)            always True
    L     float32 (N,)            always 1.0 (it is stored anyway, in case
                                  tiles of another SIDE are ever mixed in)

Usage:
    python3 prepare_tiles.py --pattern "artifacts/universal_n01-50_P3_lote*" --n-max 64 \
        --output artifacts/datos_tiles.npz
"""
import argparse
import glob
import json
import os

import numpy as np

from src.common.compat import normalize
from src.common.maps_io import load_maps

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
N_MAX = 64  # --n-max: storage width; must cover the num_obs_max the tiles were generated with


def load_detail(folder):
    path = os.path.join(folder, "results_detail.json")
    with open(path, encoding="utf-8") as f:
        detail = normalize(json.load(f))
    return {int(r["idx"]): r for r in detail}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="artifacts/universal_*_lote*",
                   help="glob pattern of the batch folders (each one with maps.npz)")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output", default="artifacts/datos_tiles.npz")
    args = p.parse_args()

    folders = sorted(
        d for d in glob.glob(os.path.join(args.base_dir, args.pattern))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "maps.npz"))
    )
    if not folders:
        raise SystemExit(f"No folders '{args.pattern}' with maps.npz found under {args.base_dir!r}.")
    print(f"{len(folders)} folders found.")

    all_obs, all_n, all_L = [], [], []
    for folder in folders:
        idxs, maps = load_maps(folder, columns=3)
        detail = load_detail(folder)
        n_locals = [len(m) for m in maps]
        if n_locals and max(n_locals) > N_MAX:
            raise SystemExit(f"{folder}: map with {max(n_locals)} obstacles > --n-max={N_MAX}. "
                             "Raise --n-max.")
        L_locals = [float(detail.get(idx, {}).get("L", 1.0)) for idx in idxs]
        print(f"  {folder}: {len(maps)} maps | n in [{min(n_locals, default=0)},"
              f"{max(n_locals, default=0)}] | L in [{min(L_locals, default=1.0):.2f},"
              f"{max(L_locals, default=1.0):.2f}]")
        all_obs.extend(maps)
        all_n.extend(n_locals)
        all_L.extend(L_locals)

    N = len(all_obs)
    if N == 0:
        raise SystemExit("No usable map was left.")

    obs = np.zeros((N, N_MAX, 3), dtype=np.float32)
    for k, m in enumerate(all_obs):
        obs[k, :len(m)] = m
    n = np.asarray(all_n, dtype=np.int16)
    success = np.ones(N, dtype=bool)
    L_arr = np.asarray(all_L, dtype=np.float32)

    if not np.allclose(L_arr, 1.0):
        print(f"  WARNING: {(~np.isclose(L_arr, 1.0)).sum()} maps with L != 1.0 -- "
              f"was any of the batches generated with --L different from 1.0?")

    np.savez_compressed(args.output, obs=obs, n=n, success=success, L=L_arr)
    print(f"\nSaved {args.output}: {N} maps from {len(folders)} folders")
    print(f"  n_max={N_MAX} | mean num_obs {n.mean():.1f} (min {n.min()}, max {n.max()})")
    print(f"  size: {os.path.getsize(args.output) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
