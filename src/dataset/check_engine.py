"""
check_engine.py -- Re-verifies a sample of tiles ALREADY KNOWN to be
universal (they come from a src/dataset/universal_generator.py folder, verified
with the engine that was used to generate the training dataset) against the
HPPM binary you pass it now.

If the binary changed behaviour (recalibration, new bug, different I/O
contract in pendientes.txt/tray.txt), these tiles -- which SHOULD always give
universal=True, by construction -- will start to fail. It serves to isolate
"the model generates badly" from "the engine changed" without spending GPU
time or retraining anything.

Usage:
    python3 check_engine.py --dir artifacts/universal_n01-50_P3_lote1_s142 \
        --hppm-exe artifacts/hppm_v2 -n 200
"""
import argparse
import json

import numpy as np

from src.common.domain_scaling import scaled_constants
from src.common.domain_scaling import (DEFAULT_M1, DEFAULT_M2, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_SEED, DEFAULT_TIMEOUT)
from src.common.maps_io import load_maps
from src.dataset.universal_generator import ports_and_center
from src.dataset.verify_maps import compute_repulsion, verify_one_map

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE  # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN        # --port-margin (accepted but never used by this script)
TIMEOUT = DEFAULT_TIMEOUT                # --timeout


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="folder from src/dataset/universal_generator.py (with maps.npz inside)")
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("-n", type=int, default=200, help="how many tiles to re-verify (at random)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    idxs, maps = load_maps(args.dir, columns=3)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(maps), size=min(args.n, len(maps)), replace=False)

    L = 1.0
    ports, center = ports_and_center(L, PORTS_PER_SIDE)
    const = scaled_constants(L)

    universal_count, failures_in, total_runs = 0, {}, 0
    for k in chosen:
        obs = maps[k]
        universal = True
        for port_name, point in ports.items():
            rep = compute_repulsion(obs, start=point, goal=center, sign_mode="mass",
                                    robot_radius=const["robot_radius"], safety_margin=const["safety_margin"])
            # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
            _, res = verify_one_map((0, rep, args.hppm_exe, const["step_size"], TIMEOUT, False,
                                     point, center, const["robot_radius"], const["arrival_radius"],
                                     DEFAULT_M1, DEFAULT_M2))
            total_runs += 1
            if res["state"] != "success":
                universal = False
                failures_in[port_name] = failures_in.get(port_name, 0) + 1
                failures_in[f"{port_name}_estado"] = res["state"]
                break
        universal_count += universal

    n_tested = len(chosen)
    print(f"{args.dir}: {n_tested} tiles re-verified (they WERE already universal with the original engine)")
    print(f"  with this HPPM ({args.hppm_exe}): {universal_count}/{n_tested} still report universal "
          f"({universal_count/n_tested:.1%})")
    print(f"  HPPM runs: {total_runs}")
    if failures_in:
        print(f"  first port that failed (when it failed): {json.dumps(failures_in, ensure_ascii=False)}")

    if universal_count / n_tested < 0.9:
        print("\n  *** The current HPPM engine does NOT behave like the one that generated the dataset. ***")
        print("  *** The 0% you saw in src/model/set_diffusion.py sample is the engine, not the model. ***")
    else:
        print("\n  The engine seems consistent with the one that generated the dataset -- the model's 0% "
              "has to be investigated elsewhere (it is not this).")


if __name__ == "__main__":
    main()
