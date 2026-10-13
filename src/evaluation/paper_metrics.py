"""
paper_metrics.py — Runs ALL the model evaluation tests for the paper into ONE
single folder (--output-dir), with summary.json and markdown tables ready to paste.

Tests (each one can be run alone with --tests <name>):
  1. rate_by_n       Verified success rate by n bin, with Wilson CI.
  2. cost            HPPM runs per useful tile, same units for the three
                     methods (repair / pure rejection / diffusion+filter).
  3. ddim_ablation   Success rate vs. number of DDIM steps (10/25/50/100/250).
  4. diversity       C2ST (real vs. generated), intra-set diversity (Chamfer)
                     and memorization (nearest neighbor to the training set).
  5. mosaics         Success rate of COMPLETE MOSAICS (not loose tiles), and
                     cell replacements needed.

Usage:
    python3 -m src.evaluation.paper_metrics --model-dir artifacts/ckpt_tiles_limpio_v1 \\
        --hppm-exe artifacts/hppm_v2 --training-data artifacts/tile-obstacle-maps.npz \\
        --reference-data artifacts/test_holdout.npz \\
        --output-dir artifacts/metrics_paper --workers 30

    python3 -m src.evaluation.paper_metrics ... --tests mosaics   # a single test

--reference-data is REQUIRED whenever the diversity test runs: it is the
held-out test split (the --test-out .npz that `src.model.set_diffusion train`
writes, default <dir of --data>/test_holdout.npz). There is no fallback to
--training-data on purpose -- falling back is exactly the contamination this
flag removes.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from src.dataset.verify_maps import MIN_OBS_GENERATOR
from src.common.domain_scaling import (MOSAIC_VERIFICATION_SCOPE, DEFAULT_GUIDANCE, DEFAULT_MAX_CLUSTER, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_SEED)
from src.common.compat import normalize

BINS_N = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
TESTS = ["rate_by_n", "cost", "ddim_ablation", "diversity", "mosaics"]

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
GUIDANCE = DEFAULT_GUIDANCE                    # --guidance
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE        # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN              # --port-margin
MAX_CLUSTER = DEFAULT_MAX_CLUSTER              # --max-cluster
MAX_CELL_RETRIES = 20                          # --max-cell-retries
TARGET_PER_CELL = 15.0                         # --target-per-cell
MOSAIC_OBSTACLE_CAP = 900                      # --mosaic-obstacle-cap
GPU_NAME = "RTX 4090"                          # --gpu-name
# Both cost inputs defaulted to None in the campaign runs too (the recorded
# summary.json _config has them null): leave them at None unless the real
# figures are known by hand.
DATASET_GENERATION_WORKER_HOURS = None         # --dataset-generation-worker-hours
TRAINING_GPU_HOURS = None                      # --training-gpu-hours

# Effective configuration of every parameter that is now a frozen constant:
# written into summary.json["_config"] so the audit trail of the paper's
# figures survives the freeze.
FROZEN_CONFIG = {
    "guidance": GUIDANCE,
    "ports_per_side": PORTS_PER_SIDE,
    "port_margin": PORT_MARGIN,
    "max_cluster": MAX_CLUSTER,
    "mosaic_obstacle_cap": MOSAIC_OBSTACLE_CAP,
    "max_cell_retries": MAX_CELL_RETRIES,
    "target_per_cell": TARGET_PER_CELL,
    "gpu_name": GPU_NAME,
    "training_gpu_hours": TRAINING_GPU_HOURS,
    "dataset_generation_worker_hours": DATASET_GENERATION_WORKER_HOURS,
}


# ==========================================================================
def wilson(k, n, z=1.96):
    """Wilson binomial CI (better than the normal one when k=0 or k=n)."""
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def hanley_mcneil(auc, n1, n2):
    """95% CI and two-sided p-value (H0: AUC=0.5) for a C2ST AUC, via the
    Hanley-McNeil (1982) variance estimator for the AUC of a binary
    classifier. A point estimate of 0.5288 on a small sample LOOKS close to
    the ideal 0.5, but without the CI there is no way to tell that apart from
    0.5288 on a sample large enough that even a small, genuine divergence from
    the real distribution could not hide inside sampling noise -- the CI
    (specifically, whether it contains 0.5) is what actually licenses the
    claim "indistinguishable from real", not the point estimate alone."""
    from scipy.stats import norm
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)
    se = var ** 0.5
    ci_lo, ci_hi = auc - 1.96 * se, auc + 1.96 * se
    z = (auc - 0.5) / se if se > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(ci_lo), float(ci_hi), float(p_value)


def run_command(cmd, description):
    print(f"\n{'='*70}\n>>> {description}\n{'='*70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {description} (returncode={r.returncode})")
    print(f"    ({time.time()-t0:.0f}s)")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return normalize(json.load(f))


def sample_tiles(args, output_dir, n, extra=None):
    """Calls ``src.model.set_diffusion sample`` with the common parameters.
    The sampling configuration (guidance, ports, margins, cluster limit) is
    frozen inside src.model.set_diffusion, so it is not passed here."""
    cmd = [sys.executable, "-m", "src.model.set_diffusion", "sample",
           "--dir", args.model_dir, "--n", str(n),
           "--hppm-exe", args.hppm_exe, "--output-dir", output_dir,
           "--workers", str(args.workers), "--seed", str(args.seed)]
    return cmd + (extra or [])


# ==========================================================================
# 1. Success rate per n bin
# ==========================================================================
def test_rate_by_n(args, base):
    test_dir = os.path.join(base, "1_rate_by_n")
    os.makedirs(test_dir, exist_ok=True)
    rows = []
    for lo, hi in BINS_N:
        out_dir = os.path.join(test_dir, f"n{lo:02d}-{hi:02d}")
        # The per-bin n bounds are passed through the child's --num-obs-min/max
        # flags (survivors on purpose: they are the "uniform n per bin" regime
        # of this test). Do not freeze them in src.model.set_diffusion.
        run_command(sample_tiles(args, out_dir, args.n_per_bin,
                                 ["--num-obs-min", str(lo), "--num-obs-max", str(hi)]),
                    f"[1/5 rate_by_n] bin n={lo}-{hi} ({args.n_per_bin} candidates)")
        report = read_json(os.path.join(out_dir, "report.json"))
        k = report["count_by_status"].get("success", 0)
        n_eval = report["mapas_evaluados"]
        rate, ci_lo, ci_hi = wilson(k, n_eval)
        d_cluster = report.get("mapas_descartados_cluster") or 0
        d_few = report.get("mapas_descartados_pocos") or 0
        d_collapsed = report.get("mapas_descartados_colapsados") or 0
        accounted = n_eval + d_cluster + d_few + d_collapsed
        rows.append({"bin_n": f"{lo}-{hi}", "successes": k, "evaluated": n_eval,
                     "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "discarded_cluster": d_cluster, "discarded_few": d_few,
                     "discarded_collapsed": d_collapsed,
                     "n_requested": args.n_per_bin,
                     "accounted": accounted,
                     "breakdown_sums_to_n_per_bin": accounted == args.n_per_bin,
                     "num_obs_evaluated_mean": report.get("mean_num_obs_evaluated"),
                     "sampling_time_sec": report.get("sampling_time_sec")})
    # Aggregate over ALL bins, with its own Wilson CI. This is what Table 1's
    # footer sums by hand; Table 2 uses a DIFFERENT regime (n drawn from the
    # checkpoint distribution), so the two aggregates are not expected to match.
    tot_k = sum(r["successes"] for r in rows)
    tot_eval = sum(r["evaluated"] for r in rows)
    tot_req = sum(r["n_requested"] for r in rows)
    a_rate, a_lo, a_hi = wilson(tot_k, tot_eval)
    r_rate, r_lo, r_hi = wilson(tot_k, tot_req)
    return {"bins": rows, "n_requested_per_bin": args.n_per_bin,
            "sampling_regime": "uniform_n_per_bin",
            "sampling_regime_note": ("n is drawn uniformly inside each bin, with the same number of "
                                     "candidates per bin. Its aggregate is NOT comparable with "
                                     "cost.diffusion_plus_filter.verified_success_rate, which draws "
                                     "n from the checkpoint's n distribution."),
            "aggregate_rate": {
                "successes": tot_k, "evaluated": tot_eval, "rate": a_rate,
                "ci_lo": a_lo, "ci_hi": a_hi,
                "denominator": "evaluated (candidates that reached HPPM verification)"},
            "aggregate_rate_over_requested": {
                "successes": tot_k, "requested": tot_req, "rate": r_rate,
                "ci_lo": r_lo, "ci_hi": r_hi,
                "denominator": "requested (every candidate asked of the model)"},
            "breakdown_check": {
                "all_bins_sum_to_n_per_bin": all(r["breakdown_sums_to_n_per_bin"] for r in rows),
                "formula": "evaluated + discarded_cluster + discarded_few + discarded_collapsed == n_requested"}}


# ==========================================================================
# 2. Compared cost (HPPM runs per useful tile)
# ==========================================================================
def test_cost(args, base):
    test_dir = os.path.join(base, "2_cost")
    os.makedirs(test_dir, exist_ok=True)
    results = {}

    # (a) pure rejection, measured right here (not quoted from another run)
    out_dir_pure = os.path.join(test_dir, "pure_rejection")
    # Its generation configuration (cluster limit, overlap margin, ports,
    # port margin) is frozen inside src.dataset.fast_generate.
    run_command([sys.executable, "-m", "src.dataset.fast_generate", "--hppm-exe", args.hppm_exe,
                 "--num-obs-min", "1", "--num-obs-max", "50", "--n", str(args.n_cost_rejection),
                 "--workers", str(args.workers), "--output-dir", out_dir_pure,
                 "--seed", str(args.seed)],
                f"[2/5 cost] pure rejection ({args.n_cost_rejection} candidates)")
    rp = read_json(os.path.join(out_dir_pure, "report.json"))
    t_pure = rp["time_sec"]
    n_succ_pure = rp["maps_success"]
    results["pure_rejection"] = {
        "hppm_runs_per_success": rp["hppm_runs_per_success"],
        "acceptance_rate": rp["tasa_aceptacion"],
        "candidates_per_sec": rp["candidates_per_sec"],
        "successes": rp["maps_success"], "attempted": rp["candidates_attempted"],
        "total_wall_time_sec": t_pure,
        "seconds_per_successful_tile": round(t_pure / n_succ_pure, 4) if n_succ_pure else None,
        "worker_seconds_per_successful_tile": (round(t_pure * args.workers / n_succ_pure, 4)
                                               if n_succ_pure else None),
        "workers": args.workers,
        "sampling_regime": "uniform_n_1_50",
        "time_note": ("end to end: random generation + geometric filter + HPPM verification, "
                      "wall-clock with `workers` processes. worker_seconds = wall x workers "
                      "(upper bound on CPU time, assumes the workers were saturated).")}

    # (b) diffusion + filter: derived from the detail of a sample_tiles run.
    # corridas_hppm is per candidate in results_detail.json (the verifier
    # stops at the first route that fails, so it is not a fixed 12).
    out_dir_dif = os.path.join(test_dir, "diffusion")
    run_command(sample_tiles(args, out_dir_dif, args.n_cost_diffusion),
                f"[2/5 cost] diffusion+filter ({args.n_cost_diffusion} candidates)")
    detail = read_json(os.path.join(out_dir_dif, "results_detail.json"))
    report_d = read_json(os.path.join(out_dir_dif, "report.json"))
    runs_total = sum(r.get("corridas_hppm", 0) for r in detail)
    n_success = sum(1 for r in detail if r["state"] == "success")
    t_samp = report_d.get("sampling_time_sec")
    t_filt = report_d.get("filter_time_sec")
    t_ver = report_d.get("verification_time_sec")
    t_ver_cpu = report_d.get("verification_cpu_time_sec")
    timings = [t_samp, t_filt, t_ver]
    t_total = round(sum(timings), 3) if all(t is not None for t in timings) else None
    results["diffusion_plus_filter"] = {
        "hppm_runs_per_success": round(runs_total / max(1, n_success), 2),
        "hppm_runs_total": runs_total,
        "successes": n_success, "candidates_evaluated": len(detail),
        "candidates_sampled": report_d.get("candidates_sampled", args.n_cost_diffusion),
        "verified_success_rate": report_d["verified_success_rate"],
        "sampling_regime": "checkpoint_n_distribution",
        "sampling_time_sec": t_samp,
        "filter_time_sec": t_filt,
        "verification_time_sec": t_ver,
        "verification_cpu_time_sec": t_ver_cpu,
        "total_wall_time_sec": t_total,
        "seconds_per_successful_tile": (round(t_total / n_success, 4)
                                        if t_total is not None and n_success else None),
        "worker_seconds_per_successful_tile": (
            round((t_samp + t_filt + t_ver_cpu) / n_success, 4)
            if None not in (t_samp, t_filt, t_ver_cpu) and n_success else None),
        "workers": report_d.get("workers", args.workers),
        "time_note": ("end to end = (a) DDIM sampling on the GPU + (b) geometric filtering + "
                      "(c) HPPM verification (wall-clock with `workers` processes). Model loading "
                      "excluded. worker_seconds = sampling + filter + summed per-candidate HPPM "
                      "CPU time, i.e. the figure that does not depend on the worker count."),
        "note": ("does not include the amortized cost of generating the training dataset "
                 "nor of training -- see section 11 of the report")}

    # (c) local repair: quoted from the original dataset (not re-measured, that would
    # mean regenerating the whole dataset); it is explicit that this is an inherited figure.
    results["local_repair"] = {
        "hppm_runs_per_success": 209.6,
        "source": "resumen_dataset_universal.md (generation of the 4 original batches)",
        "note": "inherited figure, not re-measured in this run"}
    results["cost_accounting"] = _cost_accounting(args, results)
    return results


def _parse_training_hours(log_path):
    """Hours of training wall-clock from a `set_diffusion train` stdout log: the
    last `| <x> min` printed on a `step ...` line. The training loop is not
    instrumented on purpose; it already prints elapsed minutes since the
    (re)start of the process, so a run resumed from ultimo.pt only reports the
    time of its LAST segment -- sum the segments by hand in that case."""
    import re
    last = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*step\s+\d+ \|.*\|\s*([0-9.]+) min", line)
            if m:
                last = float(m.group(1))
    return None if last is None else last / 60.0


def _cost_accounting(args, results):
    """The four separate cost items reviewer 3.1 asks for, each in its OWN unit
    (they are deliberately never added up into one number)."""
    train_h, train_src = None, None
    if TRAINING_GPU_HOURS is not None:
        train_h, train_src = TRAINING_GPU_HOURS, "--training-gpu-hours (manual)"
    elif args.training_log and os.path.exists(args.training_log):
        train_h = _parse_training_hours(args.training_log)
        train_src = f"parsed from {args.training_log} (last 'step ... | N min' line)"
    dif = results.get("diffusion_plus_filter", {})
    n_ok = dif.get("successes", 0)
    n_cand = max(1, dif.get("candidates_sampled") or 1)
    t_inf, t_ver = dif.get("sampling_time_sec"), dif.get("verification_time_sec")
    t_ver_cpu = dif.get("verification_cpu_time_sec")
    return {
        "dataset_generation_worker_hours": {
            "value": DATASET_GENERATION_WORKER_HOURS,
            "unit": "CPU worker-hours, one-off, amortized over every model trained on this dataset",
            "source": ("--dataset-generation-worker-hours (known figure of the leveling campaign)"
                       if DATASET_GENERATION_WORKER_HOURS is not None else
                       "NOT PROVIDED -- pass --dataset-generation-worker-hours")},
        "training_gpu_hours": {
            "value": None if train_h is None else round(train_h, 3),
            "unit": f"hours on one {GPU_NAME} GPU, wall-clock, one training run (this seed)",
            "source": train_src or "NOT PROVIDED -- pass --training-log <train stdout> or --training-gpu-hours"},
        "inference_seconds_per_tile": {
            "per_candidate_sampled": None if t_inf is None else round(t_inf / n_cand, 5),
            "per_successful_tile": None if not n_ok or t_inf is None else round(t_inf / n_ok, 5),
            "unit": "GPU wall-clock seconds of DDIM sampling (+ geometric filter in `filter_seconds_*`)",
            "filter_seconds_per_successful_tile": (
                None if not n_ok or dif.get("filter_time_sec") is None else round(dif["filter_time_sec"] / n_ok, 5))},
        "verification_seconds_per_tile": {
            "per_candidate_wall": None if t_ver is None else round(t_ver / n_cand, 5),
            "per_successful_tile_wall": None if not n_ok or t_ver is None else round(t_ver / n_ok, 5),
            "per_successful_tile_cpu": None if not n_ok or t_ver_cpu is None else round(t_ver_cpu / n_ok, 5),
            "workers": dif.get("workers"),
            "unit": "seconds of HPPM verification; `wall` depends on the worker count, `cpu` does not"},
        "speedup_note": ("The reported 'speedup' is the REDUCTION IN HPPM CALLS PER USEFUL TILE "
                         "(cost.pure_rejection.hppm_runs_per_success / "
                         "cost.diffusion_plus_filter.hppm_runs_per_success) under this experiment's "
                         "conditions. It is NOT a speedup of the whole process: it excludes dataset "
                         "generation and training, which are the four items above, in their own "
                         "units, and are never summed here.")}


# ==========================================================================
# 3. DDIM steps ablation
# ==========================================================================
def test_ddim_ablation(args, base):
    test_dir = os.path.join(base, "3_ddim_ablation")
    os.makedirs(test_dir, exist_ok=True)
    rows = []
    for steps in args.ddim_steps_to_test:
        out_dir = os.path.join(test_dir, f"ddim{steps:04d}")
        t0 = time.time()
        run_command(sample_tiles(args, out_dir, args.n_ablation, ["--ddim-steps", str(steps)]),
                    f"[3/5 ddim_ablation] {steps} steps ({args.n_ablation} candidates)")
        report = read_json(os.path.join(out_dir, "report.json"))
        k = report["count_by_status"].get("success", 0)
        n_eval = report["mapas_evaluados"]
        rate, ci_lo, ci_hi = wilson(k, n_eval)
        rows.append({"ddim_steps": steps, "successes": k, "evaluated": n_eval,
                     "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "sampling_regime": "checkpoint_n_distribution",
                     "sampling_time_sec": report.get("sampling_time_sec"),
                     "filter_time_sec": report.get("filter_time_sec"),
                     "verification_time_sec": report.get("verification_time_sec"),
                     "total_test_time_sec": round(time.time() - t0, 1)})
    return {"steps_tested": args.ddim_steps_to_test, "results": rows,
            "sampling_regime": "checkpoint_n_distribution"}


# ==========================================================================
# 4. Diversity / realism / memorization
# ==========================================================================
def _tile_features(obs):
    """Per-tile features, route-agnostic (unlike
    analizar_generated.py, which measures occupancy of the diagonal band -- it
    does not apply to universal tiles)."""
    x, y, r = obs[:, 0], obs[:, 1], obs[:, 2]
    n = len(obs)
    if n < 2:
        return [n, r.mean(), 0.0, float((np.pi * r ** 2).sum()), 0.0, 0.0, 0.0, 0.0, 0.0]
    d = np.linalg.norm(obs[:, None, :2] - obs[None, :, :2], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    cells = np.clip((x * 4).astype(int), 0, 3) * 4 + np.clip((y * 4).astype(int), 0, 3)
    cnt = np.bincount(cells, minlength=16)
    expected = n / 16
    return [n, float(r.mean()), float(r.std()), float((np.pi * r ** 2).sum()),
            float(nn.mean()), float(nn.min()),
            float(((cnt - expected) ** 2 / max(expected, 1e-9)).sum()),
            float((cnt > 0).mean()),
            float(np.hypot(x - 0.5, y - 0.5).mean())]


def _chamfer(a, b):
    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a[:, :2]), cKDTree(b[:, :2])
    return 0.5 * (tb.query(a[:, :2])[0].mean() + ta.query(b[:, :2])[0].mean())


def _load_memorization_pool(args):
    """Training tiles that back the nearest-neighbour index of the memorization
    test.

    The file passed to ``set_diffusion train --data`` (``--training-data`` here)
    still CONTAINS the held-out test rows -- the split only stops using them, it
    does not delete them. If the test holdout sidecar records which source rows
    are test, they are removed from the index here; otherwise every test tile
    would sit at distance 0 from itself, the p5 threshold would go back to ~0 and
    the memorization fraction would be meaningless again -- exactly the
    contamination this change removes.
    """
    mem = np.load(args.training_data)
    n_mem, obs_mem = mem["n"], mem["obs"]
    meta_path = (args.reference_data or "") + ".meta.json"
    if args.reference_data and os.path.exists(meta_path):
        from src.common.maps_io import file_sha256
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        same = os.path.abspath(args.training_data) == meta.get("data")
        if not same and int(meta.get("data_rows", -1)) == len(n_mem):
            same = file_sha256(args.training_data) == meta.get("data_sha256")
        if same and "test_indices" in meta:
            keep = np.ones(len(n_mem), dtype=bool)
            keep[np.asarray(meta["test_indices"], dtype=np.int64)] = False
            print(f"    memorization index: {int(keep.sum())} training tiles from "
                  f"{args.training_data} "
                  f"({len(n_mem) - int(keep.sum())} held-out test rows excluded)")
            return obs_mem[keep], n_mem[keep]
        print(f"    WARNING: --training-data is not the dataset the test holdout came from "
              f"({meta_path} -> {meta.get('data')}); its test rows cannot be excluded.")
    else:
        print(f"    WARNING: no sidecar {meta_path}; cannot prove the held-out test rows are "
              f"absent from --training-data. If they are present, memorization is optimistic.")
    return obs_mem, n_mem


def test_diversity(args, base):
    from scipy.spatial import cKDTree
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from src.common.maps_io import load_maps
    from src.common.tile_descriptors import occupancy_hist, occupancy_hists

    if not args.reference_data:
        raise SystemExit(
            "missing --reference-data: do not use the training dataset as the real reference "
            "(that contaminated C2ST/Chamfer/memorization). Pass the held-out test split written "
            "by `src.model.set_diffusion train` (--test-out, default <dir of --data>/"
            "test_holdout.npz), e.g. --reference-data artifacts/test_holdout.npz")
    if not os.path.exists(args.reference_data):
        raise SystemExit(f"--reference-data {args.reference_data} does not exist.")

    test_dir = os.path.join(base, "4_diversity")
    os.makedirs(test_dir, exist_ok=True)

    out_dir_gen = os.path.join(test_dir, "generated")
    run_command(sample_tiles(args, out_dir_gen, args.n_diversity),
                f"[4/5 diversity] generating {args.n_diversity} candidates")
    idxs, maps = load_maps(out_dir_gen, columns=3)
    detail = {int(r["idx"]): r for r in read_json(os.path.join(out_dir_gen, "results_detail.json"))}
    gen = [m for i, m in zip(idxs, maps) if detail.get(i, {}).get("state") == "success"]
    print(f"    {len(gen)} universal tiles generated for the analysis")
    if len(gen) < 50:
        return {"error": f"only {len(gen)} universal tiles generated, too few to analyze"}

    # REAL reference: the held-out TEST split, never the training dataset.
    d = np.load(args.reference_data)
    n_col, obs_col = d["n"], d["obs"]
    rng = np.random.default_rng(args.seed)
    # reference: same n profile as the generated ones, sampled from the test holdout
    n_gen = np.array([len(m) for m in gen])
    sel = []
    for n_obj in n_gen[:args.n_reference]:
        cands = np.where(n_col == n_obj)[0]
        if len(cands):
            sel.append(int(rng.choice(cands)))
    ref = [obs_col[i, :int(n_col[i])] for i in sel]
    print(f"    {len(ref)} real reference tiles (same n profile) from the held-out test split "
          f"{args.reference_data}")

    Fg = np.array([_tile_features(m) for m in gen])
    Fr = np.array([_tile_features(m) for m in ref])
    X = np.vstack([Fg, Fr])
    y = np.r_[np.ones(len(Fg)), np.zeros(len(Fr))]
    pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), X, y, cv=5,
                           method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, pr))
    auc_ci_lo, auc_ci_hi, auc_p_value = hanley_mcneil(auc, len(Fg), len(Fr))
    # without num_obs: the model generates the n it is asked for, it is not a learned property
    pr2 = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), X[:, 1:], y, cv=5,
                            method="predict_proba")[:, 1]
    auc_without_n = float(roc_auc_score(y, pr2))
    auc_wo_ci_lo, auc_wo_ci_hi, auc_wo_p_value = hanley_mcneil(auc_without_n, len(Fg), len(Fr))

    pairs = min(args.chamfer_pairs, len(gen) * (len(gen) - 1) // 2)
    div_gen = float(np.mean([_chamfer(gen[i], gen[j]) for i, j in
                             [(int(rng.integers(len(gen))), int(rng.integers(len(gen))))
                              for _ in range(pairs)] if i != j]))
    div_real = float(np.mean([_chamfer(ref[i], ref[j]) for i, j in
                              [(int(rng.integers(len(ref))), int(rng.integers(len(ref))))
                               for _ in range(pairs)] if i != j]))

    # memorization: grid occupancy descriptor, nearest neighbour to the TRAINING
    # set (that is what "memorized" means). The threshold is the p5 of the
    # held-out test tiles' distance to that training set -- a genuine "how close
    # do real, unseen tiles get?" scale, instead of the ~0 the contaminated
    # reference used to produce.
    mem_obs, mem_n = _load_memorization_pool(args)
    n_train = min(args.memorization_samples, len(mem_n))
    train_idx = rng.choice(len(mem_n), size=n_train, replace=False)
    H = occupancy_hists(mem_obs, mem_n, rows=train_idx)
    tree = cKDTree(H)
    d_gen = tree.query(np.array([occupancy_hist(m) for m in gen]))[0]
    d_ref = tree.query(np.array([occupancy_hist(m) for m in ref]))[0]
    threshold = float(np.percentile(d_ref, 5))

    return {
        "sampling_regime": "checkpoint_n_distribution",
        "n_generated_analyzed": len(gen), "n_reference": len(ref),
        "reference_data": os.path.abspath(args.reference_data),
        "C2ST_auc": round(auc, 4),
        "C2ST_auc_ci95": [round(auc_ci_lo, 4), round(auc_ci_hi, 4)],
        "C2ST_auc_p_value": round(auc_p_value, 4),
        "C2ST_auc_without_num_obs": round(auc_without_n, 4),
        "C2ST_auc_ci95_without_num_obs": [round(auc_wo_ci_lo, 4), round(auc_wo_ci_hi, 4)],
        "C2ST_auc_p_value_without_num_obs": round(auc_wo_p_value, 4),
        "C2ST_interpretation": ("~0.5 = indistinguishable from real (good); ~1.0 = the "
                                "classifier separates them perfectly (the model does not capture "
                                "the real distribution). The 95% CI (Hanley-McNeil) must CONTAIN "
                                "0.5 to claim 'indistinguishable from real' -- a point estimate "
                                "near 0.5 whose CI excludes it is evidence of a real, if small, "
                                "difference, not noise."),
        "chamfer_diversity": {"generated": round(div_gen, 5), "real": round(div_real, 5),
                              "interpretation": ("generated << real suggests mode collapse "
                                                 "(little variety)")},
        "memorization": {
            "nn_generated_to_train_median": round(float(np.median(d_gen)), 5),
            "nn_reference_test_to_train_median": round(float(np.median(d_ref)), 5),
            "threshold_p5_test_to_train": round(threshold, 5),
            "frac_generated_closer_than_real_p5": round(float((d_gen < threshold).mean()), 4),
            "n_train_compared": int(n_train),
            "training_index_data": os.path.abspath(args.training_data),
            "interpretation": ("high fraction = the model is copying tiles from the training set "
                               "instead of generating new ones; the threshold is where the closest "
                               "5% of held-out real tiles to the training set sits")},
    }


# ==========================================================================
# 5. Complete mosaics
# ==========================================================================
def test_mosaics(args, base):
    test_dir = os.path.join(base, "5_mosaics")
    os.makedirs(test_dir, exist_ok=True)
    rows = []
    for side in args.mosaic_sizes:
        cells = side * side
        # automatic target: moderate average per cell (high per-tile success
        # rate zone, see table 1), capped at --mosaic-obstacle-cap (by
        # default 900, with margin below the engine's real MAX_OBSTACULOS=1000).
        # WITHOUT this, without --objetivo-obstaculos each cell draws its n from the
        # checkpoint distribution (which averages high) and large mosaics
        # always hit the engine ceiling, regardless of the
        # repair budget -- that is what happened in the real run, it was not a
        # limitation of the model/solver, it was this bug.
        target = min(int(TARGET_PER_CELL * cells), MOSAIC_OBSTACLE_CAP)
        if target < cells * MIN_OBS_GENERATOR:
            print(f"  WARNING: {side}x{side} ({cells} cells) does not even fit the minimum "
                  f"({cells * MIN_OBS_GENERATOR}) under --mosaic-obstacle-cap="
                  f"{MOSAIC_OBSTACLE_CAP} -- skipping this size.")
            continue
        for rep_i in range(args.mosaic_repetitions):
            out_dir = os.path.join(test_dir, f"{side}x{side}_rep{rep_i}")
            # The mosaic generation/verification configuration (ports, margins,
            # cluster limit, guidance, cell-retry budget, plot) is frozen inside
            # src.pipeline.assembler; --target-obstacles is computed per size here
            # and must stay a flag there (frozen at None it would silently switch
            # every cell to the checkpoint-n regime and break this battery).
            cmd = [sys.executable, "-m", "src.pipeline.assembler", "--model-dir", args.model_dir,
                   "--rows", str(side), "--columns", str(side),
                   "--target-obstacles", str(target),
                   "--hppm-exe", args.hppm_exe,
                   "--workers", str(args.workers),
                   "--seed", str(args.seed + rep_i),
                   "--output-dir", out_dir]
            print(f"\n{'='*70}\n>>> [5/5 mosaics] {side}x{side} rep {rep_i+1}/"
                  f"{args.mosaic_repetitions}\n{'='*70}", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd)
            dt = time.time() - t0
            report_path = os.path.join(out_dir, "mosaic_report.json")
            if r.returncode != 0 or not os.path.exists(report_path):
                # the assembler exits with SystemExit if a cell cannot be generated,
                # or if the target would end up exceeding MAX_OBSTACULOS_MOTOR
                rows.append({"side": side, "rep": rep_i, "final_state": "generation_failure",
                             "target_obstacles": target, "n_obstacles": None,
                             "retries": None, "first_try": False, "time_sec": round(dt, 1)})
                print(f"    *** failure (returncode={r.returncode}) -- counted as a failure ***")
                continue
            report_mosaic = read_json(report_path)
            rows.append({"side": side, "rep": rep_i, "final_state": report_mosaic["final_state"],
                         "target_obstacles": target, "n_obstacles": report_mosaic["n_obstacles"],
                         "retries": report_mosaic.get("cell_retries_used"),
                         "first_try": bool(report_mosaic.get("solved_first_try", False)),
                         "time_sec": round(dt, 1)})

    summary = {}
    for side in args.mosaic_sizes:
        sub = [f for f in rows if f["side"] == side]
        ok = [f for f in sub if f["final_state"] == "success"]
        first = [f for f in ok if f["first_try"]]
        repaired = [f for f in ok if not f["first_try"]]
        rate, ci_lo, ci_hi = wilson(len(ok), len(sub))
        b_rate, b_lo, b_hi = wilson(len(first), len(sub))
        retries = [f["retries"] for f in ok if f["retries"] is not None]
        summary[f"{side}x{side}"] = {
            "target_obstacles": sub[0]["target_obstacles"] if sub else None,
            "valid_mosaics": len(ok), "attempted": len(sub),
            # AFTER repairs (any number of cell replacements <= max_cell_retries)
            "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "rate_after_repair": {"successes": len(ok), "attempted": len(sub),
                                  "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi},
            # BEFORE repairs: valid on the first HPPM verification of the assembled mosaic
            "rate_before_repair": {"successes": len(first), "attempted": len(sub),
                                   "rate": b_rate, "ci_lo": b_lo, "ci_hi": b_hi},
            "resolved_first_try": len(first), "resolved_after_repair": len(repaired),
            "retries_mean": round(float(np.mean(retries)), 2) if retries else None,
            "retries_max": int(max(retries)) if retries else None,
            "n_obstacles_mean": round(float(np.mean([f["n_obstacles"] for f in ok])), 1) if ok else None,
            "mean_time_sec": round(float(np.mean([f["time_sec"] for f in sub])), 1) if sub else None}
    return {"detail": rows, "summary_by_size": summary,
            "max_cell_retries": MAX_CELL_RETRIES,
            "repetitions": args.mosaic_repetitions,
            "sampling_regime": f"fixed_total_target (~{TARGET_PER_CELL:g} obstacles/cell)",
            "verification_scope": MOSAIC_VERIFICATION_SCOPE,
            "ci_note": ("Wilson 95% CI. k/k successes is NOT a mathematical guarantee: it only "
                        "bounds the true rate below by the CI's lower limit.")}


# ==========================================================================
def write_tables(res, path):
    """Markdown tables ready to paste into the paper."""
    L = ["# Model metrics — tables for the paper\n"]

    if "rate_by_n" in res and "bins" in res["rate_by_n"]:
        L.append("\n## 1. Verified success rate by density (n)\n")
        L.append("| n | successes/evaluated | rate [95% CI] | discarded by cluster |")
        L.append("|---|---|---|---|")
        for f in res["rate_by_n"]["bins"]:
            if f["rate"] is None:
                L.append(f"| {f['bin_n']} | - | no data | - |")
            else:
                L.append(f"| {f['bin_n']} | {f['successes']}/{f['evaluated']} | "
                         f"{f['rate']:.1%} [{f['ci_lo']:.1%}, {f['ci_hi']:.1%}] | "
                         f"{f['discarded_cluster']} |")
        ag = res["rate_by_n"].get("aggregate_rate")
        if ag and ag["rate"] is not None:
            L.append(f"| **all bins** | {ag['successes']}/{ag['evaluated']} | "
                     f"{ag['rate']:.2%} [{ag['ci_lo']:.2%}, {ag['ci_hi']:.2%}] | |")
        L.append(f"\nSampling regime: `{res['rate_by_n'].get('sampling_regime')}` (uniform n per bin) -- "
                 f"the aggregate is NOT comparable with Table 2's, which uses "
                 f"`checkpoint_n_distribution`.")

    if "cost" in res:
        c = res["cost"]
        L.append("\n## 2. Compared cost (HPPM runs per useful tile)\n")
        L.append("| Method | HPPM runs / useful tile | acceptance rate | note |")
        L.append("|---|---|---|---|")
        if "local_repair" in c:
            L.append(f"| Local repair | {c['local_repair']['hppm_runs_per_success']} "
                     f"| 72.3% | {c['local_repair']['note']} |")
        if "pure_rejection" in c:
            r = c["pure_rejection"]
            L.append(f"| Pure rejection | {r['hppm_runs_per_success']} | "
                     f"{r['acceptance_rate']:.2%} | measured in this run |")
        if "diffusion_plus_filter" in c:
            dd = c["diffusion_plus_filter"]
            L.append(f"| **Diffusion + filter** | **{dd['hppm_runs_per_success']}** | "
                     f"{dd['verified_success_rate']:.2%} | does not include training cost |")
        L.append("\n| Method | wall-clock s / successful tile (end to end) | worker-s / successful tile |")
        L.append("|---|---|---|")
        for key, label in (("pure_rejection", "Pure rejection"), ("diffusion_plus_filter", "Diffusion + filter")):
            if key in c:
                L.append(f"| {label} | {c[key].get('seconds_per_successful_tile')} | "
                         f"{c[key].get('worker_seconds_per_successful_tile')} |")
        if "diffusion_plus_filter" in c:
            dd = c["diffusion_plus_filter"]
            L.append(f"\nDiffusion+filter breakdown (s): sampling {dd.get('sampling_time_sec')} | "
                     f"filter {dd.get('filter_time_sec')} | HPPM verification {dd.get('verification_time_sec')} "
                     f"(wall, {dd.get('workers')} workers).")
        if "cost_accounting" in c:
            L.append(f"\n> {c['cost_accounting']['speedup_note']}")

    if "ddim_ablation" in res and "results" in res["ddim_ablation"]:
        L.append("\n## 3. Ablation: DDIM steps\n")
        L.append("| steps | successes/evaluated | rate [95% CI] | sampling time (s) |")
        L.append("|---|---|---|---|")
        for f in res["ddim_ablation"]["results"]:
            L.append(f"| {f['ddim_steps']} | {f['successes']}/{f['evaluated']} | "
                     f"{f['rate']:.1%} [{f['ci_lo']:.1%}, {f['ci_hi']:.1%}] | "
                     f"{f['sampling_time_sec']} |")

    if "diversity" in res and "C2ST_auc" in res["diversity"]:
        d = res["diversity"]
        mem = d.get("memorization", {})
        L.append("\n## 4. Generative quality\n")
        L.append(f"Real reference: held-out test split `{os.path.basename(d.get('reference_data', '?'))}` "
                 f"(NOT the training dataset).\n")
        ci = d.get("C2ST_auc_ci95")
        ci_wo = d.get("C2ST_auc_ci95_without_num_obs")
        ci_txt = f"[{ci[0]}, {ci[1]}]" if ci else "-"
        ci_wo_txt = f"[{ci_wo[0]}, {ci_wo[1]}]" if ci_wo else "-"
        L.append("| Metric | Value | 95% CI (Hanley-McNeil) | p-value (H0: AUC=0.5) | Interpretation |")
        L.append("|---|---|---|---|---|")
        L.append(f"| C2ST AUC | {d['C2ST_auc']} | {ci_txt} | {d.get('C2ST_auc_p_value', '-')} | "
                 f"0.5 = indistinguishable from real; CI must contain 0.5 to claim it |")
        L.append(f"| C2ST AUC (without num_obs) | {d['C2ST_auc_without_num_obs']} | {ci_wo_txt} | "
                 f"{d.get('C2ST_auc_p_value_without_num_obs', '-')} | same, without the controlled variable |")
        L.append(f"| Chamfer diversity (gen) | {d['chamfer_diversity']['generated']} | - | - | "
                 f"vs. real: {d['chamfer_diversity']['real']} |")
        L.append(f"| Memorization (frac < real p5) | {mem.get('frac_generated_closer_than_real_p5')} "
                 f"| - | - | high = copying the training set (p5 threshold "
                 f"{mem.get('threshold_p5_test_to_train')}) |")

    if "mosaics" in res and "summary_by_size" in res["mosaics"]:
        L.append("\n## 5. Complete mosaics\n")
        L.append("Verification: ONLY the corner-to-corner route (1 HPPM route), NOT the 12-route "
                 "per-tile criterion. Wilson 95% CI; k/k is not a guarantee.\n")
        L.append("| Size | target obstacles | before repair | after repair [95% CI] | resolved 1st try / after repair | cell replacements (mean/max) | real obstacles | mean time |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, v in res["mosaics"]["summary_by_size"].items():
            rate_txt = (f"{v['rate']:.0%} [{v['ci_lo']:.0%}, {v['ci_hi']:.0%}]"
                        if v["rate"] is not None else "-")
            if v["attempted"] == 0:
                L.append(f"| {k} | - | 0/0 (skipped, does not fit under --mosaic-obstacle-cap) | "
                         "- | - | - | - |")
                continue
            bf = v["rate_before_repair"]
            L.append(f"| {k} | {v.get('target_obstacles', '-')} | "
                     f"{bf['successes']}/{bf['attempted']} ({bf['rate']:.0%} "
                     f"[{bf['ci_lo']:.0%}, {bf['ci_hi']:.0%}]) | "
                     f"{v['valid_mosaics']}/{v['attempted']} {rate_txt} | "
                     f"{v['resolved_first_try']}/{v['resolved_after_repair']} | "
                     f"{v['retries_mean']}/{v['retries_max']} | "
                     f"{v['n_obstacles_mean']} | {v['mean_time_sec']}s |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--hppm-exe", required=True)
    p.add_argument("--training-data", default="artifacts/tile-obstacle-maps.npz",
                   help="dataset the model was trained on (the same --data passed to "
                        "`set_diffusion train`). It is NO LONGER the real reference: it is only the "
                        "nearest-neighbour index of the memorization test (its held-out test rows "
                        "are excluded when the test-holdout sidecar allows it).")
    p.add_argument("--reference-data", default=None,
                   help="held-out TEST split .npz written by `src.model.set_diffusion train` "
                        "(--test-out, default <dir of --data>/test_holdout.npz). REQUIRED whenever "
                        "the diversity test runs: it is the real reference for C2ST, Chamfer "
                        "diversity and the memorization threshold. Never falls back to "
                        "--training-data.")
    p.add_argument("--output-dir", default="artifacts/metrics_paper")
    p.add_argument("--tests", nargs="+", choices=TESTS, default=None,
                   help="run only these tests (default: all)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # sample sizes per test
    p.add_argument("--n-per-bin", type=int, default=6000, help="test 1: candidates per n bin")
    p.add_argument("--n-cost-rejection", type=int, default=200000, help="test 2: pure rejection candidates")
    p.add_argument("--n-cost-diffusion", type=int, default=5000, help="test 2: diffusion candidates")
    p.add_argument("--n-ablation", type=int, default=3000, help="test 3: candidates per ddim value")
    p.add_argument("--ddim-steps-to-test", type=int, nargs="+", default=[10, 25, 50, 100, 250])
    p.add_argument("--n-diversity", type=int, default=8000, help="test 4: candidates to generate")
    p.add_argument("--n-reference", type=int, default=1000, help="test 4: real reference tiles")
    p.add_argument("--chamfer-pairs", type=int, default=2000)
    p.add_argument("--memorization-samples", type=int, default=20000, help="test 4: training tiles to index")
    p.add_argument("--mosaic-sizes", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--mosaic-repetitions", type=int, default=30,
                   help="independent mosaics per size. 30/30 gives a Wilson 95%% CI of "
                        "[88.6%%, 100%%]; 10/10 only [72.3%%, 100%%]")
    # cost accounting (reviewer 3.1)
    p.add_argument("--training-log", default=None,
                   help="stdout of `set_diffusion train` for THIS run; training GPU-hours are "
                        "parsed from its last 'step ... | N min' line")
    args = p.parse_args()

    tests = args.tests or TESTS
    if "diversity" in tests and not args.reference_data:
        # Hard failure (before anything runs), on purpose: there is no safe
        # default. Using the training dataset as the "real" reference is the
        # contamination this flag exists to prevent.
        raise SystemExit(
            "missing --reference-data: the diversity test must compare against a held-out test "
            "split, not against the training dataset. Generate it once with "
            "`python3 -m src.model.set_diffusion train --data <dataset> ...` (it writes "
            "<dir of --data>/test_holdout.npz or whatever you pass to --test-out) and then pass "
            "e.g. --reference-data artifacts/test_holdout.npz. See "
            "src/evaluation/check_split_leakage.py to verify the split first.")
    base = args.output_dir
    os.makedirs(base, exist_ok=True)
    print(f"Tests to run: {', '.join(tests)}")
    print(f"Everything goes to {base}/\n")

    summary_path = os.path.join(base, "summary.json")
    res = read_json(summary_path) if os.path.exists(summary_path) else {}
    # vars(args) no longer sees the frozen parameters: they are written back
    # explicitly so summary.json keeps the exact effective configuration the
    # campaign ran with (same keys as the published artifacts).
    res.setdefault("_config", {}).update({**vars(args), **FROZEN_CONFIG})

    t0 = time.time()
    functions = {"rate_by_n": test_rate_by_n, "cost": test_cost,
                 "ddim_ablation": test_ddim_ablation, "diversity": test_diversity,
                 "mosaics": test_mosaics}
    for name in tests:
        try:
            res[name] = functions[name](args, base)
        except SystemExit as e:
            print(f"\n*** {name} FAILED: {e} -- saving what there is and continuing ***")
            res[name] = {"error": str(e)}
        except Exception as e:
            print(f"\n*** {name} FAILED ({type(e).__name__}): {e} -- continuing ***")
            res[name] = {"error": f"{type(e).__name__}: {e}"}
        # save after EVERY test: if something gets cut off, the previous one is not lost
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=float)
        write_tables(res, os.path.join(base, "tables.md"))
        print(f"\n--- {name} done, summary.json and tables.md updated "
              f"({(time.time()-t0)/60:.1f} min accumulated) ---")

    print(f"\n{'='*70}\nALL DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  {base}/summary.json   (all raw numbers)")
    print(f"  {base}/tables.md      (markdown tables for the paper)")
    print(f"  {base}/1_rate_by_n/ 2_cost/ 3_ddim_ablation/ 4_diversity/ 5_mosaics/  (raw outputs)")


if __name__ == "__main__":
    main()
