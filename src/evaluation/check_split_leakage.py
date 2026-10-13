"""
check_split_leakage.py -- near-duplicate leakage check between the held-out
TEST split (``test_holdout.npz``, written by ``src/model/set_diffusion.py
train``) and the TRAINING set.

WHY THIS EXISTS
---------------
Two reviewers asked for it: two almost identical obstacle configurations split
across train and test inflate EVERY generative-quality metric (C2ST, Chamfer
diversity, memorization) because the "unseen" reference is not really unseen.
Run this ONCE right after generating the test holdout and BEFORE any serious
training; if a bin comes out hot, re-draw the holdout with another partition
before trusting any number that uses it.

WHAT IT MEASURES
----------------
For every test tile it computes the distance to its nearest neighbour in the
training set, with the SAME descriptor as the memorization test: the 16x16 grid
occupancy weighted by ``r**2`` and normalized to sum 1 (see
``src/common/tile_descriptors.py``). "Same ruler" is deliberate: leakage and
memorization must not disagree about what a similar tile is.

The near-duplicate threshold is the ``PERCENTILE`` constant (5) percentile of
the TRAINING tiles' leave-one-out nearest-neighbour distance: a test tile whose
nearest training neighbour is closer than that is as close to training as the
closest 5% of training tiles are to each other, i.e. a near-duplicate. On a
clean i.i.d. split the expected fraction below that threshold is ~5%, so the
report shows the per-n-bin fraction AND that null, and flags any bin above
``ALERT_FRAC`` (0.05) explicitly -- never hidden inside an aggregate.

The threshold is computed SEPARATELY PER n-BIN (not once globally). Sparse
tiles (n<=10) occupy only a handful of the k*k descriptor cells, so their
leave-one-out distances to each other are naturally smaller than those of
dense tiles -- a single global threshold judged sparse tiles against the
typical closeness of dense ones and flagged ~35% of n=1-10 as "leakage" when
spot-checking raw obstacle coordinates showed ~95% of those were unrelated
tiles that happened to land in the same coarse cell. Each bin is judged
against its OWN population's typical nearest-neighbour closeness instead.

Usage:
    python3 -m src.evaluation.check_split_leakage \\
        --data artifacts/tile-obstacle-maps.npz \\
        --test artifacts/test_holdout.npz

    # if you already have a file with ONLY the training tiles:
    python3 -m src.evaluation.check_split_leakage \\
        --train-data artifacts/train_only.npz --test artifacts/test_holdout.npz
"""
import argparse
import json
import os
import time

import numpy as np

from src.common.maps_io import file_sha256
from src.common.tile_descriptors import (OCCUPANCY_BINS, nearest_distance,
                                          nearest_other_distance, occupancy_hists)
from src.common.domain_scaling import DEFAULT_SEED

# Same bins as paper_metrics.BINS_N so both reports can be read side by side.
BINS_N = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
# k of the k x k occupancy descriptor, same descriptor as the memorization test.
# FROZEN AT 16: the old argument's help text said "default 8" and was
# obsolete and wrong (the argument's default was already OCCUPANCY_BINS = 16 in
# src/common/tile_descriptors.py; the paper describes a 16x16 descriptor).
DESCRIPTOR_BINS = 16
PERCENTILE = 5.0   # --percentile: near-duplicate threshold percentile of the training LOO distances
ALERT_FRAC = 0.05  # --alert-frac: a bin above this fraction below the threshold is reported as leakage


def _load_test(test_path):
    """Returns ``(n, obs, success, success_key, L)``. ``success``/``L`` can be
    ``None`` if the file does not have those columns -- callers that only need
    ``n``/``obs`` (the descriptor) ignore them; ``--dedup-out`` needs the full
    set to write a subset with the SAME columns as the source file."""
    if not os.path.exists(test_path):
        raise SystemExit(f"--test {test_path} does not exist.")
    with np.load(test_path) as d:
        files = set(d.files)
        if "obs" not in files or "n" not in files:
            raise SystemExit(f"{test_path}: expected the columns 'obs' and 'n'.")
        n = d["n"].astype(np.int64)
        obs = np.asarray(d["obs"], dtype=np.float32)
        success_key = next((k for k in ("success", "exito") if k in files), None)
        success = np.asarray(d[success_key]) if success_key else None
        L = np.asarray(d["L"]) if "L" in files else None
        return n, obs, success, success_key, L


def _load_train_pool(args, n_test):
    """Return (n_train, obs_train). Either the explicit --train-data file, or
    --data with the rows that went to the test holdout removed."""
    if args.train_data:
        with np.load(args.train_data) as d:
            n_tr, obs_tr = d["n"].astype(np.int64), np.asarray(d["obs"], dtype=np.float32)
        print(f"Training pool: {len(n_tr)} tiles from --train-data {args.train_data} (used as-is).")
        return n_tr, obs_tr

    with np.load(args.data) as d:
        n_all, obs_all = d["n"].astype(np.int64), np.asarray(d["obs"], dtype=np.float32)

    meta_path = args.test + ".meta.json"
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"--data needs the test-holdout sidecar {meta_path} to know which rows are test, and "
            f"it is missing. Either re-run `src.model.set_diffusion train` with the current code "
            f"(it writes the sidecar next to --test-out), or pass --train-data with a file that "
            f"contains ONLY training tiles.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    same = os.path.abspath(args.data) == meta.get("data")
    if not same and int(meta.get("data_rows", -1)) == len(n_all):
        same = file_sha256(args.data) == meta.get("data_sha256")
    if not same or "test_indices" not in meta:
        raise SystemExit(
            f"--data {args.data} is not the dataset that generated {args.test} "
            f"(sidecar says {meta.get('data')}, {meta.get('data_rows')} rows). Using it would let "
            f"every test tile find itself at distance 0. Pass the right --data, or --train-data "
            f"with a training-only file.")

    test_idx = np.asarray(meta["test_indices"], dtype=np.int64)
    keep = np.ones(len(n_all), dtype=bool)
    keep[test_idx] = False
    print(f"Training pool: {int(keep.sum())} tiles from --data {args.data} "
          f"({len(test_idx)} held-out test rows excluded).")
    return n_all[keep], obs_all[keep]


def _bin_mask(n_col, lo, hi):
    return (n_col >= lo) & (n_col <= hi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True,
                   help="held-out test split .npz (the --test-out of `set_diffusion train`)")
    p.add_argument("--data", default=None,
                   help="the full dataset that `set_diffusion train --data` saw (train + test). "
                        "The test rows are excluded using the .meta.json sidecar.")
    p.add_argument("--train-data", default=None,
                   help="alternative to --data: a .npz containing ONLY training tiles.")
    p.add_argument("--output", default=None, help="optional path for the JSON report")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="only used to subsample the threshold tiles")
    p.add_argument("--max-train", type=int, default=0,
                   help="cap the training tiles used for the nearest-neighbour index (0 = all; "
                        "subsampling only speeds things up and can hide very close duplicates, so "
                        "keep 0 unless memory forces it)")
    p.add_argument("--threshold-sample", type=int, default=20000,
                   help="training tiles whose leave-one-out distance defines the near-duplicate "
                        "threshold (default 20000, the same order as the memorization test's index). "
                        "The nearest-neighbour lookup for the TEST tiles still uses the FULL index.")
    p.add_argument("--nn-batch-size", type=int, default=1000,
                   help="query rows processed per chunk in the brute-force nearest-neighbour search "
                        "(see src/common/tile_descriptors.py:nearest_distance) -- lower it if this "
                        "runs out of memory with a very large training index; it does not change "
                        "the result, only memory/throughput.")
    p.add_argument("--dedup-out", default=None,
                   help="if given, write a copy of --test with every row flagged as near-duplicate "
                        "(below its bin's threshold, in ANY bin -- not just alerted ones) removed, "
                        "plus a <path>.meta.json sidecar recording which source rows were dropped and "
                        "why (same pattern as set_diffusion.py's test_holdout.npz.meta.json). Use this "
                        "deduplicated file as --reference-data instead of the original holdout.")
    args = p.parse_args()

    if bool(args.data) == bool(args.train_data):
        raise SystemExit("pass exactly one of --data (full dataset + sidecar) or --train-data "
                         "(training-only file).")

    n_test, obs_test, success_test, success_key_test, L_test = _load_test(args.test)
    n_train, obs_train = _load_train_pool(args, n_test)
    if len(n_train) < 2:
        raise SystemExit(f"only {len(n_train)} training tiles -- nothing to compare against.")
    if args.max_train and len(n_train) > args.max_train:
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(n_train), size=args.max_train, replace=False)
        print(f"WARNING: --max-train={args.max_train}: subsampling the training index; very close "
              f"duplicates outside the subsample will not be seen.")
        n_train, obs_train = n_train[sub], obs_train[sub]

    print(f"Test tiles: {len(n_test)} | training tiles: {len(n_train)} | "
          f"descriptor: {DESCRIPTOR_BINS}x{DESCRIPTOR_BINS} occupancy (r^2 weighted)")

    k = DESCRIPTOR_BINS
    H_train = occupancy_hists(obs_train, n_train, k=k)
    H_test = occupancy_hists(obs_test, n_test, k=k)

    # Nearest training neighbour of every test tile -- against the FULL index,
    # so a near-duplicate anywhere in training is seen, whatever bin it falls
    # in (a near-duplicate can differ from the test tile by one obstacle).
    # Brute force, not a KD-tree: see nearest_distance's docstring -- a tree
    # is pathological at this descriptor's dimensionality (k*k = up to 256).
    d_test = nearest_distance(H_test, H_train, batch_size=args.nn_batch_size)

    rng = np.random.default_rng(args.seed)

    def bin_threshold(n_mask):
        """``PERCENTILE`` of the nearest-OTHER-neighbour distance of a
        sample of this bin's TRAINING tiles, measured against the SAME FULL
        training index ``H_train`` that ``d_test`` is measured against -- not
        a bin-restricted sub-index. Querying a smaller, bin-only pool would
        make every threshold systematically larger than ``d_test`` (fewer
        candidates to be close to), an apples-to-oranges comparison that has
        nothing to do with leakage. Each bin still gets its OWN threshold
        (see the module docstring for why one global number is not a fair
        ruler for both sparse and dense tiles), it is just calibrated on the
        same scale ``d_test`` lives on.

        The probes ARE rows of ``H_train`` itself, so their own distance to
        themselves (0) is excluded before taking the minimum -- see
        ``nearest_other_distance``'s docstring."""
        rows_bin = np.nonzero(n_mask)[0]
        if len(rows_bin) < 2:
            return None, 0
        n_thr = min(args.threshold_sample, len(rows_bin))
        thr_rows = rng.choice(rows_bin, size=n_thr, replace=False)
        d_loo = nearest_other_distance(thr_rows, H_train, batch_size=args.nn_batch_size)
        return float(np.percentile(d_loo, PERCENTILE)), n_thr

    def stats(test_mask, train_mask):
        cnt = int(test_mask.sum())
        if cnt == 0:
            return None
        threshold, n_thr = bin_threshold(train_mask)
        if threshold is None:
            return {"test_tiles": cnt, "threshold": None,
                     "note": "fewer than 2 training tiles in this bin -- no threshold"}
        dd = d_test[test_mask]
        below = int((dd <= threshold).sum())
        return {"test_tiles": cnt, "threshold": threshold, "threshold_train_sample": n_thr,
                "below_threshold": below, "fraction": below / cnt,
                "nn_median": float(np.median(dd)), "nn_min": float(dd.min()),
                "exact_descriptor_duplicates": int((dd <= 0.0).sum())}

    # Per-row flag (not just the per-bin aggregate count): True where a test
    # tile's distance is at or below ITS OWN bin's threshold -- the rows
    # --dedup-out removes. Built across ALL bins, not only the ones that trip
    # ALERT_FRAC: the threshold already encodes "suspiciously close for
    # this bin", a clean reference should not contain any row that fails it,
    # alerted bin or not.
    below_mask = np.zeros(len(n_test), dtype=bool)

    rows = []
    for lo, hi in BINS_N:
        test_mask = _bin_mask(n_test, lo, hi)
        s = stats(test_mask, _bin_mask(n_train, lo, hi))
        if s is None:
            rows.append({"bin_n": f"{lo}-{hi}", "test_tiles": 0})
        else:
            rows.append({"bin_n": f"{lo}-{hi}", **s})
            if s.get("threshold") is not None:
                below_mask[test_mask] = d_test[test_mask] <= s["threshold"]
    # anything outside the canonical bins is never silently dropped
    test_mask_other = n_test > BINS_N[-1][1]
    s_other = stats(test_mask_other, n_train > BINS_N[-1][1])
    if s_other is not None:
        rows.append({"bin_n": f">{BINS_N[-1][1]}", **s_other})
        if s_other.get("threshold") is not None:
            below_mask[test_mask_other] = d_test[test_mask_other] <= s_other["threshold"]

    # "ALL" aggregates the per-bin verdicts (each test tile already judged
    # against ITS OWN bin's threshold) -- there is no single global threshold
    # to report any more.
    scored = [r for r in rows if r.get("threshold") is not None]
    overall = {
        "test_tiles": sum(r["test_tiles"] for r in scored),
        "below_threshold": sum(r["below_threshold"] for r in scored),
        "fraction": (sum(r["below_threshold"] for r in scored) /
                     max(1, sum(r["test_tiles"] for r in scored))),
        "nn_median": float(np.median(d_test)), "nn_min": float(d_test.min()),
        "exact_descriptor_duplicates": int((d_test <= 0.0).sum()),
    } if scored else None

    null_frac = PERCENTILE / 100.0
    print(f"\nThreshold: p{PERCENTILE:g} of the training leave-one-out distance, "
          f"computed SEPARATELY PER n-bin (see table) -- not one global number.")
    print(f"On a clean i.i.d. split the expected fraction below it is ~{null_frac:.1%}; "
          f"bins above {ALERT_FRAC:.1%} are flagged.\n")
    header = (f"{'bin n':>8} {'test':>7} {'below':>7} {'fraction':>9} {'threshold':>10} "
              f"{'nn median':>10} {'nn min':>9} {'exact dup':>9}")
    print(header)
    print("-" * len(header))

    alerts = []
    for r in rows:
        if not r.get("test_tiles"):
            print(f"{r['bin_n']:>8} {0:>7} {'-':>7} {'-':>9} {'-':>10} {'-':>10} {'-':>9} {'-':>9}")
            continue
        if r.get("threshold") is None:
            print(f"{r['bin_n']:>8} {r['test_tiles']:>7} {'-':>7} {'-':>9} {'-':>10} "
                  f"{'-':>10} {'-':>9} {'-':>9}  (too few training tiles for a threshold)")
            continue
        hot = r["fraction"] > ALERT_FRAC
        if hot:
            alerts.append(r)
        flag = "  <-- LEAKAGE" if hot else ""
        if r["test_tiles"] < 100:
            flag += "  (low n: treat with caution)"
        print(f"{r['bin_n']:>8} {r['test_tiles']:>7} {r['below_threshold']:>7} "
              f"{r['fraction']:>8.1%} {r['threshold']:>10.5f} {r['nn_median']:>10.5f} "
              f"{r['nn_min']:>9.5f} {r['exact_descriptor_duplicates']:>9}{flag}")

    print("-" * len(header))
    if overall is not None:
        print(f"{'ALL':>8} {overall['test_tiles']:>7} {overall['below_threshold']:>7} "
              f"{overall['fraction']:>8.1%} {'(per bin)':>10} {overall['nn_median']:>10.5f} "
              f"{overall['nn_min']:>9.5f} {overall['exact_descriptor_duplicates']:>9}")

    if alerts:
        bins = ", ".join(r["bin_n"] for r in alerts)
        verdict = (f"LEAKAGE: {len(alerts)} bin(s) above ALERT_FRAC={ALERT_FRAC:.1%} "
                   f"({bins}), each judged against its OWN bin's near-duplicate threshold. "
                   f"Re-draw the test holdout with another partition before trusting ANY C2ST / "
                   f"Chamfer / memorization number that uses it.")
    else:
        verdict = (f"OK: no bin above ALERT_FRAC={ALERT_FRAC:.1%}; the split looks clean at "
                   f"the '{k}x{k} occupancy' descriptor scale.")
    print(f"\n{verdict}")

    flagged_positions = np.nonzero(below_mask)[0]
    report = {
        "test": os.path.abspath(args.test),
        "train_source": os.path.abspath(args.train_data or args.data),
        "n_test": int(len(n_test)), "n_train": int(len(n_train)),
        "descriptor": f"{k}x{k} occupancy (r^2 weighted, normalized)",
        "percentile": PERCENTILE,
        "threshold_note": "computed separately per n-bin -- see bins[*].threshold",
        "null_fraction_expected": null_frac, "alert_frac": ALERT_FRAC,
        "overall": overall, "bins": rows, "leakage_bins": [r["bin_n"] for r in alerts],
        "verdict": verdict,
        "below_threshold_row_positions": flagged_positions.tolist(),
        "below_threshold_note": ("row positions WITHIN --test (0-indexed) flagged in ANY bin, not "
                                 "just alerted ones -- see --dedup-out to remove them."),
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"Saved {args.output}")

    if args.dedup_out:
        keep = ~below_mask
        cols = {"obs": obs_test[keep], "n": n_test[keep].astype(np.int16)}
        if success_key_test is not None:
            cols[success_key_test] = success_test[keep]
        if L_test is not None:
            cols["L"] = L_test[keep]
        os.makedirs(os.path.dirname(os.path.abspath(args.dedup_out)) or ".", exist_ok=True)
        np.savez_compressed(args.dedup_out, **cols)
        dedup_meta = {
            "source_test": os.path.abspath(args.test),
            "source_test_sha256": file_sha256(args.test),
            "source_test_rows": int(len(n_test)),
            "dedup_rows": int(keep.sum()),
            "removed_rows": int(len(n_test) - keep.sum()),
            "removed_row_positions": flagged_positions.tolist(),
            "descriptor": f"{k}x{k} occupancy (r^2 weighted, normalized)",
            "percentile": PERCENTILE,
            "per_bin_thresholds": rows,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": ("Held-out TEST split with near-duplicate rows (per their OWN bin's "
                     "leave-one-out threshold, see check_split_leakage.py) removed. Same column "
                     "layout as --test. Use this file, not the original holdout, as "
                     "--reference-data for src/evaluation/paper_metrics.py when comparing results "
                     "across seeds -- so a near-duplicate tile is not the reason two runs' "
                     "diversity/memorization numbers disagree."),
        }
        with open(args.dedup_out + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(dedup_meta, f, indent=2, default=float)
        print(f"\nDedup holdout written: {args.dedup_out} | {int(keep.sum())} tiles kept, "
              f"{int(len(n_test) - keep.sum())} removed ({(len(n_test) - keep.sum()) / len(n_test):.1%})")


if __name__ == "__main__":
    main()
