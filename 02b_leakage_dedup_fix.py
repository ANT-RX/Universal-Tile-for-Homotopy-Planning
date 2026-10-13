#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_leakage_dedup_fix.py  --  Builds the training-only split and runs the leakage audit.

WHY --data CANNOT BE USED HERE
------------------------------
    --data ... is not the dataset that generated test_holdout_dedup.npz
    (sidecar says None, None rows).

The sidecar of test_holdout_dedup.npz records which HOLDOUT ROWS were
dropped, but not the test_indices inside the 600,000 dataset. Without that,
--data cannot know which 30,000 rows to exclude from the training index, and
every test tile would find itself at distance 0 -> fraction below threshold
~100%.

THE CLEAN WAY OUT
-----------------
The --help itself offers the alternative: --train-data with an .npz that
contains ONLY training tiles. This script builds it from the test_indices of
the ORIGINAL holdout (the 30,000 rows that training never saw) and then
launches the audit.

Note: the 30,000 originals are excluded, not the ones that survived
deduplication. That is how training really happened, and it is what makes the
audit honest.

Usage:
    python3 02b_leakage_dedup_fix.py                 # build and run
    python3 02b_leakage_dedup_fix.py --build-only
    python3 02b_leakage_dedup_fix.py (descriptor frozen at 16)
"""
import argparse, glob, hashlib, json, os, subprocess, sys

# ---------------------------------------------------------------------------
# Frozen campaign configuration: the campaign values are constants here.
# The remaining flags are the machine-dependent ones (paths, --root) and
# --nn-batch-size, which is a memory/throughput knob like --workers.
# ---------------------------------------------------------------------------
DESCRIPTOR_BINS = 16   # --descriptor-bins: k of the k x k occupancy descriptor
BUILD_ONLY = False     # --build-only: stop after building train_only.npz
REDO = False           # --redo: rebuild train_only.npz even if it already exists

P = argparse.ArgumentParser()
P.add_argument("--root", default=os.getcwd())
P.add_argument("--data", default=None)
P.add_argument("--test-dedup", default=None)
P.add_argument("--holdout-meta", default=None,
               help="sidecar of the ORIGINAL holdout (the one that carries test_indices)")
P.add_argument("--train-out", default=None)
P.add_argument("--nn-batch-size", type=int, default=0, help="0 = script default")
A = P.parse_args()

ROOT = os.path.abspath(A.root)
OUT = os.path.join(ROOT, "artifacts", "pending")
os.makedirs(OUT, exist_ok=True)

DATA = A.data or os.path.join(ROOT, "tile-obstacle-maps.npz")
DEDUP = A.test_dedup or os.path.join(ROOT, "artifacts", "test_holdout_dedup.npz")
TRAIN_OUT = A.train_out or os.path.join(ROOT, "artifacts", "train_only.npz")
REPORT = os.path.join(OUT, "leakage_report_dedup.json")

def die(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)

print("=" * 72)
print("02b - leakage audit with --train-data")
print("=" * 72)

for f, label in ((DATA, "--data"), (DEDUP, "--test")):
    if not os.path.exists(f):
        die(f"{label} does not exist: {f}")

# ------------------------------------------------- 1. holdout sidecar ---
print("\n--- 1. Locating the sidecar of the ORIGINAL holdout ---")
meta_path = A.holdout_meta
if not meta_path:
    candidates = []
    for pattern in ("artifacts/test_holdout*.meta.json", "artifacts/**/test_holdout*.meta.json"):
        candidates += glob.glob(os.path.join(ROOT, pattern), recursive=True)
    # the _dedup sidecar has no test_indices, so it is not the one we need
    candidates = [c for c in sorted(set(candidates)) if "dedup" not in os.path.basename(c).lower()]
    if not candidates:
        die("I did not find the sidecar of the original holdout.\n"
            "       Pass it with --holdout-meta artifacts/test_holdout_final.npz.meta.json")
    for c in candidates:
        try:
            m = json.load(open(c))
        except Exception:
            continue
        if any(k in m for k in ("test_indices", "indices", "test_idx")):
            meta_path = c
            break
    meta_path = meta_path or candidates[0]
print(f"  {os.path.relpath(meta_path, ROOT)}")

meta = json.load(open(meta_path))
print(f"  keys: {', '.join(sorted(meta))}")

idx = next((meta[k] for k in ("test_indices", "indices", "test_idx") if k in meta), None)
if idx is None:
    die(f"{meta_path} does not carry test_indices. Without them the train split cannot be separated.")
print(f"  test_indices: {len(idx):,}")

# ------------------------------------------------------- 2. data hash ---
print("\n--- 2. Verifying that --data is the correct dataset ---")
declared = next((meta[k] for k in ("data_sha256", "sha256", "dataset_sha256") if k in meta), None)
rows_declared = next((meta[k] for k in ("data_rows", "rows", "n_rows") if k in meta), None)
if declared:
    h = hashlib.sha256()
    with open(DATA, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    got = h.hexdigest()
    print(f"  sidecar : {declared}")
    print(f"  file    : {got}")
    if got != declared:
        die("the hash does NOT match: this is not the dataset that generated the holdout.\n"
            "       Stop here; with another dataset the indices mean nothing.")
    print("  -> matches")
else:
    print("  (the sidecar does not declare a hash; I continue, but verify the row count)")

# -------------------------------------------------- 3. build train-only ---
print("\n--- 3. Building the training-only .npz ---")
try:
    import numpy as np
except ImportError:
    die("numpy is missing")

if os.path.exists(TRAIN_OUT) and not REDO:
    print(f"  already exists: {os.path.relpath(TRAIN_OUT, ROOT)} (use --redo to regenerate it)")
else:
    z = np.load(DATA, allow_pickle=False)
    keys = list(z.files)
    print(f"  arrays in the dataset: {', '.join(keys)}")

    n_rows = max(z[k].shape[0] for k in keys if getattr(z[k], "ndim", 0) >= 1)
    print(f"  total rows: {n_rows:,}")
    if rows_declared and int(rows_declared) != int(n_rows):
        die(f"the sidecar says {rows_declared} rows and the file has {n_rows}.")

    test_idx = np.asarray(idx, dtype=np.int64)
    if test_idx.min() < 0 or test_idx.max() >= n_rows:
        die("the test_indices are out of the dataset range.")

    mask = np.ones(n_rows, dtype=bool)
    mask[test_idx] = False
    n_train = int(mask.sum())
    print(f"  excluding {len(test_idx):,} test rows -> {n_train:,} training rows")
    if n_train != n_rows - len(set(test_idx.tolist())):
        print("  WARNING: there were repeated test_indices")

    payload = {}
    for k in keys:
        arr = z[k]
        if getattr(arr, "ndim", 0) >= 1 and arr.shape[0] == n_rows:
            payload[k] = arr[mask]
            print(f"    {k:20} {str(arr.shape):>18} -> {str(payload[k].shape)}")
        else:
            payload[k] = arr
            print(f"    {k:20} {str(getattr(arr, 'shape', '()')):>18}    (scalar/global, copied as is)")

    tmp = TRAIN_OUT + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, TRAIN_OUT)
    print(f"  written: {os.path.relpath(TRAIN_OUT, ROOT)} "
          f"({os.path.getsize(TRAIN_OUT) / 1e6:.0f} MB)")

    # Leave a record of how it was built.
    with open(TRAIN_OUT + ".meta.json", "w") as f:
        json.dump({
            "built_by": "02b_leakage_dedup_fix.py",
            "source_data": os.path.relpath(DATA, ROOT),
            "source_data_sha256": declared,
            "source_rows": int(n_rows),
            "excluded_test_indices_from": os.path.relpath(meta_path, ROOT),
            "n_excluded": int(len(test_idx)),
            "n_train_rows": int(n_train),
            "note": "training-only split; the 30,000 original holdout rows are removed",
        }, f, indent=2)

if BUILD_ONLY:
    print("\n--build-only: not launching the audit.")
    sys.exit(0)

# ------------------------------------------------------- 4. the audit ---
print("\n--- 4. Leakage audit ---")
print(f"""
  Descriptor
  ----------
  The {DESCRIPTOR_BINS}x{DESCRIPTOR_BINS} r^2-weighted occupancy descriptor is used --
  the same one the original audit and the paper's Section IV-B used. Ignore any
  stale "--help says default 8": the code default is OCCUPANCY_BINS = 16.
""")

cmd = [sys.executable, "-m", "src.evaluation.check_split_leakage",
       "--train-data", TRAIN_OUT,
       "--test", DEDUP,
       "--max-train", "0",
       "--threshold-sample", "20000",
       "--output", REPORT]
if A.nn_batch_size:
    cmd += ["--nn-batch-size", str(A.nn_batch_size)]

print("  " + " \\\n      ".join(cmd) + "\n")
rc = subprocess.call(cmd, cwd=ROOT)
if rc != 0:
    die(f"check_split_leakage exited with code {rc}.\n"
        "       If it was due to memory, retry with --nn-batch-size 256.")

# ------------------------------------------------------------ 5. summary ---
print("\n--- 5. Result ---")
rep = json.load(open(REPORT))

def find_bins(o):
    if isinstance(o, list) and o and isinstance(o[0], dict) and \
       any("frac" in str(k).lower() or "bin" in str(k).lower() for k in o[0]):
        return o
    if isinstance(o, dict):
        for v in o.values():
            r = find_bins(v)
            if r:
                return r
    return None

def g(d, *names):
    for n in names:
        for k in d:
            if n in str(k).lower():
                return d[k]
    return None

bins = find_bins(rep)
if not bins:
    print(json.dumps(rep, indent=2)[:3000])
    sys.exit(0)

print(f"\n{'bin':10} {'n_test':>8} {'threshold':>11} {'below thr.':>12} {'fraction':>10}  verdict")
worst, allok = 0.0, True
for b in bins:
    name = g(b, "bin", "range") or "?"
    ntest = g(b, "n_test", "test", "count") or 0
    thr = g(b, "threshold")
    under = g(b, "under", "below") or 0
    frac = g(b, "frac", "fraction", "rate")
    try:
        frac = float(frac)
    except (TypeError, ValueError):
        frac = (float(under) / float(ntest)) if ntest else 0.0
    if frac <= 1.0:
        frac *= 100.0
    worst = max(worst, frac)
    good = frac < 5.0
    allok &= good
    thr_s = f"{thr:.5f}" if isinstance(thr, (int, float)) else str(thr)
    print(f"{str(name):10} {ntest:>8} {thr_s:>11} {under:>12} {frac:>9.2f}%  "
          f"{'OK' if good else '<-- OVER 5%'}")

print()
if allok:
    print("All bins below the 5% threshold. Sentence for Section IV-B:\n")
    print('  "After removal, all five bins fall below the 5% alert threshold')
    print(f'   (maximum {worst:.2f}%), confirming that the deduplicated holdout used')
    print('   for all generative-quality metrics is free of near-duplicate leakage."')
else:
    print("Some bin is still above 5%. Do NOT write the sentence yet.")
    print("Look at bin 5-10 before getting alarmed: if its threshold is 0.0 it is because there are")
    print("exact duplicate descriptors INSIDE the training set itself. At low")
    print("density the space of configurations distinguishable under this descriptor")
    print("is small, so any random partition would inherit collisions.")
    print("That is a property of the domain, not a defect of the split.")

print(f"\nReport: {os.path.relpath(REPORT, ROOT)}")
