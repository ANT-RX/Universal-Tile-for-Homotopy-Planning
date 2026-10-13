#!/usr/bin/env python3
"""Download the dataset and the checkpoints this repository needs.

The repository ships code only: the dataset is hundreds of MB and each
checkpoint is ~40-150 MB, so they live outside version control. This script
fetches them into the paths the code expects and verifies every file against
a pinned sha256 before using it.

Usage:
    python3 scripts/download_assets.py              # dataset (checkpoints too, if hosted)
    python3 scripts/download_assets.py --dataset-only
    python3 scripts/download_assets.py --checkpoints-only
    python3 scripts/download_assets.py --check        # verify what is already there, download nothing

Downloads are resumable (HTTP Range) and idempotent: a file whose sha256
already matches is never downloaded again.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Hosting
# ---------------------------------------------------------------------------
# The dataset lives on Hugging Face. The FILE is renamed to
# tile-obstacle-maps.npz, so all references in the code and docs use that name.
BASE_URL = "https://huggingface.co/datasets/jantrom/tile-obstacle-maps"

# TODO(hosting): the six checkpoints are NOT uploaded yet. Set this to the
# base URL that holds them (a Hugging Face *model* repo, a Zenodo record, ...)
# and the entries in CHECKPOINTS below will start working. Until then,
# --checkpoints fails with an explanation instead of a traceback.
CHECKPOINT_BASE_URL = "https://huggingface.co/jantrom/tile-obstacle-diffusion/resolve/main"

# ---------------------------------------------------------------------------
# Pinned hashes
# ---------------------------------------------------------------------------
# This is the sha256 reported in Table I of the paper. It is the one number in
# this file that must never change: the dataset content is frozen.
DATASET_SHA256 = "95a20e222a46057d1bfc3cb1d55c359fa1bca9d9ac064b8302c4bd2ceabdfcfe"

DATASET = {
    "path": "tile-obstacle-maps.npz",
    "url": f"{BASE_URL}/resolve/main/tile-obstacle-maps.npz",
    "sha256": DATASET_SHA256,
}

# sha256 of each checkpoint, filled in from checksums.txt. The seeds are the
# six training runs of the paper (101-106); seed 101 is the one the full
# evaluation battery was run with.
CHECKPOINT_SEEDS = (101, 102, 103, 104, 105, 106)
CHECKPOINT_FILES = ("diffusion_model.pt", "modelo_final_ema.pt", "ultimo.pt")

CHUNK = 1 << 20  # 1 MiB


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_checksums(path):
    """Read ``sha256  relative/path`` lines. Missing file is not an error: the
    dataset hash is pinned in this script."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[1].strip()] = parts[0].strip()
    return out


def _fail(msg):
    """Exit with an explanation, never a traceback."""
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(2)


def download(url, dest, expected_sha256, label):
    """Resumable download + mandatory sha256 check. Returns True if it fetched
    something, False if the file was already correct."""
    if expected_sha256 is None:
        _fail(f"no sha256 is pinned for {label}. Refusing to download a file "
              f"that cannot be verified.")

    if os.path.exists(dest):
        print(f"  {label}: already present, checking sha256 ...", end="", flush=True)
        actual = sha256_of(dest)
        if actual == expected_sha256:
            print(" OK (skipping download)")
            return False
        print(f" MISMATCH\n  expected {expected_sha256}\n  actual   {actual}")
        _fail(f"{dest} exists but is not the expected file. Delete it and "
              f"re-run, or move it aside if you want to keep it.")

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    resume_from = os.path.getsize(part) if os.path.exists(part) else 0
    if resume_from:
        print(f"  {label}: resuming at {_human(resume_from)}")

    req = urllib.request.Request(url, headers={"User-Agent": "download_assets.py"})
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            _fail(
                f"{url} is not available (HTTP {e.code}).\n"
                f"  The asset is probably not uploaded yet. Set BASE_URL (and,\n"
                f"  for checkpoints, CHECKPOINT_BASE_URL) at the top of this\n"
                f"  script to the location that actually serves it."
            )
        _fail(f"{url} failed with HTTP {e.code} {e.reason}.")
    except urllib.error.URLError as e:
        _fail(f"could not reach {url}: {e.reason}. Check your network/proxy.")

    total = None
    if resp.headers.get("Content-Length"):
        total = int(resp.headers["Content-Length"]) + (resume_from if resp.status == 206 else 0)

    mode = "ab" if (resume_from and resp.status == 206) else "wb"
    if mode == "wb":
        resume_from = 0
    done = resume_from
    started = time.time()
    last = 0.0
    try:
        with open(part, mode) as f:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                f.write(block)
                done += len(block)
                now = time.time()
                if now - last >= 0.5:
                    last = now
                    if total:
                        pct = 100.0 * done / total
                        speed = done - resume_from
                        elapsed = max(now - started, 1e-6)
                        print(f"\r  {label}: {pct:5.1f}%  {_human(done)} / {_human(total)}"
                              f"  ({_human(speed / elapsed)}/s)   ", end="", flush=True)
                    else:
                        print(f"\r  {label}: {_human(done)}   ", end="", flush=True)
    finally:
        resp.close()
    print()

    print(f"  {label}: verifying sha256 ...", end="", flush=True)
    actual = sha256_of(part)
    if actual != expected_sha256:
        print(" MISMATCH")
        os.remove(part)
        _fail(f"{label} was downloaded but its sha256 does not match:\n"
              f"  expected {expected_sha256}\n  actual   {actual}\n"
              f"The partial file was deleted. Re-run to try again.")
    print(" OK")
    os.replace(part, dest)
    return True


def checkpoint_assets(checksums):
    out = []
    for seed in CHECKPOINT_SEEDS:
        for fname in CHECKPOINT_FILES:
            rel = f"artifacts/ckpt_tiles_seed{seed}/{fname}"
            key = f"{BASE_URL}/resolve/main/{rel}" if CHECKPOINT_BASE_URL is None \
                else f"{CHECKPOINT_BASE_URL.rstrip('/')}/{rel}"
            out.append({
                "path": rel,
                "url": None if CHECKPOINT_BASE_URL is None else key,
                "sha256": checksums.get(rel),
                "label": f"seed{seed}/{fname}",
            })
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-only", action="store_true", help="fetch only the dataset")
    p.add_argument("--checkpoints-only", action="store_true", help="fetch only the checkpoints")
    p.add_argument("--check", action="store_true",
                   help="verify what is already on disk and download nothing")
    p.add_argument("--checksums", default=os.path.join(root, "checksums.txt"),
                   help="sha256 manifest (default: checksums.txt at the repo root)")
    args = p.parse_args()

    checksums = load_checksums(args.checksums)
    if not checksums:
        print(f"note: {args.checksums} not found or empty; only the dataset hash "
              f"(pinned in this script) will be enforced.")

    want_dataset = not args.checkpoints_only
    want_ckpt = not args.dataset_only

    failures = []

    if want_dataset:
        print(f"Dataset -> {DATASET['path']}")
        dest = os.path.join(root, DATASET["path"])
        if args.check:
            if os.path.exists(dest):
                ok = sha256_of(dest) == DATASET["sha256"]
                print(f"  {'OK' if ok else 'MISMATCH'}  {dest}")
                if not ok:
                    failures.append(DATASET["path"])
            else:
                print(f"  MISSING  {dest}")
                failures.append(DATASET["path"])
        else:
            try:
                download(DATASET["url"], dest, DATASET["sha256"], DATASET["path"])
            except SystemExit:
                raise

    if want_ckpt:
        print("\nCheckpoints")
        if CHECKPOINT_BASE_URL is None:
            print(
                "  NOT AVAILABLE YET: the six checkpoints have not been uploaded.\n"
                "  This is a known TODO, not a failure of your checkout.\n"
                "  Upload them (Hugging Face model repo or a Zenodo record) and set\n"
                "  CHECKPOINT_BASE_URL at the top of scripts/download_assets.py.\n"
                "  The full evaluation battery in the paper was run with seed 101;\n"
                "  fetching that one alone is enough to reproduce Table IV."
            )
            if args.checkpoints_only and not args.check:
                _fail("--checkpoints-only was requested but the checkpoints are not hosted yet.")
        else:
            for asset in checkpoint_assets(checksums):
                if args.check:
                    dest = os.path.join(root, asset["path"])
                    if not os.path.exists(dest):
                        print(f"  MISSING   {asset['path']}")
                        failures.append(asset["path"])
                    elif asset["sha256"] and sha256_of(dest) != asset["sha256"]:
                        print(f"  MISMATCH  {asset['path']}")
                        failures.append(asset["path"])
                    else:
                        print(f"  OK        {asset['path']}")
                else:
                    download(asset["url"], os.path.join(root, asset["path"]),
                             asset["sha256"], asset["label"])

    if failures:
        print(f"\n{len(failures)} asset(s) missing or corrupt.")
        return 1
    print("\nAll requested assets are present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
