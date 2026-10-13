#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_cfg_y_sampling.py  --  Resolves the DDIM steps and eta, the actual
architecture and the histogram by n.

It does NOT need a GPU: checkpoints are loaded with map_location='cpu'.
It takes seconds. Run it first, it is free.

Usage:
    python3 01_cfg_y_sampling.py
    python3 01_cfg_y_sampling.py --root /path/to/repo

Output:
    artifacts/pending/01_cfg.json          <- cfg of the 6 seeds
    artifacts/pending/01_summary.txt       <- what goes into the paper
"""
import argparse, glob, json, os, re, sys

P = argparse.ArgumentParser()
P.add_argument("--root", default=os.getcwd())
P.add_argument("--seeds", default="101,102,103,104,105,106")
A = P.parse_args()

ROOT = os.path.abspath(A.root)
OUT = os.path.join(ROOT, "artifacts", "pending")
os.makedirs(OUT, exist_ok=True)
SEEDS = [s.strip() for s in A.seeds.split(",") if s.strip()]

lines = []
def say(s=""):
    print(s)
    lines.append(s)

say("=" * 72)
say("01 - cfg of the checkpoints and sampling configuration")
say("=" * 72)

# ---------------------------------------------------------------- 1. cfg ----
try:
    import torch
except ImportError:
    say("ERROR: torch is missing. pip install --break-system-packages torch")
    sys.exit(1)

cfgs, hists = {}, {}
say("\n--- 1. actual cfg of each checkpoint ---")
for S in SEEDS:
    ck_path = os.path.join(ROOT, "artifacts", f"ckpt_tiles_seed{S}", "diffusion_model.pt")
    if not os.path.exists(ck_path):
        say(f"  seed {S}: NOT FOUND -> {ck_path}")
        continue
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as e:
        say(f"  seed {S}: error while loading ({type(e).__name__}: {e})")
        continue

    cfg = dict(ck.get("config", ck.get("cfg", {})))
    hists[S] = cfg.pop("hist_n_success", None)          # kept separate
    cfgs[S] = cfg

    # REAL parameter count, which is what validates the "9.8 M" in the paper.
    n_par = None
    for k in ("model", "model_state_dict", "state_dict", "ema", "ema_state_dict"):
        sd = ck.get(k)
        if isinstance(sd, dict) and sd:
            try:
                n_par = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
                break
            except Exception:
                pass
    say(f"  seed {S}: step={ck.get('step', ck.get('paso', '?'))}  "
        f"params={'%.2f M' % (n_par / 1e6) if n_par else '?'}")

if cfgs:
    with open(os.path.join(OUT, "01_cfg.json"), "w") as f:
        json.dump(cfgs, f, indent=2, ensure_ascii=False, default=str)

    # The 6 runs must be identical except for the seed.
    ref_seed = SEEDS[0] if SEEDS[0] in cfgs else list(cfgs)[0]
    ref = cfgs[ref_seed]
    say(f"\n  Comparison against seed {ref_seed} (they must differ ONLY in the seed):")
    diffs = 0
    for S, c in cfgs.items():
        if S == ref_seed:
            continue
        d = {k: (ref.get(k), c.get(k)) for k in set(ref) | set(c)
             if ref.get(k) != c.get(k) and k not in ("seed", "semilla", "--seed")}
        if d:
            diffs += 1
            say(f"    seed {S}: DIFFERS in {d}")
    say("    -> all identical except for the seed" if diffs == 0
        else f"    -> WARNING: {diffs} runs differ in something other than the seed")

    say(f"\n  Values the paper claims in Table I (confirm them here):")
    for key in ("d_model", "layers", "capas", "heads", "cabezas", "T",
                "n_max", "success_only", "solo_exito", "steps", "pasos",
                "batch", "lr", "warmup", "ema", "cfg_dropout"):
        if key in ref:
            say(f"    {key:16} = {ref[key]}")

    say("\n  Field 'step': all 6 must say 100000 "
        "(seed 106 printed 'step None' in the campaign).")

# ------------------------------------------------- 2. DDIM steps and eta ---
say("\n--- 2. DDIM steps and eta of the main results ---")
say("  Searching the evaluation artifacts for the actual keys used...")

KEYS = ("ddim", "eta", "guidance", "steps", "pasos", "seed", "semilla",
        "n_per_bin", "workers", "ports", "margin", "cluster")
found = {}

# We sweep the evaluation JSONs without assuming an exact path.
candidates = []
for pattern in ("artifacts/**/*.json", "results*/**/*.json", "*.json"):
    candidates += glob.glob(os.path.join(ROOT, pattern), recursive=True)
candidates = [c for c in sorted(set(candidates)) if "pending" not in c][:400]

def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            if any(t in kl for t in KEYS) and not isinstance(v, (dict, list)):
                found.setdefault(str(k), set()).add((str(v), path or "/"))
            walk(v, f"{path}/{k}")
    elif isinstance(o, list) and len(o) < 50:
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]")

scanned = 0
for c in candidates:
    try:
        if os.path.getsize(c) > 60_000_000:
            continue
        with open(c) as f:
            walk(json.load(f), os.path.relpath(c, ROOT))
        scanned += 1
    except Exception:
        pass

say(f"  ({scanned} JSON files read)")
if found:
    for k in sorted(found):
        values = sorted(found[k])[:6]
        say(f"    {k:24} = {', '.join(v for v, _ in values)}")
        say(f"    {'':24}   seen in: {values[0][1]}")
else:
    say("    No key was found. The evaluation artifacts do not")
    say("    record the sampling configuration -> it has to come from the code.")

# eta is almost never persisted; it is usually a code default.
say("\n  Default value of eta in the source code:")
eta_pattern = re.compile(r"(eta\s*[:=]\s*[0-9.]+|--eta|default\s*=\s*[0-9.]+\s*[,)].{0,40}eta)", re.I)
hits = 0
for py in glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True):
    try:
        with open(py, encoding="utf-8", errors="ignore") as f:
            for i, ln in enumerate(f, 1):
                if "eta" in ln.lower() and eta_pattern.search(ln):
                    say(f"    {os.path.relpath(py, ROOT)}:{i}: {ln.strip()[:100]}")
                    hits += 1
                    if hits > 15:
                        break
    except Exception:
        pass
    if hits > 15:
        break
if hits == 0:
    say("    'eta' does not appear as a parameter. If the DDIM sampler does not expose it,")
    say("    it is deterministic DDIM: in the paper write 'eta = 0 (deterministic DDIM)'.")

# --------------------------------------------------- 3. histogram by n ---
say("\n--- 3. Histogram by n of the dataset ---")
h = next((v for v in hists.values() if v), None)
if h is None:
    say("  The cfg does not carry 'hist_n_success'.")
else:
    try:
        items = sorted(((int(k), int(v)) for k, v in h.items()), key=lambda x: x[0]) \
                if isinstance(h, dict) else list(enumerate(h))
        tot = sum(v for _, v in items)
        say(f"  total = {tot:,} tiles")
        bins = {"5-10": 0, "11-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "other": 0}
        for n, v in items:
            for lab, (lo, hi) in {"5-10": (5, 10), "11-20": (11, 20), "21-30": (21, 30),
                                  "31-40": (31, 40), "41-50": (41, 50)}.items():
                if lo <= n <= hi:
                    bins[lab] += v
                    break
            else:
                bins["other"] += v
        for lab, v in bins.items():
            say(f"    {lab:7} {v:>9,}" + ("   <-- should be 120,000 (12,000 x 10 n values)"
                                          if lab == "5-10" else ""))
        say("  (useful to document that the balancing to 12,000/bin was met)")
    except Exception as e:
        say(f"  Could not summarize: {e}")

# ---------------------------------------------------------------- closing ----
say("\n" + "=" * 72)
say("WHAT TO DO WITH THIS")
say("=" * 72)
say("""
In the paper, Section III-D and Table I:

  "Tiles are sampled via DDIM with 100 steps and eta = <VALUE>, the
   configuration used for all reported results unless stated otherwise."

  And add a row to Table I:
      DDIM steps / eta      100 / <VALUE>

  If eta did not show up above anywhere, it is deterministic DDIM: set eta = 0.
  Do not leave it undeclared.

If 'params' above does not come out at ~9.8 M, fix the number in the paper.
  (Computed from the declared architecture, 9.44 M + embeddings ~ 9.8 M,
   so it should match.)
""")

with open(os.path.join(OUT, "01_summary.txt"), "w") as f:
    f.write("\n".join(lines))
print(f"\n[saved] {os.path.join(OUT, '01_summary.txt')}")
