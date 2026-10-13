#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_generation_rate_by_n.py  --  Replaces the "97%+" of the Introduction with a measured rate.

The "97%+" is a placeholder: it does not come from any measured result.
The same goes for "16.8% for n in [46,50]", which comes from the ORIGINAL
campaign (460,545 dataset), not from the balanced corpus.

This script has two modes:

  --mode recover   (default, FREE)
      It searches the metadata and logs the balancing campaign already
      produced for the acceptance rate by n. If it is there, nothing to measure.

  --mode plan
      If it is not there, it computes how many candidates you need per bin
      for the precision you request and prints the exact measurement commands.

RESOURCES: CPU, no GPU. It is the HPPM engine in C.

Usage:
    python3 03_generation_rate_by_n.py
    python3 03_generation_rate_by_n.py --mode plan --precision 1.0
"""
import argparse, glob, json, math, os, re

# ---------------------------------------------------------------------------
# Frozen campaign configuration: the campaign values are constants here.
# --root (a path) and --workers (cores, machine-dependent) stay as flags.
#
# NOTE: MODE is frozen to "recover", so the "plan" branch below is unreachable.
# ---------------------------------------------------------------------------
MODE = "recover"   # --mode: "recover" (search existing results) or "plan"
PRECISION = 1.0    # --precision: target half-width of the 95% CI, in points

P = argparse.ArgumentParser()
P.add_argument("--root", default=os.getcwd())
P.add_argument("--workers", type=int, default=30)
A = P.parse_args()

ROOT = os.path.abspath(A.root)
OUT = os.path.join(ROOT, "artifacts", "pending")
os.makedirs(OUT, exist_ok=True)

BINS = [("5-10", 5, 10), ("11-20", 11, 20), ("21-30", 21, 30),
        ("31-40", 31, 40), ("41-50", 41, 50)]

lines = []
def say(s=""):
    print(s)
    lines.append(s)

say("=" * 72)
say("03 - generation acceptance rate (local repair) by n")
say("=" * 72)

# ======================================================= RECOVER MODE =====
if MODE == "recover":
    say("\n--- Searching for the rate by n in already existing artifacts ---")

    PAT = re.compile(r"(acceptance_rate|acceptance|accept|rate)", re.I)
    hits = []

    candidates = []
    for pattern in ("artifacts/**/*.json", "*.json", "results*/**/*.json",
                    "**/gen*meta*.json", "**/*generation*.json"):
        candidates += glob.glob(os.path.join(ROOT, pattern), recursive=True)
    candidates = [c for c in sorted(set(candidates)) if "pending" not in c][:400]

    def walk(o, path, src):
        if isinstance(o, dict):
            for k, v in o.items():
                if PAT.search(str(k)) and not isinstance(v, (dict, list)):
                    hits.append((src, f"{path}/{k}", v))
                walk(v, f"{path}/{k}", src)
        elif isinstance(o, list) and len(o) < 80:
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]", src)

    scanned = 0
    for c in candidates:
        try:
            if os.path.getsize(c) > 60_000_000:
                continue
            with open(c) as f:
                walk(json.load(f), "", os.path.relpath(c, ROOT))
            scanned += 1
        except Exception:
            pass
    say(f"  ({scanned} JSON files read)")

    if hits:
        say(f"\n  {len(hits)} matches (first 30):")
        for src, path, v in hits[:30]:
            say(f"    {src}{path} = {v}")
        say("\n  If any of them is broken down BY BIN OF n, that is the number:")
        say("  there is nothing to measure, just copy it into the paper.")
    else:
        say("  No matches in the JSONs.")

    # The generation logs usually carry it even if the JSON does not.
    say("\n--- Searching the generation logs ---")
    logs = []
    for pattern in ("*.log", "logs/*.log", "artifacts/**/*.log", "**/gen*.log"):
        logs += glob.glob(os.path.join(ROOT, pattern), recursive=True)
    logs = sorted(set(logs))[:200]
    found_log = 0
    LINE = re.compile(r"(accept|rate|bin|n\s*=|n_obs)", re.I)
    for lg in logs:
        try:
            with open(lg, encoding="utf-8", errors="ignore") as f:
                tail = f.readlines()[-400:]
            sel = [l.strip() for l in tail if LINE.search(l) and "%" in l]
            if sel:
                say(f"\n  {os.path.relpath(lg, ROOT)}:")
                for l in sel[:12]:
                    say(f"    {l[:140]}")
                found_log += 1
        except Exception:
            pass
    if not found_log:
        say("  Nothing in the logs.")

    say("\n" + "-" * 72)
    say("If the rate broken down by bin of n did NOT appear above, run:")
    say("    python3 03_generation_rate_by_n.py --mode plan")
    say("-" * 72)

# ============================================================ PLAN MODE =====
else:
    say(f"\n--- Measurement plan (95% CI of +/- {PRECISION} pp) ---")

    # n = z^2 p(1-p) / e^2 ; we use the expected rates as a reference.
    z, e = 1.96, PRECISION / 100.0
    expected = {"5-10": 0.97, "11-20": 0.80, "21-30": 0.55,
                "31-40": 0.30, "41-50": 0.168}

    say(f"\n  {'bin':8} {'expected rate':>14} {'min candidates':>16} {'suggested':>10}")
    plan = {}
    for lab, lo, hi in BINS:
        p = expected[lab]
        n = math.ceil(z * z * p * (1 - p) / (e * e))
        suggested = max(6000, int(math.ceil(n / 1000.0) * 1000))
        plan[lab] = suggested
        say(f"  {lab:8} {p*100:>13.1f}% {n:>16,} {suggested:>10,}")

    say(f"\n  Total: {sum(plan.values()):,} candidates in the 5 bins.")
    say("  The 'suggested' column rounds up with a floor of 6,000, which is")
    say("  the per-bin size the campaign already used. Only the two numbers that go")
    say("  into the paper (bins 5-10 and 41-50) need this precision; if you are short of")
    say("  time, measure those two and skip the three in between.")

    say("""
  ------------------------------------------------------------------
  IMPORTANT WARNING
  ------------------------------------------------------------------
  With small --n, BATCH STARVATION was detected: the workers are left
  without work and the timing measurement comes out invalid. Do not measure
  the rate with batches of hundreds of candidates. A single large batch per bin.

  And RECOMPILE the HPPM engine on this machine before measuring: a stale
  binary produced 0 successes out of millions of candidates.

      gcc -O2 -o engine/hppm engine/HPPMDinamico3.c -lm
      chmod +x engine/hppm
  ------------------------------------------------------------------
""")

    say("  Commands (adjust the flags to the actual --help of fast_generate.py):\n")
    say("  #!/usr/bin/env bash")
    say("  set -uo pipefail")
    say("  mkdir -p artifacts/pending/genrate")
    for lab, lo, hi in BINS:
        say(f"""
  # --- bin {lab} ---
  python3 src/dataset/fast_generate.py \\
      --n {plan[lab]} \\
      --n-min {lo} --n-max {hi} \\
      --workers {A.workers} \\
      --repair \\
      --seed 42 \\
      --out artifacts/pending/genrate/bin_{lab.replace('-', '_')}.json \\
      2>&1 | tee artifacts/pending/genrate/bin_{lab.replace('-', '_')}.log""")

    say("""
  Afterwards, to aggregate the results:
      python3 03_generation_rate_by_n.py --mode recover

  ------------------------------------------------------------------
  APPROXIMATE COST
  ------------------------------------------------------------------
  From Table III: local repair = 209.6 HPPM runs per useful tile,
  and pure rejection spends ~0.0053 worker-s per HPPM run. That puts the
  sparse bin on the order of a few minutes with 30 workers, and the dense bin
  quite a bit more because each candidate consumes more repair attempts.
  It is an order-of-magnitude estimate, not a measurement:
  start with bin 41-50 to calibrate the real time
  before launching all five.
""")

say("\n" + "=" * 72)
say("WHAT TO DO WITH THE RESULT")
say("=" * 72)
say("""
In the Introduction, replace the current sentence:

  "The local-repair success rate is highly sensitive to density, reaching
   97%+ in sparse environments (n in [5,10], the lowest density the
   pipeline admits) but dropping to 16.8% in highly cluttered spaces with
   46 to 50 obstacles."

with the two measured numbers, declaring the corpus:

  "The local-repair success rate is highly sensitive to density, reaching
   <X>% for n in [5,10] -- the lowest density the pipeline admits -- and
   dropping to <Y>% for n in [41,50], measured over the balanced generation
   campaign."

Two warnings:
  1. Use [41,50], not [46,50]: they are the bins of the balanced campaign.
     Mixing a range from one campaign with tables from another is exactly
     the kind of inconsistency to avoid.
  2. If you decide NOT to measure, delete the two numbers and leave only the
     trend ("drops sharply with obstacle density"). A figure without support is
     worse than none: it is the only number in the paper that does not
     come from the balanced corpus.
""")

with open(os.path.join(OUT, "03_rate_by_n.txt"), "w") as f:
    f.write("\n".join(lines))
print(f"\n[saved] {os.path.join(OUT, '03_rate_by_n.txt')}")
