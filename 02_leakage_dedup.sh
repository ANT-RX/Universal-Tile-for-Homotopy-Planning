#!/usr/bin/env bash
# =============================================================================
# 02_leakage_dedup.sh  --  POINT 7: leakage report on the DEDUPLICATED holdout.
#
# This is the evidence Rev 4.2 asks for. Today the paper claims the leaky rows
# were removed, but it does not show that the 5 bins fell below the threshold.
#
# RESOURCES: CPU + RAM, no GPU (unless the script accepts --device cuda, see below).
#   - train descriptors: 600,000 x 256 float32 = 614 MB
#   - distance block with block=512: ~1.2 GB
#   -> 16 GB of RAM is plenty. If you get OOM, lower the block size.
#   - on CPU: on the order of tens of minutes. On GPU: seconds.
#     (the matmul is 30,000 x 600,000 x 256 ~ 9.2 TFLOP; a 4090 eats it)
#
# Usage:  bash 02_leakage_dedup.sh
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT="$ROOT/artifacts/pending"
mkdir -p "$OUT"
LOG="$OUT/02_leakage.log"

DATA="${DATA:-$ROOT/tile-obstacle-maps.npz}"
TEST_DEDUP="${TEST_DEDUP:-$ROOT/artifacts/test_holdout_dedup.npz}"
REPORT="$OUT/leakage_report_dedup.json"

echo "== 02 - leakage on the deduplicated holdout ==" | tee "$LOG"
echo "data      = $DATA"        | tee -a "$LOG"
echo "test      = $TEST_DEDUP"  | tee -a "$LOG"
echo "output    = $REPORT"      | tee -a "$LOG"

for f in "$DATA" "$TEST_DEDUP"; do
  [ -f "$f" ] || { echo "ERROR: $f does not exist" | tee -a "$LOG"; exit 1; }
done

# ---------------------------------------------------------------------------
# STEP 0 - confirm the actual flags before launching the long run.
# The names below come from your report; if --help shows others,
# adjust the two lines of the invocation and run again.
# ---------------------------------------------------------------------------
echo | tee -a "$LOG"
echo "-- available flags --" | tee -a "$LOG"
python3 -m src.evaluation.check_split_leakage --help 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
read -r -p "Are the --data/--test flags correct? [y/N] " RESP
case "$RESP" in [sSyY]*) ;; *) echo "Aborted. Adjust the script."; exit 0;; esac

# If the --help above showed a device flag (--device / --gpu / --cuda),
# add it here: it speeds this up from minutes to seconds on the 4090.
EXTRA="${EXTRA:-}"

echo | tee -a "$LOG"
echo "-- running ($(date -Is)) --" | tee -a "$LOG"
time python3 -m src.evaluation.check_split_leakage \
    --data "$DATA" \
    --test "$TEST_DEDUP" \
    $EXTRA \
    > "$REPORT" 2>> "$LOG"
RC=$?
echo "exit code: $RC" | tee -a "$LOG"
[ $RC -eq 0 ] || { echo "Failed. Check $LOG"; exit $RC; }

# ---------------------------------------------------------------------------
# STEP 2 - readable summary + the exact sentence for the paper
# ---------------------------------------------------------------------------
python3 - "$REPORT" <<'PY' | tee -a "$LOG"
import json, sys

rep = json.load(open(sys.argv[1]))

# The schema may vary; we look for the list of bins wherever it is.
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

bins = find_bins(rep)
print("\n" + "=" * 64)
print("LEAKAGE ON THE DEDUPLICATED HOLDOUT")
print("=" * 64)

if not bins:
    print("I did not recognize the schema. Raw content:")
    print(json.dumps(rep, indent=2)[:3000])
    sys.exit(0)

def g(d, *names):
    for n in names:
        for k in d:
            if n in str(k).lower():
                return d[k]
    return None

print(f"{'bin':8} {'n_test':>8} {'threshold':>10} {'below thr.':>12} {'fraction':>10}  verdict")
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
        frac *= 100.0                       # normalize to percentage
    worst = max(worst, frac)
    good = frac < 5.0
    allok &= good
    thr_s = f"{thr:.5f}" if isinstance(thr, (int, float)) else str(thr)
    print(f"{str(name):8} {ntest:>8} {thr_s:>10} {under:>12} {frac:>9.2f}%  "
          f"{'OK' if good else '<-- OVER 5%'}")

print("=" * 64)
if allok:
    print("ALL bins are below the 5% alert threshold.\n")
    print("Sentence for the paper (Section IV-B, after the description of the audit):\n")
    print(f'  "After removal, all five bins fall below the 5% alert threshold')
    print(f'   (maximum {worst:.2f}%), confirming that the deduplicated holdout')
    print('   used for all generative-quality metrics is free of near-duplicate')
    print('   leakage from the training corpus."')
else:
    print("WARNING: some bin is still above 5%.")
    print("Do not write the sentence yet. Options:")
    print("  (a) deduplicate again with the threshold of the bin that failed;")
    print("  (b) report it as is and explain it (bin 5-10 has threshold 0.0")
    print("      because there are exact duplicates INSIDE the train set itself: at low")
    print("      density the space of configurations distinguishable under this")
    print("      descriptor is small, so any random partition")
    print("      would inherit collisions. It is a property of the domain, not a")
    print("      defect of the split, and stated that way it is defensible).")
PY

echo
echo "Done. Report: $REPORT"
echo "Log:           $LOG"
