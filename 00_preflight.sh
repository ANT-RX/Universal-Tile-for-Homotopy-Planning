#!/usr/bin/env bash
# =============================================================================
# 00_preflight.sh  --  Checks the environment BEFORE spending machine time.
#
# It computes nothing. It only checks that the files are there, that the scripts
# accept the flags we are going to use, and leaves an inventory in artifacts/.
#
# Usage:   bash 00_preflight.sh   (from artifacts/ or from the project root)
# Output:  artifacts/pending/00_preflight.log
# =============================================================================
set -uo pipefail   # no -e: we want to see ALL failures, not abort on the first one

# --- Adaptive ROOT detection ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ROOT:-}" ]; then
  if [ -d "$SCRIPT_DIR/src" ]; then
    ROOT="$SCRIPT_DIR"
  elif [ -d "$SCRIPT_DIR/../src" ]; then
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  elif [ -d "$(pwd)/src" ]; then
    ROOT="$(pwd)"
  elif [ -d "$(pwd)/../src" ]; then
    ROOT="$(cd "$(pwd)/.." && pwd)"
  else
    ROOT="$(pwd)"
  fi
fi

# We move to the project root and add ROOT to PYTHONPATH
# so that both 'python3 -m src...' and relative paths always work.
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT="$ROOT/artifacts/pending"
mkdir -p "$OUT"
LOG="$OUT/00_preflight.log"
exec > >(tee "$LOG") 2>&1

ok(){   printf '  [ OK ]  %s\n' "$*"; }
bad(){  printf '  [MISSING] %s\n' "$*"; }
head_(){ printf '\n== %s ==\n' "$*"; }

echo "Preflight - $(date -Is)"
echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "ROOT       = $ROOT"
echo "PYTHONPATH = $PYTHONPATH"

head_ "1. Hardware"
nproc 2>/dev/null | sed 's/^/  cores: /'
free -g 2>/dev/null | awk '/^Mem:/{print "  RAM total: "$2" GB  |  available: "$7" GB"}'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
else
  echo "  GPU: not detected (no problem: the checks below do not need it)"
fi
df -h "$ROOT" 2>/dev/null | awk 'NR==2{print "  free disk in ROOT: "$4}'

head_ "2. Input data"
for f in tile-obstacle-maps.npz \
         artifacts/test_holdout_dedup.npz; do
  if [ -f "$ROOT/$f" ]; then
    ok "$f  ($(du -h "$ROOT/$f" | cut -f1))"
  else
    bad "$f  <-- not in $ROOT"
  fi
done

# The original holdout (pre-deduplication) is optional but useful for comparison.
for f in artifacts/test_holdout.npz artifacts/test_holdout.npz.meta.json; do
  [ -f "$ROOT/$f" ] && ok "$f (optional)" || echo "  [ -- ]  $f (optional, not critical)"
done

head_ "3. Dataset hash"
# The sidecar aborts if the hash does not match. Better to know it now.
if [ -f "$ROOT/tile-obstacle-maps.npz" ]; then
  echo -n "  sha256 computed: "
  sha256sum "$ROOT/tile-obstacle-maps.npz" | cut -d' ' -f1
  echo "  sha256 expected: 95a20e222a46057d1bfc3cb1d55c359fa1bca9d9ac064b8302c4bd2ceabdfcfe"
  echo "  (if they do not match, STOP: this is not the dataset of the 6-seed campaign)"
fi

head_ "4. Checkpoints of the 6 seeds"
for S in 101 102 103 104 105 106; do
  CK="$ROOT/artifacts/ckpt_tiles_seed$S/diffusion_model.pt"
  [ -f "$CK" ] && ok "seed $S" || bad "seed $S  ($CK)"
done

head_ "5. Training logs"
for S in 101 102 103 104 105 106; do
  for CAND in "$ROOT/train_seed$S.log" "$ROOT/logs/train_seed$S.log" "$ROOT/artifacts/train_seed$S.log"; do
    [ -f "$CAND" ] && { ok "seed $S -> $CAND"; break; }
  done
done

head_ "6. Scripts and their actual flags"
# The flag names below are not verified against the code; this prints the actual --help.
for M in src.evaluation.check_split_leakage; do
  echo "--- python3 -m $M --help"
  python3 -m "$M" --help 2>&1 | sed 's/^/    /' | head -40
done

for F in src/dataset/fast_generate.py src/dataset/verify_maps.py; do
  if [ -f "$ROOT/$F" ]; then
    echo "--- python3 $F --help"
    python3 "$ROOT/$F" --help 2>&1 | sed 's/^/    /' | head -40
  else
    bad "$F not found at $ROOT/$F (adjust the path in 03_generation_rate_by_n.py)"
  fi
done

head_ "7. HPPM engine binary"
# Known incident from your campaign: a stale binary gave 0 successes out of
# millions of candidates. ALWAYS recompile on the new machine.
for B in engine/hppm engine/hppm_v2 engine/HPPMDinamico3 bin/hppm; do
  [ -f "$ROOT/$B" ] && { ok "$B"; ls -la "$ROOT/$B" | sed 's/^/        /'; }
done
echo "  >>> RECOMPILE the engine on this machine before measuring:"
echo "      gcc -O2 -o engine/hppm engine/HPPMDinamico3.c -lm"
echo "      and confirm execution permissions (chmod +x)."

head_ "8. Python dependencies"
python3 - <<'PY'
import importlib, sys
print("  python:", sys.version.split()[0])
for m in ["numpy","scipy","torch","matplotlib","sklearn"]:
    try:
        mod = importlib.import_module(m)
        print(f"  [ OK ]  {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  [MISSING] {m}  ({type(e).__name__})")
try:
    import torch
    print("  cuda available:", torch.cuda.is_available())
except Exception:
    pass
PY
echo "  (install if missing: pip install --break-system-packages scipy matplotlib)"

echo
echo "Preflight finished. Log at: $LOG"
echo "Check the [MISSING] entries before continuing."
