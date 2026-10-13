"""
DIRECT diffusion over the SET of obstacles (x, y, r) of a UNIVERSAL TILE,
without an autoencoder and WITHOUT path conditioning. A universal tile is a
small tile (n in [1,50], SIDE=1.0 FIXED, does not grow) verified to solve the
4*P port->center routes simultaneously (P=ports per side, see
src/dataset/universal_generator.py); tiles are later assembled into mosaics of
any size (src/pipeline/assembler.py, not part of this file).

Model: transformer WITHOUT positional encoding (equivariant to permutations),
adaLN-Zero conditioning (DiT style) on the diffusion step t, the number of
obstacles n (masked in attention and loss) and the class (0=not universal,
1=universal, 2=null for classifier-free guidance). No decoder: the control
with real latents showed the autoencoder was the bottleneck (C2ST=1.0), so
here the model removes noise directly from the coordinates.

SIDE is FIXED at 1.0. Even so normalize()/denormalize() are left L-aware
(instead of assuming 1.0 by hand) in case tiles of another SIDE are ever
mixed -- it is mathematically a no-op with constant L=1.0, it costs nothing.

NOTE -- config_map.py is deliberately NOT used here: its constants (SIDE=2.3,
N_MAX=500) do NOT reflect the tile dataset. src/dataset/universal_generator.py
does not import it either -- it imports from src/common/domain_scaling.py.
Until those two files are reconciled, this module also uses
src/common/domain_scaling.py as the source of truth, just like
src/dataset/universal_generator.py.

Expected format of datos_tiles.npz (see src/dataset/prepare_tiles.py):
    obs    float32 (N, n_max, 3)   x, y, r in [0,SIDE]x[0,SIDE], SIDE=1.0
    n      int16   (N,)
    success bool   (N,)            always True (src/dataset/universal_generator.py
                                   only saves tiles that are already universal)
    L      float32 (N,)            always 1.0

Usage:
  # 1) data (see src/dataset/prepare_tiles.py)
  # 2) train
  python3 src/model/set_diffusion.py train --data artifacts/datos_tiles.npz --dir artifacts/ckpt_tiles --steps 100000
  # 3) sample + verify universality with HPPM (4*P port->center routes)
  python3 src/model/set_diffusion.py sample --dir artifacts/ckpt_tiles --num-obs 25 --n 2000 --guidance 1 \
      --ports-per-side 3 --port-margin 0.04 --hppm-exe artifacts/hppm_v2 \
      --output-dir tiles_n25_g1_s42 --seed 42

COMPATIBILITY NOTE: checkpoints from the previous version (with the path
embedding) do NOT load here -- retrain from scratch; the model is smaller now
(N_MAX~50-64 instead of ~250-500), so training is faster -- measure with
src/model/gpu_cost.py.
"""
import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from src.common.domain_scaling import MAX_DENSITY, MIN_RADIUS as R_MIN
from src.common.compat import load_checkpoint, normalize
from src.common.domain_scaling import (DEFAULT_DDIM_STEPS, DEFAULT_GUIDANCE, DEFAULT_L, DEFAULT_M1, DEFAULT_M2, DEFAULT_MAX_CLUSTER, DEFAULT_OVERLAP_MARGIN, DEFAULT_PORTS_PER_SIDE, DEFAULT_PORT_MARGIN, DEFAULT_SEED, DEFAULT_TIMEOUT)
from src.common.maps_io import file_sha256

NULL_CLASS = 2

# Fixed seed of the HELD-OUT TEST split. It is deliberately NOT --seed:
# the test holdout must not change when someone retrains with a different
# training seed by mistake, or no run would be comparable with any other.
# It only affects which rows become test; train/val keep using --seed.
TEST_SPLIT_SEED = 20261009

# Name of the sidecar with the provenance of the test holdout (source dataset
# fingerprint + source row indices). Kept OUT of the .npz so the holdout has
# exactly the columns of the dataset that produced it (obs, n, success, L).
TEST_HOLDOUT_META_SUFFIX = ".meta.json"

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
# --- `train` subcommand: the training hyperparameters of the campaign -----
STEPS = 100000              # --steps
TRAIN_BATCH = 512           # --batch
LR = 3e-4                   # --lr
WARMUP = 2000               # --warmup
EMA = 0.9995                # --ema
CFG_DROPOUT = 0.15          # --cfg-dropout
D_MODEL = 256               # --d-model
LAYERS = 8                  # --layers
HEADS = 8                   # --heads
T = 1000                    # --T
EVAL_EVERY = 2000           # --eval-every
COMPILE = False             # --compile
TEST_FRAC = 0.05            # --test-frac
VAL_FRAC = 0.05             # --val-frac
OVERWRITE_TEST = False      # --overwrite-test
# The campaign trained on universal tiles only: the 18 shipped checkpoints
# store success_only: true, so the flag's store_true default (False) is NOT
# the value the campaign used.
SUCCESS_ONLY = True

# --- `sample` subcommand --------------------------------------------------
L_FIXED = DEFAULT_L         # --L (the tile scheme uses a fixed 1.0)
PORTS_PER_SIDE = DEFAULT_PORTS_PER_SIDE  # --ports-per-side
PORT_MARGIN = DEFAULT_PORT_MARGIN        # --port-margin
MAX_CLUSTER = DEFAULT_MAX_CLUSTER        # --max-cluster
OVERLAP_MARGIN = DEFAULT_OVERLAP_MARGIN  # --overlap-margin
GUIDANCE = DEFAULT_GUIDANCE              # --guidance
ETA = 0.0                                # --eta
TIMEOUT = DEFAULT_TIMEOUT                # --timeout
NUM_OBS = 0                              # --num-obs (0 = random in [min,max] per map)
SAMPLE_BATCH = 500                       # --batch (sample subcommand; TRAIN_BATCH is the training one)


# ==========================================================================
# Normalization (L-aware for generality; in practice L=1.0 always)
# ==========================================================================
def r_max(n, L):
    """max_radius(n, L) vectorized -- see src/common/domain_scaling.py:max_radius."""
    return (MAX_DENSITY * L.float() ** 2 / (n.float().clamp_min(1) * math.pi)).sqrt()


def normalize(obs, n, L):
    rmax = r_max(n, L).view(-1, 1)
    Lc = L.view(-1, 1).float()
    return torch.stack([obs[..., 0] / Lc * 2 - 1,
                        obs[..., 1] / Lc * 2 - 1,
                        (obs[..., 2] - R_MIN) / (rmax - R_MIN) * 2 - 1], dim=-1)


def denormalize(z, n, L):
    rmax = r_max(n, L).view(-1, 1)
    Lc = L.view(-1, 1).float()
    return torch.stack([(z[..., 0] + 1) / 2 * Lc,
                        (z[..., 1] + 1) / 2 * Lc,
                        R_MIN + (z[..., 2] + 1) / 2 * (rmax - R_MIN)], dim=-1)


def build_mask(n, n_max, device):
    return torch.arange(n_max, device=device)[None, :] < n[:, None]


# ==========================================================================
# Cosine schedule
# ==========================================================================
def cosine_alphas_cumprod(T=1000, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = torch.clamp(1 - ac[1:] / ac[:-1], 1e-5, 0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def time_embedding(t, d):
    half = d // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    arg = t.float()[:, None] * freq[None, :]
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


# ==========================================================================
# Model (WITHOUT a path embedding -- see the module docstring)
# ==========================================================================
class Block(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, c, pad):
        g1, b1, a1, g2, b2, a2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        x = self.n1(h) * (1 + g1) + b1
        x, _ = self.att(x, x, x, key_padding_mask=pad, need_weights=False)
        h = h + a1 * x
        x = self.n2(h) * (1 + g2) + b2
        return h + a2 * self.mlp(x)


class SetDenoiser(nn.Module):
    def __init__(self, d=256, layers=8, heads=8):
        super().__init__()
        self.d = d
        self.entrada = nn.Linear(3, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.n_mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.clase = nn.Embedding(3, d)
        self.bloques = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.n_final = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_final = nn.Linear(d, 2 * d)
        self.salida = nn.Linear(d, 3)
        for m in (self.ada_final, self.salida):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, n, class_labels, pad):
        n_max = x.shape[1]  # the width of the tensor IS the checkpoint capacity
        c = (self.t_mlp(time_embedding(t, self.d)) + self.n_mlp((n.float() / n_max)[:, None])
             + self.clase(class_labels))
        h = self.entrada(x)
        for b in self.bloques:
            h = b(h, c, pad)
        g, s = self.ada_final(c).unsqueeze(1).chunk(2, dim=-1)
        return self.salida(self.n_final(h) * (1 + g) + s)


def build_model(cfg):
    return SetDenoiser(cfg["d_model"], cfg["layers"], cfg["heads"])


# ==========================================================================
# Training
# ==========================================================================
def select_device(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_raw(path_npz):
    """Raw, unfiltered columns of a tiles .npz, as plain numpy.

    ``(obs, n, success, L, success_key)``. No normalization, no torch, no
    success filtering: the 3-way split has to see the dataset exactly as it
    is on disk so the holdout it writes back is a faithful partition of it.
    """
    with np.load(path_npz) as d:
        files = set(d.files)
        if "obs" not in files or "n" not in files:
            raise SystemExit(f"{path_npz}: expected at least the columns 'obs' and 'n' "
                             f"(found {sorted(files)}).")
        obs = np.ascontiguousarray(d["obs"], dtype=np.float32)
        n = d["n"].astype(np.int64)
        success_key = next((k for k in ("success", "exito") if k in files), None)
        if success_key is None:
            raise SystemExit(f"{path_npz}: no 'success' column (nor the legacy 'exito') -- "
                             f"found {sorted(files)}.")
        success = d[success_key].astype(bool)
        if "L" in files:
            L = d["L"].astype(np.float32)
        else:
            print("WARNING: the .npz has no 'L' column -- L=1.0 is assumed for all maps.")
            L = np.ones(len(n), dtype=np.float32)
    return obs, n, success, L, success_key


def to_tensors(obs, n, L, success, device):
    """Normalized ``x0`` plus the conditioning tensors of a subset of rows."""
    obs_t = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(device)
    n_t = torch.from_numpy(np.ascontiguousarray(n, dtype=np.int64)).to(device)
    L_t = torch.from_numpy(np.ascontiguousarray(L, dtype=np.float32)).to(device)
    success_t = torch.from_numpy(np.ascontiguousarray(success, dtype=bool)).long().to(device)
    n_max = obs_t.shape[1]
    x0 = normalize(obs_t, n_t, L_t) * build_mask(n_t, n_max, device)[..., None]
    return x0, n_t, success_t, L_t


def _default_test_out(data_path):
    return os.path.join(os.path.dirname(os.path.abspath(data_path)) or ".", "test_holdout.npz")


def prepare_test_holdout(args, test_out, obs, n, success, L, success_key):
    """Return the source row indices of the held-out TEST split.

    Subtracted from the dataset BEFORE train/val, with a FIXED seed
    (``TEST_SPLIT_SEED``) that does not depend on ``--seed``. Written once per
    dataset to ``--test-out``; afterwards it is reused untouched (unless
    ``--overwrite-test``) so every training run -- whatever its seed -- is
    evaluated against the exact same unseen tiles.
    """
    N = len(n)
    meta_path = test_out + TEST_HOLDOUT_META_SUFFIX

    if os.path.exists(test_out) and not OVERWRITE_TEST:
        if not os.path.exists(meta_path):
            raise SystemExit(
                f"{test_out} already exists but its provenance metadata {meta_path} is missing, "
                f"so it cannot be proven to belong to {args.data!r}. The test holdout is fixed "
                f"once per dataset: pass --overwrite-test to regenerate it (that CHANGES the "
                f"split), or point --test-out somewhere else.")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        digest = file_sha256(args.data)
        if meta.get("data_sha256") != digest or int(meta.get("data_rows", -1)) != N:
            raise SystemExit(
                f"--test-out {test_out} was generated from another dataset "
                f"(rows={meta.get('data_rows')}, sha256={str(meta.get('data_sha256'))[:12]}...) and "
                f"does not match --data {args.data} (rows={N}, sha256={digest[:12]}...). The test "
                f"holdout is fixed once per dataset: pass --overwrite-test to replace it, or use "
                f"another --test-out.")
        idx = np.asarray(meta["test_indices"], dtype=np.int64)
        print(f"Test holdout REUSED (not regenerated): {test_out} | {len(idx)} tiles | "
              f"fixed seed {meta.get('seed', TEST_SPLIT_SEED)} | source {meta.get('data')}")
        return idx

    if TEST_FRAC <= 0.0:
        print("WARNING: --test-frac=0 -- no held-out test set is written. The generative-quality "
              "tests will not have an uncontaminated real reference.")
        return np.zeros(0, dtype=np.int64)

    n_test = max(1, min(int(round(N * TEST_FRAC)), N - 1))
    rng = np.random.default_rng(TEST_SPLIT_SEED)
    idx = np.sort(rng.permutation(N)[:n_test])

    os.makedirs(os.path.dirname(os.path.abspath(test_out)) or ".", exist_ok=True)
    # Same columns as the input dataset (obs, n, success/exito, L); n is stored
    # back as int16, the canonical tiles dtype (n <= a few hundred at most).
    np.savez_compressed(test_out, obs=obs[idx], n=n[idx].astype(np.int16),
                        **{success_key: success[idx]}, L=L[idx])

    meta = {
        "data": os.path.abspath(args.data),
        "data_rows": int(N),
        "data_sha256": file_sha256(args.data),
        "test_rows": int(len(idx)),
        "test_frac": float(TEST_FRAC),
        "seed": int(TEST_SPLIT_SEED),
        "n_max": int(obs.shape[1]),
        "success_key": success_key,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_indices": idx.tolist(),
        "note": ("Held-out TEST split, fixed seed independent of the training --seed. NEVER "
                 "used for training or checkpoint selection. Real reference for "
                 "src/evaluation/paper_metrics.py --reference-data; run "
                 "src/evaluation/check_split_leakage.py once before trusting any metric."),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"Test holdout GENERATED: {test_out} | {len(idx)} tiles ({len(idx)/N:.2%} of {N}) | "
          f"fixed seed {TEST_SPLIT_SEED} | source {args.data}")
    return idx


def diffusion_loss(model, ab, x0, n, class_labels, device, gen=None):
    B, n_max = x0.shape[0], x0.shape[1]
    m = build_mask(n, n_max, device)
    t = torch.randint(0, len(ab), (B,), device=device, generator=gen)
    eps = torch.randn(x0.shape, device=device, generator=gen) * m[..., None]
    a = ab[t].view(-1, 1, 1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    pred = model(xt, t, n, class_labels, ~m)
    return (((pred.float() - eps) ** 2) * m[..., None]).sum() / (m.sum() * 3)


def train(args):
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    use_cuda = device.startswith("cuda")
    amp = use_cuda and torch.cuda.is_bf16_supported()
    os.makedirs(args.dir, exist_ok=True)

    if not (0.0 <= TEST_FRAC < 1.0) or not (0.0 <= VAL_FRAC < 1.0):
        raise SystemExit("--test-frac and --val-frac must be in [0, 1).")
    if TEST_FRAC + VAL_FRAC >= 1.0:
        raise SystemExit("--test-frac + --val-frac must be < 1.0 (they are disjoint fractions).")

    # Raw dataset first: the test holdout is carved out BEFORE anything else,
    # with the fixed TEST_SPLIT_SEED -- it does not depend on --seed.
    obs_all, n_all, success_all, L_all, success_key = load_raw(args.data)
    N, n_max = len(n_all), obs_all.shape[1]
    test_out = args.test_out or _default_test_out(args.data)
    test_reused = os.path.exists(test_out) and not OVERWRITE_TEST
    idx_test = prepare_test_holdout(args, test_out, obs_all, n_all, success_all, L_all, success_key)

    # The rest is split train/val exactly as before (randperm seeded by --seed),
    # so training itself stays reproducible.
    is_test = np.zeros(N, dtype=bool)
    is_test[idx_test] = True
    rest = np.nonzero(~is_test)[0]
    if len(rest) < 2:
        raise SystemExit(f"only {len(rest)} rows left after removing {len(idx_test)} test rows -- "
                         f"lower --test-frac or use a bigger --data.")
    perm = torch.randperm(len(rest), device=device)
    n_val = min(max(256, int(round(len(rest) * VAL_FRAC))), len(rest) - 1)
    iv = rest[perm[:n_val].cpu().numpy()]
    it = rest[perm[n_val:].cpu().numpy()]
    if SUCCESS_ONLY:
        it = it[success_all[it]]
        iv = iv[success_all[iv]]

    # Tensors only for the rows training/validation may see. The test holdout
    # is NEVER converted, never put on the device, never touched again here.
    x0, n, class_labels, L = to_tensors(obs_all[it], n_all[it], L_all[it], success_all[it], device)
    x0_v, n_v, cl_v_all, L_v = to_tensors(obs_all[iv], n_all[iv], L_all[iv], success_all[iv], device)
    n_max = x0.shape[1]
    n_train_success = n_all[it][success_all[it]]
    del obs_all, n_all, L_all

    status = "reused" if test_reused else "written"
    print(f"Data: {N} maps ({class_labels.float().mean():.1%} universal) | n_max={n_max} | "
          f"L in [{L.min().item():.2f}, {L.max().item():.2f}] | device {device}")
    print(f"Split: train {len(it)} | val {len(iv)} | test {len(idx_test)} "
          f"(test fixed seed {TEST_SPLIT_SEED}, val seed {args.seed})")
    print(f"Test holdout ({status}): {test_out}")
    print("  The test holdout is NEVER used for training or checkpoint selection; it exists only "
          "as the real reference of the generative-quality tests.")
    print("  Run ONCE, before trusting any metric, the near-duplicate leakage check:")
    print(f"    python3 -m src.evaluation.check_split_leakage --data {args.data} --test {test_out}")

    cfg = {"d_model": D_MODEL, "layers": LAYERS, "heads": HEADS, "T": T,
           "success_only": SUCCESS_ONLY, "data": args.data, "n_max": n_max,
           "hist_n_success": np.bincount(n_train_success, minlength=n_max + 1).tolist(),
           "test_holdout": test_out, "test_split_seed": TEST_SPLIT_SEED,
           "val_frac": VAL_FRAC, "test_frac": TEST_FRAC}
    model = build_model(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ab = cosine_alphas_cumprod(T).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_fwd = torch.compile(model) if COMPILE else model

    step0, best = 0, float("inf")
    path_last = os.path.join(args.dir, "ultimo.pt")
    if os.path.exists(path_last):
        ck = load_checkpoint(path_last, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step0, best = ck["step"], ck["best"]
        print(f"Resuming from step {step0} (best val {best:.5f})")

    def lr_at(step):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        prog = (step - WARMUP) / max(1, STEPS - WARMUP)
        return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    history = []
    t0 = time.time()
    accum = 0.0
    for step in range(step0, STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = torch.randint(len(it), (TRAIN_BATCH,), device=device)
        cl = class_labels[idx].clone()
        if not SUCCESS_ONLY:
            cl[torch.rand(len(cl), device=device) < CFG_DROPOUT] = NULL_CLASS
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss = diffusion_loss(model_fwd, ab, x0[idx], n[idx], cl, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(EMA, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        accum += loss.item()

        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            gen = torch.Generator(device=device).manual_seed(1234)
            n_val_maps = len(iv)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                vs = []
                for k in range(0, n_val_maps, TRAIN_BATCH):
                    sl = slice(k, k + TRAIN_BATCH)
                    cl_v = cl_v_all[sl] if not SUCCESS_ONLY else torch.ones_like(cl_v_all[sl])
                    vs.append(diffusion_loss(ema, ab, x0_v[sl], n_v[sl], cl_v, device, gen).item()
                              * (sl.stop - sl.start))
                val = sum(vs) / n_val_maps
            tr = accum / EVAL_EVERY
            accum = 0.0
            is_best = val < best
            best = min(best, val)
            history.append({"step": step + 1, "train": tr, "val_ema": val, "lr": lr_at(step)})
            print(f"step {step + 1:>7} | train {tr:.5f} | val(EMA) {val:.5f} | lr {lr_at(step):.2e} | "
                  f"{(time.time() - t0) / 60:.1f} min" + ("  * best" if is_best else ""))
            if is_best:
                torch.save({"model": ema.state_dict(), "config": cfg, "step": step + 1, "val": val},
                           os.path.join(args.dir, "diffusion_model.pt"))
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "best": best, "config": cfg}, path_last)
            with open(os.path.join(args.dir, "historial.json"), "a", encoding="utf-8") as f:
                f.write(json.dumps(history[-1]) + "\n")
    torch.save({"model": ema.state_dict(), "config": cfg, "step": STEPS},
               os.path.join(args.dir, "modelo_final_ema.pt"))
    print("Done.")


# ==========================================================================
# Sampling
# ==========================================================================
@torch.no_grad()
def sample_batch(model, ab, n, L, guidance, steps, eta, gen, device, success_only, n_max):
    B = len(n)
    m = build_mask(n, n_max, device)
    pad = ~m
    x = torch.randn(B, n_max, 3, device=device, generator=gen) * m[..., None]
    ts = torch.linspace(len(ab) - 1, 0, steps, device=device).long()
    c_univ = torch.ones(B, dtype=torch.long, device=device)
    c_null = torch.full((B,), NULL_CLASS, dtype=torch.long, device=device)
    for i, t in enumerate(ts):
        tt = t.repeat(B)
        eps = model(x, tt, n, c_univ, pad).float()
        if guidance != 0 and not success_only:
            eps_u = model(x, tt, n, c_null, pad).float()
            eps = (1 + guidance) * eps - guidance * eps_u
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
        if eta > 0 and i + 1 < len(ts):
            sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).sqrt()
            x = (a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps
                 + sigma * torch.randn(x.shape, device=device, generator=gen))
        else:
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        x = x * m[..., None]
    return denormalize(x, n, L)


# --------------------------------------------------------------------------
def _largest_cluster(obs, overlap_margin):
    """Size of the largest overlap cluster of a full map
    (Union-Find; two obstacles belong to the same cluster if
    center distance - (r_i+r_j) < overlap_margin) -- the SAME criterion that
    src/dataset/universal_generator.py uses for --max-cluster when generating the data.
    The model does not learn this constraint reliably on its own
    (see check_clusters.py of the previous scheme; the same risk applies
    here) -- that is why it is filtered instead of trusting the model to respect it."""
    n = len(obs)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    xs, ys, rs = obs[:, 0], obs[:, 1], obs[:, 2]
    for i in range(n):
        dx = xs[i + 1:] - xs[i]
        dy = ys[i + 1:] - ys[i]
        gap = np.hypot(dx, dy) - (rs[i + 1:] + rs[i])
        for j_rel in np.where(gap < overlap_margin)[0]:
            j = i + 1 + int(j_rel)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    import collections
    sizes = collections.Counter(find(i) for i in range(n))
    return max(sizes.values())


def _universal_filter(obs, protected, port_margin, min_radius, max_radius, max_obstacles, L):
    """Analogous to src/dataset/verify_maps.py:geometric_filter, but against ANY
    number of protected points (the 4*P ports + the center) instead of
    only 2 (start/goal). Clamps position/radius and removes obstacles that
    invade any of the protected points -- the same port_margin that
    src/dataset/universal_generator.py uses during generation (it is NOT the robot's
    clearance, that is verified separately by HPPM)."""
    if len(obs) == 0:
        return obs
    filtered = []
    for x, y, r in obs:
        x = float(np.clip(x, 0.0, L))
        y = float(np.clip(y, 0.0, L))
        r = float(np.clip(r, min_radius, max_radius))
        if any(math.hypot(x - px, y - py) <= r + port_margin for px, py in protected):
            continue
        filtered.append([x, y, r])
        if len(filtered) >= max_obstacles:
            break
    return np.array(filtered, dtype=np.float32) if filtered else np.zeros((0, 3), dtype=np.float32)


def _verify_universal_candidate(args_tuple):
    """Runs the 4*P port->center routes IN ORDER on a candidate already
    generated by the model, and stops at the first one that fails -- WITHOUT repairing
    (unlike src/dataset/universal_generator.py:verify_universal, which repairs: here
    we are only evaluating what the model already produced, not generating new
    data). 'success' = the 4*P routes ALL succeeded."""
    from src.dataset.verify_maps import compute_repulsion, verify_one_map
    (idx, obs, exe, step_size, timeout, ports, center,
     robot_radius, safety_margin, arrival_radius, m1, m2) = args_tuple
    t_cpu0 = time.perf_counter()
    obs_arr = np.asarray(obs, dtype=np.float64)
    runs, res = 0, None
    for port_name, point in ports.items():
        rep = compute_repulsion(obs_arr, start=point, goal=center, sign_mode="mass",
                                robot_radius=robot_radius, safety_margin=safety_margin)
        _, res = verify_one_map((idx, rep, exe, step_size, timeout, False,
                                 point, center, robot_radius, arrival_radius, m1, m2))
        runs += 1
        if res["state"] != "success":
            res["failure_at"] = port_name
            break
    res["idx"], res["num_obs"], res["corridas_hppm"] = idx, len(obs_arr), runs
    res["verification_cpu_time_sec"] = time.perf_counter() - t_cpu0
    return res


def sample(args):
    from src.common.maps_io import save_maps
    from src.dataset.verify_maps import MIN_OBS_GENERATOR, detect_collapse, build_report
    from src.dataset.universal_generator import ports_and_center
    from src.common.domain_scaling import MIN_RADIUS, scaled_constants
    from src.common.domain_scaling import max_radius as max_radius_L

    device = select_device(args.device)
    path_ckpt = args.model or os.path.join(args.dir, "diffusion_model.pt")
    ck = load_checkpoint(path_ckpt, map_location=device)
    cfg = ck["config"]
    n_max = cfg["n_max"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    ab = cosine_alphas_cumprod(cfg["T"]).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if NUM_OBS > 0:
        if NUM_OBS > n_max:
            raise SystemExit(f"--num-obs={NUM_OBS} exceeds the checkpoint capacity "
                             f"(n_max={n_max}).")
        ns = np.full(args.n, NUM_OBS)
    elif args.num_obs_min is not None or args.num_obs_max is not None:
        lo = args.num_obs_min if args.num_obs_min is not None else 1
        hi = args.num_obs_max if args.num_obs_max is not None else n_max
        if hi > n_max:
            raise SystemExit(f"--num-obs-max={hi} exceeds the checkpoint capacity (n_max={n_max}).")
        ns = rng.integers(lo, hi + 1, size=args.n)
    else:
        h = np.array(cfg["hist_n_success"], dtype=float)
        ns = rng.choice(len(h), size=args.n, p=h / h.sum())

    L = L_FIXED
    ports, center = ports_and_center(L, PORTS_PER_SIDE)
    protected = list(ports.values()) + [center]
    const = scaled_constants(L)
    port_margin_i = PORT_MARGIN * L

    print(f"Model {path_ckpt} (step {ck.get('step')}) | n={args.n} | num_obs={NUM_OBS or 'range/distribution'} | "
          f"L={L} (checkpoint n_max={n_max}) | P={PORTS_PER_SIDE} ({len(ports)} ports) | "
          f"guidance={GUIDANCE} | DDIM steps={args.ddim_steps} | eta={ETA}")

    t0 = time.time()
    raw = []
    Lb_const = torch.full((SAMPLE_BATCH,), L, device=device)
    for k in range(0, args.n, SAMPLE_BATCH):
        nb = torch.from_numpy(ns[k:k + SAMPLE_BATCH]).long().to(device)
        Lb = Lb_const[:len(nb)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            o = sample_batch(model, ab, nb, Lb, GUIDANCE, args.ddim_steps, ETA, gen,
                             device, cfg["success_only"], n_max).float().cpu().numpy()
        for j in range(len(nb)):
            raw.append(o[j, :int(nb[j])])
    t_sampling = time.time() - t0

    t_filter0 = time.time()
    tasks, idx_ok, maps_ok = [], [], []
    n_few = n_collapsed = n_bad_cluster = 0
    removed = 0
    for i, o in enumerate(raw):
        max_radius_i = max_radius_L(max(len(o), 1), L)
        f = _universal_filter(o, protected, port_margin_i, MIN_RADIUS, max_radius_i, n_max, L)
        removed += len(o) - len(f)
        if len(f) < MIN_OBS_GENERATOR:
            n_few += 1
            continue
        if detect_collapse(f):
            n_collapsed += 1
            continue
        if MAX_CLUSTER is not None and _largest_cluster(f, OVERLAP_MARGIN) > MAX_CLUSTER:
            n_bad_cluster += 1
            continue
        # NOTE: the compiled engine ignores m1/m2 (it overwrites them with 1.0/4.0).
        tasks.append((i, f.tolist(), args.hppm_exe, const["step_size"], TIMEOUT,
                      ports, center, const["robot_radius"], const["safety_margin"],
                      const["arrival_radius"], DEFAULT_M1, DEFAULT_M2))
        idx_ok.append(i)
        maps_ok.append(f)
    t_filter = time.time() - t_filter0
    print(f"  sampling {t_sampling:.1f}s | filter {t_filter:.1f}s | filter removed {removed} obstacles "
          f"({removed / max(1, sum(len(o) for o in raw)):.1%}) | "
          f"discarded: {n_few} few, {n_collapsed} collapsed, {n_bad_cluster} with cluster>"
          f"{MAX_CLUSTER} | verifying {len(tasks)} candidates against {len(ports)} "
          f"ports each (up to {len(tasks) * len(ports)} HPPM runs)...")

    t_verify0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_verify_universal_candidate, tasks, chunksize=8))
    t_verify = time.time() - t_verify0   # wall-clock with --workers processes in parallel
    t_verify_cpu = sum(r.get("verification_cpu_time_sec", 0.0) for r in results)
    os.makedirs(args.output_dir, exist_ok=True)
    save_maps(args.output_dir, idx_ok, maps_ok, n_max=n_max)
    rep = build_report(results, "success", n_few + n_collapsed + n_bad_cluster,
                       n_empty=0, n_collapsed=n_collapsed, n_few=n_few)
    rep["mapas_descartados_cluster"] = n_bad_cluster
    rep.update({"model": "set_diffusion (universal tiles)", "checkpoint": path_ckpt,
                "guidance": GUIDANCE, "num_obs_requested": NUM_OBS or "range/distribution",
                "obstacles_removed_by_filter": removed, "L": L, "n_max_checkpoint": n_max,
                "ports_per_side": PORTS_PER_SIDE, "port_margin": PORT_MARGIN,
                "max_cluster_requested": MAX_CLUSTER, "overlap_margin": OVERLAP_MARGIN,
                "num_obs_min": args.num_obs_min, "num_obs_max": args.num_obs_max,
                "ddim_steps": args.ddim_steps, "eta": ETA, "seed": args.seed,
                "sampling_time_sec": round(t_sampling, 3),
                "filter_time_sec": round(t_filter, 3),
                "verification_time_sec": round(t_verify, 3),
                "verification_cpu_time_sec": round(t_verify_cpu, 3),
                "workers": args.workers, "candidates_sampled": int(args.n),
                "timing_note": ("sampling = DDIM inference on the model device (wall-clock, "
                                "includes host transfer); filter = geometric filtering + collapse/"
                                "cluster checks (single process); verification = HPPM routes "
                                "(wall-clock with `workers` processes; verification_cpu_time_sec "
                                "is the summed per-candidate time across workers). Model loading "
                                "is excluded from all three.")})
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=float)
    with open(os.path.join(args.output_dir, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "tray"} for r in results], f, default=float)
    print(json.dumps({k: rep[k] for k in ("success_rate_over_total_generated", "verified_success_rate",
                                          "mean_num_obs_evaluated", "count_by_status")},
                     ensure_ascii=False, default=float))


def main():
    p = argparse.ArgumentParser(description="Diffusion over universal tiles")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("train")
    e.add_argument("--data", required=True)
    e.add_argument("--dir", default="artifacts/ckpt_tiles")
    e.add_argument("--device", default="auto")
    e.add_argument("--seed", type=int, default=DEFAULT_SEED)
    e.add_argument("--test-out", default=None,
                   help="where to persist the test holdout .npz (default: <dir of --data>/"
                        "test_holdout.npz). It is fixed once per dataset and reused as-is on later "
                        "runs, whatever --seed.")

    m = sub.add_parser("sample")
    m.add_argument("--dir", default="artifacts/ckpt_tiles")
    m.add_argument("--model", default=None, help="defaults to <dir>/diffusion_model.pt")
    m.add_argument("--n", type=int, default=2000)
    m.add_argument("--num-obs-min", type=int, default=None)
    m.add_argument("--num-obs-max", type=int, default=None)
    m.add_argument("--ddim-steps", type=int, default=DEFAULT_DDIM_STEPS)
    m.add_argument("--hppm-exe", default="artifacts/hppm_v2")
    m.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    m.add_argument("--output-dir", default="artifacts/tiles_muestras")
    m.add_argument("--device", default="auto")
    m.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = p.parse_args()
    train(args) if args.cmd == "train" else sample(args)


if __name__ == "__main__":
    main()
