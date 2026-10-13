"""
Measures, on the real GPU, the time per step and the peak training memory of
the diffusion denoiser of src/model/set_diffusion.py at different n (number of
obstacles per map) and different batches, without needing to have data or
checkpoints.

It serves to replace the extrapolations (CPU -> GPU, a single measured point)
with real numbers from your hardware, BEFORE committing to a long training
run at large n.

Usage:
    python3 src/model/gpu_cost.py --n 100 300 500 1000 2000 --batch 512
    python3 src/model/gpu_cost.py --n 500 1000 2000 5000 --batch 512 256 128 64 32
"""
import argparse
import time

import torch

from src.model.set_diffusion import SetDenoiser, cosine_alphas_cumprod

# ==========================================================================
# FROZEN campaign configuration.
# These were argparse flags; they are now module constants holding the exact
# `default=` the campaign ran with. Do not change a value here.
# ==========================================================================
D_MODEL = 256       # --d-model
LAYERS = 8          # --layers
HEADS = 8           # --heads
MEASURED_STEPS = 10  # --measured-steps
BATCH = [512]        # --batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[16, 32, 50, 64],
                   help="default according to the universal tile scheme (n<=50-64), not to the "
                        "previous large-domain scheme (n up to 2000)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    use_cuda = device.startswith("cuda")
    if not use_cuda:
        print("WARNING: no CUDA available; the numbers will not represent real GPU performance.")

    ab = cosine_alphas_cumprod(1000).to(device)
    print(f"{'n':>6} | {'batch':>6} | {'s/step':>8} | {'100k steps':>12} | {'peak mem':>10}")
    for n in args.n:
        N_MAX = n  # the model no longer has its own fixed N_MAX: the capacity of tensor x
                   # (dim 1) IS the effective n_max for this one-off test -- see forward()
        model = SetDenoiser(D_MODEL, LAYERS, HEADS).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for batch in BATCH:
            if use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                x0 = torch.randn(batch, N_MAX, 3, device=device)
                nb = torch.full((batch,), n, dtype=torch.long, device=device)
                class_labels = torch.ones(batch, dtype=torch.long, device=device)
                # no padding: the n maps of this test have exactly n
                # "real" obstacles each (the module's build_mask() is computed
                # with n_max=x.shape[1], which here coincides with n -- no padding is needed)
                pad = torch.zeros(batch, n, dtype=torch.bool, device=device)
                t = torch.randint(0, 1000, (batch,), device=device)

                def one_step():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                        eps = torch.randn_like(x0)
                        a = ab[t].view(-1, 1, 1)
                        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
                        pred = model(xt, t, nb, class_labels, pad)
                        loss = (pred - eps).pow(2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                for _ in range(3):  # warm-up (kernel compilation, cache)
                    one_step()
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(MEASURED_STEPS):
                    one_step()
                if use_cuda:
                    torch.cuda.synchronize()
                dt = (time.time() - t0) / MEASURED_STEPS
                mem = torch.cuda.max_memory_allocated(device) / 1e9 if use_cuda else float("nan")
                print(f"{n:>6} | {batch:>6} | {dt:>7.3f}s | {dt * 100000 / 3600:>10.1f} h | {mem:>8.2f} GB")
            except torch.cuda.OutOfMemoryError:
                print(f"{n:>6} | {batch:>6} | OUT OF MEMORY (OOM) -- lower the batch size (BATCH is a frozen constant)")
                if use_cuda:
                    torch.cuda.empty_cache()
        del model, opt


if __name__ == "__main__":
    main()
