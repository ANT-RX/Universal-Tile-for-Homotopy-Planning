"""
server_diagnostics.py — Quick check that the new environment is as
expected BEFORE launching anything at volume: how many cores Python really
sees (vs "assigned"), whether torch sees the GPU and how much free VRAM there
is, disk space, and whether gcc is present to compile HPPMDinamico3.c.

It does not run anything from the project (neither HPPM nor the model) -- it is a pure
environment check. Usage:
    python3 -m src.pipeline.server_diagnostics
    python3 -m src.pipeline.server_diagnostics --expected-workers 30 --min-disk-gb 20
"""
import argparse
import os
import shutil
import subprocess
import sys


def line(ok, text):
    mark = "OK " if ok else "!! "
    print(f"[{mark}] {text}")
    return ok


def check_cpu(expected_workers):
    print("\n== CPU ==")
    detected = os.cpu_count()
    print(f"  os.cpu_count() (what the project scripts see by default): {detected}")
    if detected is None:
        return line(False, "os.cpu_count() returned None -- the defaults 'os.cpu_count()-2' will fail.")
    if detected != expected_workers:
        return line(False, f"os.cpu_count()={detected} != --expected-workers={expected_workers}. "
                           f"Do NOT trust the --workers defaults of the scripts -- always pass "
                           f"--workers {expected_workers} explicitly.")
    return line(True, f"os.cpu_count()={detected} matches what was expected. Even so, better to pass "
                      f"--workers {expected_workers} explicitly instead of trusting the default.")


def check_gpu():
    print("\n== GPU ==")
    try:
        import torch
    except ImportError:
        return line(False, "the 'torch' package is not installed in this Python. Install it (with the "
                           "correct CUDA support for the 4090) before continuing.")
    print(f"  torch {torch.__version__}")
    if not torch.cuda.is_available():
        return line(False, "torch.cuda.is_available() = False. Check the NVIDIA driver / the torch "
                           "installation with CUDA (torch installed without CUDA is the most common "
                           "reason).")
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    vram_gb = props.total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    print(f"  device: {name} | total VRAM: {vram_gb:.1f} GB | bf16: {bf16}")
    ok = ("4090" in name) and bf16
    if "4090" not in name:
        line(False, f"the device name does not say '4090' ({name!r}) -- confirm that it is the right GPU.")
    if not bf16:
        line(False, "bf16 not supported -- unexpected on a 4090, check the driver/CUDA toolkit.")
    return line(ok, "GPU correctly detected by torch." if ok else "check above.")


def check_disk(path, min_gb):
    print("\n== Disk ==")
    total, used, free = shutil.disk_usage(path)
    free_gb = free / 1e9
    print(f"  free in {path!r}: {free_gb:.1f} GB")
    return line(free_gb >= min_gb, f"free_gb={free_gb:.1f} vs requested minimum={min_gb}")


def check_gcc():
    print("\n== Compiler (for HPPMDinamico3.c) ==")
    path = shutil.which("gcc")
    if not path:
        return line(False, "could not find 'gcc' in PATH. Install it (apt install build-essential or "
                           "equivalent) before compiling HPPMDinamico3.c.")
    try:
        v = subprocess.run(["gcc", "--version"], capture_output=True, text=True, timeout=5)
        print(f"  {path} -> {v.stdout.splitlines()[0]}")
    except Exception as e:
        return line(False, f"gcc found at {path} but running --version failed: {e}")
    return line(True, f"gcc available at {path}")


def check_versions():
    """Installed torch/numpy/scipy/scikit-learn vs. the pins in requirements.txt --
    different versions between the 6 seed servers are a silent reproducibility leak."""
    import importlib
    import os
    import re
    print("\n== Python packages vs requirements.txt ==")
    req = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "requirements.txt")
    pins = {}
    if os.path.exists(req):
        with open(req, encoding="utf-8") as f:
            for ln in f:
                m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", ln.strip())
                if m:
                    pins[m.group(1).lower()] = m.group(2)
    print(f"  python {sys.version.split()[0]}")
    ok = True
    for pkg, mod in (("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy"),
                     ("scikit-learn", "sklearn")):
        try:
            have = importlib.import_module(mod).__version__.split("+")[0]
        except Exception as e:
            ok = line(False, f"{pkg}: cannot import ({e})") and ok
            continue
        want = pins.get(pkg)
        if want is None:
            line(True, f"{pkg} {have} (not pinned)")
        else:
            ok = line(have == want, f"{pkg} installed={have} pinned={want}") and ok
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected-workers", type=int, default=30)
    p.add_argument("--min-disk-gb", type=float, default=20.0)
    p.add_argument("--disk-path", default=".")
    args = p.parse_args()

    results = [
        check_cpu(args.expected_workers),
        check_gpu(),
        check_disk(args.disk_path, args.min_disk_gb),
        check_gcc(),
        check_versions(),
    ]
    print("\n" + "=" * 40)
    if all(results):
        print("Everything OK. Continue with the initial tests (see the next step).")
    else:
        print("There are items marked '!!' above -- resolve them before running anything at volume.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
