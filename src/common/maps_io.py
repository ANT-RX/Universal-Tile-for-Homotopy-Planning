"""
Compact map input/output: ONE maps.npz file per run instead of a folder
mapa_NNNNN/ with two files per map (which, with tens of thousands of maps,
exhausts the disk or the inodes).

maps.npz format:
    obs : float32 (N, 100, 4)  x, y, r, p_neto  (padded with zeros)
    n   : int16   (N,)         actual obstacles of each map
    idx : int32   (N,)         index of the map in the run (the same as in
                               results_detail.json)

load_maps() also reads the old format (mapa_* folders), so previous results
remain analyzable.
"""
import glob
import hashlib
import os

import numpy as np

N_MAX = 500  # WAS 100: it SILENTLY truncated any map with more than 100
             # obstacles (save_maps did m[:N_MAX] without warning). With the
             # new dataset reaching n=500, that lost obstacles without error.
             # NOTE: this is the STORAGE limit (.npz), not the N_MAX of the
             # diffusion model (set_diffusion.py is still at 100 -- see section
             # 1 of the design; it has to be raised there too, and retrained
             # from scratch, before the model can generate maps with more than
             # 100 obstacles).
FILENAME = "maps.npz"


def file_sha256(path, chunk=1 << 22):
    """SHA-256 of a file's bytes, for cheap provenance checks.

    Used to prove that a persisted held-out test split belongs to the exact
    dataset it was carved from (see ``src/model/set_diffusion.py``
    ``prepare_test_holdout``), so it is never silently reused for another one.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_maps(folder, idxs, maps, n_max: int = None):
    """maps: list of arrays (n_i, 3) or (n_i, 4)."""
    n_max = N_MAX if n_max is None else n_max
    os.makedirs(folder, exist_ok=True)
    obs = np.zeros((len(maps), n_max, 4), dtype=np.float32)
    n = np.zeros(len(maps), dtype=np.int16)
    for k, m in enumerate(maps):
        m = np.asarray(m, dtype=np.float32)
        if len(m) > n_max:
            raise ValueError(
                f"save_maps: map {idxs[k]} has {len(m)} obstacles > n_max={n_max}. "
                "Previously this was SILENTLY truncated -- now it is explicitly rejected. "
                "Raise n_max (or the module's N_MAX) if this is expected.")
        obs[k, :len(m), :m.shape[1]] = m
        n[k] = len(m)
    np.savez_compressed(os.path.join(folder, FILENAME), obs=obs, n=n,
                        idx=np.asarray(idxs, dtype=np.int32))


def load_maps(folder, columns=3):
    """Returns (idxs, maps) with maps as a list of arrays (n_i, columns)."""
    path = os.path.join(folder, FILENAME)
    if os.path.exists(path):
        with np.load(path) as d:
            # extract once: indexing an NpzFile inside a loop
            # decompresses the whole array on every access
            obs, n, idx = d["obs"], d["n"], d["idx"]
        return ([int(i) for i in idx],
                [obs[k, :int(n[k]), :columns] for k in range(len(n))])
    idxs, maps = [], []
    for sub in sorted(glob.glob(os.path.join(folder, "mapa_*"))):
        m = np.loadtxt(os.path.join(sub, "obstaculos.txt"), ndmin=2)[:, :columns].astype(np.float32)
        idxs.append(int(os.path.basename(sub).split("_")[1]))
        maps.append(m)
    return idxs, maps
