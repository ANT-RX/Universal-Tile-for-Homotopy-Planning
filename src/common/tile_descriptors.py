"""
Shared per-tile descriptors used by the evaluation tests.

The ONE descriptor defined here is the grid occupancy already used by the
memorization test of ``src/evaluation/paper_metrics.py``: an ``k x k``
occupancy grid over the tile ``[0, side] x [0, side]``, each obstacle
weighted by its area ``r**2`` and the whole histogram normalized to sum 1.
It is deliberately coarse: it captures the spatial distribution of obstacle
mass without depending on the exact obstacle list, so two tiles are "near
duplicates" when their occupancy patterns coincide.

It lives here (instead of being copy-pasted) because two tests need the very
same definition of "tile similarity":

  * the memorization test of ``src/evaluation/paper_metrics.py``;
  * the near-duplicate split-leakage check of
    ``src/evaluation/check_split_leakage.py``.

If the descriptor changes here, every metric that uses it changes with it --
which is the point: train/test leakage has to be measured with the same ruler
as memorization.
"""
import numpy as np

# Was 8 (the historical `hist(obs, k=8)` of paper_metrics.test_diversity()):
# too coarse for sparse tiles (n<=10 populates at most 10 of 64 cells, so
# unrelated tiles collide into the same coarse histogram by chance -- see
# check_split_leakage.py's n=1-10 bin, which flagged ~35% "leakage" that was
# ~95% false positives once checked against raw obstacle coordinates).
OCCUPANCY_BINS = 16

# Universal tiles are always [0, 1] x [0, 1] (TILE_SIDE = 1.0, see
# src/common/domain_scaling.py); the side is a parameter only so the
# descriptor keeps working if tiles of another side are ever mixed in.
OCCUPANCY_SIDE = 1.0


def occupancy_hist(obs, k=OCCUPANCY_BINS, side=OCCUPANCY_SIDE):
    """Normalized ``k x k`` occupancy histogram of one tile.

    ``obs`` is an ``(m, 3)`` array of ``x, y, r``. Returns a ``(k*k,)``
    float array normalized to sum 1 (all zeros for an empty tile).

    Bit-for-bit the same thing the memorization test used before this module
    existed:

        np.histogram2d(obs[:, 0], obs[:, 1], bins=k, range=[[0, 1], [0, 1]],
                       weights=obs[:, 2] ** 2).ravel()  # then h / h.sum()
    """
    obs = np.asarray(obs, dtype=np.float64)
    if obs.size == 0:
        return np.zeros(k * k, dtype=np.float64)
    h, _, _ = np.histogram2d(obs[:, 0], obs[:, 1], bins=k,
                             range=[[0.0, side], [0.0, side]],
                             weights=obs[:, 2] ** 2)
    return (h / max(h.sum(), 1e-12)).ravel()


def occupancy_hists(obs, n, rows=None, k=OCCUPANCY_BINS, side=OCCUPANCY_SIDE):
    """Stack ``occupancy_hist`` for a set of padded tiles.

    ``obs`` is the padded ``(N, n_max, 3)`` array of an ``.npz`` dataset and
    ``n`` its ``(N,)`` obstacle-count column; ``rows`` selects which tiles to
    describe (default: all of them). Only the first ``n[i]`` obstacles of each
    tile are used, exactly like ``load_maps(columns=3)`` does.
    """
    obs = np.asarray(obs)
    n = np.asarray(n)
    if rows is None:
        rows = np.arange(len(n))
    return np.asarray([occupancy_hist(obs[i, :int(n[i])], k=k, side=side)
                       for i in rows], dtype=np.float64)


# ==========================================================================
# Nearest-neighbour search under this descriptor.
#
# A KD-tree (``scipy.spatial.cKDTree``) degenerates toward brute force past a
# few dozen dimensions, and at k=16 (OCCUPANCY_BINS**2 = 256-dim histograms)
# that degeneration is pathological, not just slow: one 570k-training x
# 30k-test query ran for over an hour and sat at ~43 GB resident on a
# 256-core machine without finishing -- the tree itself becomes huge and
# nearly useless at this dimensionality, independent of CPU/RAM available.
# Chunked brute force via the squared-distance expansion
# (||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, the ``- 2 a.b`` term done as one
# BLAS matmul per chunk) is both much faster and far lower memory here.
# ==========================================================================
def nearest_distance(H_query, H_index, batch_size=1000):
    """Exact nearest-neighbour Euclidean distance from each row of
    ``H_query`` to the closest row of ``H_index``.

    Assumes the rows of ``H_query`` are NOT themselves rows of ``H_index``
    (use ``nearest_other_distance`` when they are, e.g. a leave-one-out
    distance within the same pool). ``batch_size`` bounds the temporary
    ``(batch_size, len(H_index))`` distance matrix -- lower it if this runs
    out of memory for a very large index.
    """
    H_index = np.ascontiguousarray(H_index, dtype=np.float32)
    H_query = np.ascontiguousarray(H_query, dtype=np.float32)
    idx_norm = np.einsum("ij,ij->i", H_index, H_index)
    out = np.empty(len(H_query), dtype=np.float64)
    for start in range(0, len(H_query), batch_size):
        end = min(start + batch_size, len(H_query))
        q = H_query[start:end]
        q_norm = np.einsum("ij,ij->i", q, q)
        d2 = q_norm[:, None] + idx_norm[None, :] - 2.0 * (q @ H_index.T)
        np.maximum(d2, 0.0, out=d2)  # fp noise can make same-point d2 slightly negative
        out[start:end] = np.sqrt(d2.min(axis=1))
    return out


def nearest_other_distance(probe_rows, H_index, batch_size=1000):
    """Like ``nearest_distance``, but the probes ARE rows of ``H_index``
    (given by their integer row indices ``probe_rows``): each probe's own
    row (distance 0 to itself) is excluded before taking the minimum, i.e. a
    leave-one-out nearest-OTHER-neighbour distance within ``H_index``."""
    H_index = np.ascontiguousarray(H_index, dtype=np.float32)
    idx_norm = np.einsum("ij,ij->i", H_index, H_index)
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    out = np.empty(len(probe_rows), dtype=np.float64)
    for start in range(0, len(probe_rows), batch_size):
        end = min(start + batch_size, len(probe_rows))
        rows = probe_rows[start:end]
        q = H_index[rows]
        q_norm = idx_norm[rows]
        d2 = q_norm[:, None] + idx_norm[None, :] - 2.0 * (q @ H_index.T)
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(len(rows)), rows] = np.inf  # exclude self
        out[start:end] = np.sqrt(d2.min(axis=1))
    return out
