"""
HPPM invocation and trajectory evaluation for generated maps.

Pipeline:
  1. Computes repulsion parameters (deterministic criterion from the paper)
  2. Generates input files for HPPM (obstaculos.txt + pendientes.txt)
  3. Runs the HPPM C executable on each map
  4. Evaluates the trajectory and computes the verified success rate

Usage:
    # Compile HPPM (once)
    gcc -O2 -o hppm HPPMDinamico3.c -lm
"""

import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

# Physical constants live in src.common.domain_scaling (single source of truth).
# The aliases below keep the local stage names, on purpose: see the WARNING
# further down -- ROBOT_RADIUS and GENERATOR_EXCLUSION_RADIUS are TWO DIFFERENT
# things and must never be merged into one symbol.
from src.common.domain_scaling import ARRIVAL_RADIUS_BASE, DEFAULT_M1, DEFAULT_M2, STEP_SIZE_BASE
from src.common.domain_scaling import GENERATOR_EXCLUSION_RADIUS_BASE as GENERATOR_EXCLUSION_RADIUS
from src.common.domain_scaling import MAX_DENSITY as REAL_MAX_DENSITY
from src.common.domain_scaling import MIN_RADIUS as REAL_MIN_RADIUS
from src.common.domain_scaling import ROBOT_RADIUS_BASE as ROBOT_RADIUS
from src.common.domain_scaling import SAFETY_MARGIN_BASE as SAFETY_MARGIN
from src.common.domain_scaling import START_GOAL_MARGIN_BASE

# ==========================================================================
# Physical and dataset-distribution constants
#
# WARNING: these are TWO different things that historically were conflated
# into a single value ("RADIO_ROBOT"). Do not unify them.
#
#   ROBOT_RADIUS (0.02) = effective radius used ONLY in the Python evaluation
#   (evaluate_trajectory) to decide whether the trajectory "touches" an
#   obstacle, that is, to label the 'collision' state. It is a convention of
#   THIS pipeline: the C engine (HPPMDinamico3.c) treats the robot as a POINT
#   and has no robot radius at all.
#
#   GENERATOR_EXCLUSION_RADIUS (0.15) = obstacle-free margin around the start
#   and the goal. It is not a robot radius either: it exists for NUMERICAL
#   STABILITY of the C engine, which computes the homotopy energy as
#       dist = dx^2 + dy^2 - radio^2   (lines 224-242 of HPPMDinamico3.c)
#   with respect to the start and the goal; if an obstacle ends up too close to
#   either of the two, that denominator tends to zero and the energy term blows
#   up. 1_generar_dataset.py used exactly 0.15, so the 100k real maps respect
#   that margin: it is also a property of the training distribution. The
#   geometric filter must use THIS value so as not to accept maps in a region
#   outside the training distribution.
# ==========================================================================
P_BASE = 0.0020  # base repulsion magnitude (mean range of the original generator)
MIN_OBS_GENERATOR = 5  # random.randint(5, 100) in 1_generar_dataset.py
ABSOLUTE_MAX_RADIUS = (REAL_MAX_DENSITY / (MIN_OBS_GENERATOR * math.pi)) ** 0.5  # ~0.149

# ==========================================================================
# Step 1: Geometric filter and collapse detection
# ==========================================================================
def geometric_filter(obs: np.ndarray, start=(0.0, 0.0), goal=(1.0, 1.0),
                     min_radius=REAL_MIN_RADIUS, max_radius=None,
                     robot_radius=None, start_goal_margin=None,
                     max_obstacles=100, L: float = 1.0) -> np.ndarray:
    """Filters and repairs obstacles so that they satisfy the original
    generator's constraints and do not crash the C engine.

    It directly targets the ~430 fallo_motor_c (21.7%) we had been seeing:
    - Clamps radii to the generator's ABSOLUTE range: [0.015, r_max(5)=0.149].
      It USED to clip to 0.050, which deformed ~39% of the REAL maps
      (r_max exceeds 0.05 with fewer than ~45 obstacles) and biased toward
      success. Radius-vs-density plausibility is judged by detect_collapse, not
      by this clamp.
    - Removes obstacles inside the safety zone of A and B
    - Removes obstacles outside the workspace [0,1]x[0,1]
    - Limits to MAX_OBSTACLES (100) so as not to overflow the C buffer
    """
    if len(obs) == 0:
        return obs
    # At L=1 (today's domain) these are ABSOLUTE_MAX_RADIUS /
    # GENERATOR_EXCLUSION_RADIUS / 0.02 unchanged (behaviour identical to the
    # original). At L!=1 they scale linearly (max_radius ~ sqrt(area) ~ L; the
    # margins, section 3.3).
    if max_radius is None:
        max_radius = ABSOLUTE_MAX_RADIUS * L
    if robot_radius is None:
        robot_radius = GENERATOR_EXCLUSION_RADIUS * L
    if start_goal_margin is None:
        start_goal_margin = START_GOAL_MARGIN_BASE * L

    filtered = []
    for i in range(len(obs)):
        x, y, r = obs[i]

        # Clamp position to the workspace. The generator uses
        # uniform(0,L) WITHOUT a margin; it used to clip to [0.01,0.99]*L, which
        # moved obstacles in ~77% of the real maps (at L=1).
        x = np.clip(x, 0.0, L)
        y = np.clip(y, 0.0, L)

        # Clamp radius to the realistic range of the original generator
        r = np.clip(r, min_radius, max_radius)

        # Discard if it lies inside the safety zone of the start or the goal
        dist_start = math.hypot(x - start[0], y - start[1])
        dist_goal = math.hypot(x - goal[0], y - goal[1])
        if dist_start <= (r + robot_radius + start_goal_margin):
            continue
        if dist_goal <= (r + robot_radius + start_goal_margin):
            continue

        filtered.append([x, y, r])

        # Do not exceed the C buffer limit
        if len(filtered) >= max_obstacles:
            break

    return np.array(filtered, dtype=np.float32) if filtered else np.zeros((0, 3), dtype=np.float32)

def mean_nearest_neighbor_distance(obs: np.ndarray) -> float:
    """Mean distance to the nearest neighbor, averaged over all the obstacles
    of the map. Same metric used to characterize the real dataset (minimum
    observed over 100k real maps: 0.0394 average per map)."""
    n = len(obs)
    if n < 2:
        return float("inf")
    d = np.linalg.norm(obs[:, None, :2] - obs[None, :, :2], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())

def bounding_box_extent(obs: np.ndarray) -> float:
    """Diagonal of the bounding box of the centres (x,y)."""
    return math.hypot(obs[:, 0].max() - obs[:, 0].min(), obs[:, 1].max() - obs[:, 1].min())


def theoretical_mean_radius(n: int) -> float:
    """Expected mean radius according to the real formula of the data
    generator (1_generar_dataset.py): the more obstacles, the smaller the
    radii, deterministically. It serves as a reference to detect implausible
    radii."""
    if n <= 0:
        return REAL_MIN_RADIUS + 0.005
    max_radius = math.sqrt(REAL_MAX_DENSITY / (n * math.pi))
    max_radius = max(max_radius, REAL_MIN_RADIUS + 0.005)
    return (REAL_MIN_RADIUS + max_radius) / 2

def detect_collapse(obs: np.ndarray, nn_threshold: float = 0.025, extent_threshold: float = 0.5,
                    radius_factor_threshold: float = 2.0, min_obs: int = 15) -> bool:
    """Detects whether the obstacles are degenerate -- THREE different and
    complementary failure modes; any one of them is enough:

    1. Almost duplicated points (abnormally small nearest-neighbor distance).
       Calibrated against the real minimum observed over 100k real maps
       (0.0394 average per map); the default 0.025 stays below it with margin
       so as not to penalize dense but legitimate overlap.

    2. The WHOLE map concentrated in a small region of the [0,1]x[0,1] square
       (small bounding box), even if the points INSIDE that region are not
       literally on top of one another. In the real dataset, even the map with
       the most overlapping obstacles occupies almost the whole square,
       regardless of density.

    3. Mean radius far above what corresponds to that number of obstacles,
       according to the real formula of the data generator
       (theoretical_mean_radius). Diagnosed empirically: the samples generated
       by the diffusion model can have radii 150%-900% larger than expected for
       their density -- a direct and measurable mechanism for the collapse
       (with such large circles, any group of 90+ obstacles is necessarily
       piled up).

    The three checks only apply with min_obs or more active obstacles -- a map
    with few naturally close/large obstacles is valid, not degenerate (large
    radii are NORMAL when there are few obstacles)."""
    if len(obs) < min_obs:
        return False
    if mean_nearest_neighbor_distance(obs) < nn_threshold:
        return True
    if bounding_box_extent(obs) < extent_threshold:
        return True
    mean_r = float(obs[:, 2].mean())
    if mean_r > radius_factor_threshold * theoretical_mean_radius(len(obs)):
        return True
    return False

# ==========================================================================
# Step 2: Compute repulsion parameters
# (exact replica of preprocesar_y_guardar from 2_evaluador_masivo.py)
# ==========================================================================
def compute_repulsion(obs: np.ndarray, start=(0.0, 0.0), goal=(1.0, 1.0),
                      sign_mode: str = "mass",
                      robot_radius: float = None, safety_margin: float = None) -> np.ndarray:
    """Assigns the repulsion sign by position with respect to the A->B
    diagonal, clusters with Union-Find, and homogenizes signs within each
    cluster. Returns an (N, 4) array with columns x, y, r, p_neto.

    robot_radius/safety_margin: by default they use the module constants
    (ROBOT_RADIUS=0.02, SAFETY_MARGIN=0.05, calibrated at L=1). In a scaled
    domain (L!=1) the scaled versions must be passed -- see
    domain_scaling.scaled_constants -- or the "cluster" criterion is evaluated
    with L=1 distances on a larger map and stops making physical sense."""
    robot_radius = ROBOT_RADIUS if robot_radius is None else robot_radius
    safety_margin = SAFETY_MARGIN if safety_margin is None else safety_margin
    n = len(obs)
    result = np.zeros((n, 4))
    result[:, :3] = obs

    # Sign by cross product with respect to the start->goal diagonal (vectorized)
    dx_route, dy_route = goal[0] - start[0], goal[1] - start[1]
    cross = dx_route * (obs[:, 1] - start[1]) - dy_route * (obs[:, 0] - start[0])
    result[:, 3] = P_BASE * np.where(cross > 0, 1.0, -1.0)

    # Union-Find to cluster nearby obstacles. Distances are computed
    # vectorized (an n x n matrix with numpy, not a Python loop) --
    # the Python loop only walks the PAIRS that actually qualify as
    # neighbors (typically O(n), not O(n^2): each obstacle only has a few
    # close neighbors), not the n*(n-1)/2 total pairs. From a few hundred
    # obstacles on (assembler mosaics, not single maps) this is several orders
    # of magnitude faster than the pure loop; the result is identical to the
    # previous one, verified against the old version.
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    neighbors = [0] * n
    if n > 1:
        xs, ys, rs = obs[:, 0], obs[:, 1], obs[:, 2]
        dist = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
        gap = dist - (rs[:, None] + rs[None, :])
        threshold = 2.0 * robot_radius + safety_margin
        i_idx, j_idx = np.where(np.triu(gap < threshold, k=1))
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            union(i, j)
            neighbors[i] += 1
            neighbors[j] += 1

    # Homogenize sign within each cluster
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    for root, members in clusters.items():
        if len(members) > 1:
            if sign_mode == "first":
                # LEGACY: sign of the lowest-index member -> the result
                # depends on the ORDER of the obstacles (43% of the maps change
                # some sign when permuted; 7-9% changes the HPPM state).
                dominant_sign = 1.0 if result[members[0], 3] >= 0 else -1.0
            else:
                # "mass": side of the diagonal where the sum of cross products
                # of the cluster lies. Invariant to permutations.
                s_cross = float(sum(cross[k] for k in members))
                dominant_sign = 1.0 if s_cross > 0 else -1.0
            for idx in members:
                result[idx, 3] = abs(result[idx, 3]) * dominant_sign

    # Reduce repulsion if it has many neighbors
    for i in range(n):
        if neighbors[i] >= 2:
            result[i, 3] /= 2.0

    return result

# ==========================================================================
# Step 3: Generate input files for HPPM
# ==========================================================================
def write_hppm_files(folder: str, obstacles_with_p: np.ndarray,
                     step_size: float = STEP_SIZE_BASE,
                     start=(0.0, 0.0), goal=(1.0, 1.0),
                     m1: float = DEFAULT_M1, m2: float = DEFAULT_M2):
    """Writes obstaculos.txt and pendientes.txt in the format the C code
    expects.

    m1/m2 ("homotopy slopes", see 1_generar_dataset.py): the original comment
    said the C code overwrites them internally with 1.0/4.0 and ignores them --
    PENDING CONFIRMATION against the real .c, there is empirical evidence
    (varying m1/m2 changes the convergence speed) that this may be outdated.
    Meanwhile they are exposed as real parameters instead of hardcoded;
    default 1.0/4.0 preserves the previous behaviour."""
    # obstaculos.txt: x y r p_neto (one line per obstacle)
    with open(os.path.join(folder, "obstaculos.txt"), "w") as f:
        for obs in obstacles_with_p:
            f.write(f"{obs[0]:.6f} {obs[1]:.6f} {obs[2]:.6f} {obs[3]:.4f}\n")

    # pendientes.txt: step, m1, m2, max_iter, start_x, start_y, goal_x, goal_y
    with open(os.path.join(folder, "pendientes.txt"), "w") as f:
        f.write(f"{step_size:.4f}\n")
        f.write(f"{m1:.4f}\n")
        f.write(f"{m2:.4f}\n")
        f.write("1000\n")  # max_iter (from the file -- different from the repair --max-iter)
        f.write(f"{start[0]:.6f}\n")
        f.write(f"{start[1]:.6f}\n")
        f.write(f"{goal[0]:.6f}\n")
        f.write(f"{goal[1]:.6f}\n")

# ==========================================================================
# Step 4: Run HPPM and evaluate the trajectory
# ==========================================================================
def evaluate_trajectory(tray_path: str, obstacles: np.ndarray,
                        goal=(1.0, 1.0), arrival_radius: float = ARRIVAL_RADIUS_BASE,
                        robot_radius: float = None) -> dict:
    """Analyses the trajectory computed by HPPM. robot_radius/arrival_radius:
    by default ROBOT_RADIUS (0.02) and 0.05, calibrated at L=1 -- in a scaled
    domain pass the scaled versions (arrival_radius typically scales like
    robot_radius, see domain_scaling)."""
    robot_radius = ROBOT_RADIUS if robot_radius is None else robot_radius
    x_tray, y_tray = [], []
    try:
        with open(tray_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    x_tray.append(float(parts[0]))
                    y_tray.append(float(parts[1]))
    except Exception:
        return {"state": "read_failure", "steps": 0, "length": 0.0,
                "llego_meta": False, "collisions": False}

    steps = len(x_tray)
    if steps == 0:
        return {"state": "empty_trajectory", "steps": 0, "length": 0.0,
                "llego_meta": False, "collisions": False}

    length = sum(math.hypot(x_tray[i] - x_tray[i - 1], y_tray[i] - y_tray[i - 1])
                 for i in range(1, steps))
    arrived = math.hypot(x_tray[-1] - goal[0], y_tray[-1] - goal[1]) <= arrival_radius

    # Collision: SEGMENT-to-obstacle distance along the WHOLE trajectory,
    # not only at the points where Newton-Raphson converged (tray.txt). With a
    # large step_size, two consecutive converged points can both be "clean" and
    # the straight stretch between them still cross an entire obstacle -- the
    # old point-to-point check does not detect it (see warning in the
    # chat, RESUMEN_BUSQUEDA_PASO.txt: empirically confirmed that with
    # step>=0.15 this always happens on a map with 1011 obstacles; with our
    # step_size=0.015 the risk is rarer but not zero, and it was not filtered).
    # Projection of the centre of each obstacle onto each segment tray[i]->
    # tray[i+1], clipped to [0,1] (perpendicular to the segment if it falls
    # inside, to the nearest endpoint if it falls outside).
    collision = False
    if len(obstacles):
        obs_arr = np.asarray(obstacles, dtype=np.float64)
        centers, radii = obs_arr[:, :2], obs_arr[:, 2]
        tray_arr = np.array(list(zip(x_tray, y_tray)))
        for i in range(steps):
            p0 = tray_arr[i]
            p1 = tray_arr[i + 1] if i + 1 < steps else p0
            d_vec = p1 - p0
            len2 = float(d_vec @ d_vec)
            if len2 < 1e-14:
                d_seg = np.hypot(centers[:, 0] - p0[0], centers[:, 1] - p0[1])
            else:
                t = np.clip(((centers[:, 0] - p0[0]) * d_vec[0] +
                             (centers[:, 1] - p0[1]) * d_vec[1]) / len2, 0.0, 1.0)
                proj = p0 + np.outer(t, d_vec)
                d_seg = np.hypot(centers[:, 0] - proj[:, 0], centers[:, 1] - proj[:, 1])
            if np.any(d_seg <= (radii + robot_radius)):
                collision = True
                break

    if collision:
        status = "collision"
    elif not arrived:
        status = "loop_stagnation"
    else:
        status = "success"

    return {"state": status, "steps": steps, "length": round(length, 4),
            "llego_meta": arrived, "collisions": collision}

def verify_one_map(args_tuple):
    """Worker for ProcessPoolExecutor: runs HPPM on one map."""
    idx, obs_with_p, hppm_exe, step_size = args_tuple[:4]
    timeout_sec = args_tuple[4] if len(args_tuple) > 4 else 2.0
    return_traj = args_tuple[5] if len(args_tuple) > 5 else False
    start = args_tuple[6] if len(args_tuple) > 6 else (0.0, 0.0)
    goal = args_tuple[7] if len(args_tuple) > 7 else (1.0, 1.0)
    robot_radius = args_tuple[8] if len(args_tuple) > 8 else None  # None = ROBOT_RADIUS (L=1)
    arrival_radius = args_tuple[9] if len(args_tuple) > 9 else ARRIVAL_RADIUS_BASE  # L=1 by default
    m1 = args_tuple[10] if len(args_tuple) > 10 else 1.0
    m2 = args_tuple[11] if len(args_tuple) > 11 else 4.0

    temp_dir = tempfile.mkdtemp(prefix=f"hppm_{idx:05d}_")
    try:
        # The executable is NOT copied to the tempdir -- it is invoked by absolute
        # path with cwd=temp_dir. The binary reads/writes pendientes.txt/
        # obstaculos.txt/tray.txt/newton_log.txt relative to the process cwd (which
        # subprocess.run already sets with cwd=temp_dir), so copying it first was
        # wasted I/O -- a shutil.copy + chmod per HPPM run, over hundreds of
        # thousands of runs. With --max-iter 0 (fast generation without repair) this
        # is even more noticeable: every candidate is 1 HPPM run in the common case
        # (it fails at the first port), so the per-run overhead weighs
        # proportionally more than when every candidate cost ~209 repair runs.
        exe_abs = os.path.abspath(hppm_exe)

        # Write files
        write_hppm_files(temp_dir, obs_with_p, step_size, start, goal, m1, m2)

        # Run HPPM
        subprocess.run(
            [exe_abs], cwd=temp_dir, capture_output=True, timeout=timeout_sec
        )

        # Evaluate
        tray_path = os.path.join(temp_dir, "tray.txt")
        res = evaluate_trajectory(tray_path, obs_with_p[:, :3], goal, arrival_radius, robot_radius)
        if return_traj:
            try:
                res["tray"] = np.loadtxt(tray_path, ndmin=2)[:, :2]
            except Exception:
                res["tray"] = np.zeros((0, 2))
        return idx, res

    except subprocess.TimeoutExpired:
        res = {"state": "timeout", "steps": 0, "length": 0.0,
               "llego_meta": False, "collisions": False}
        if return_traj:
            res["tray"] = np.zeros((0, 2))
        return idx, res
    except Exception as e:
        res = {"state": f"error: {e}", "steps": 0, "length": 0.0,
               "llego_meta": False, "collisions": False}
        if return_traj:
            res["tray"] = np.zeros((0, 2))
        return idx, res
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ==========================================================================
# Step 5: Report
# ==========================================================================
def build_report(results: list, requested_class: str, n_discarded: int,
                 n_empty: int = None, n_collapsed: int = None, n_few: int = None) -> dict:
    """Computes the verified success rate and complementary metrics."""
    total = len(results)
    if total == 0:
        # every map was discarded: valid report with rate 0
        return {"clase_solicitada": requested_class,
                "maps_generated_total": n_discarded,
                "mapas_descartados_degenerated": n_discarded,
                "mapas_descartados_vacios": n_empty,
                "mapas_descartados_colapsados": n_collapsed,
                "mapas_descartados_pocos": n_few,
                "mean_num_obs_evaluated": None, "mean_num_obs_success": None,
                "mapas_evaluados": 0, "verified_success_rate": 0.0,
                "success_rate_over_total_generated": 0.0, "count_by_status": {},
                "mean_length": None, "length_std": None, "mean_steps": None,
                "advertencia": "no map passed the filters"}

    counts = {}
    n_success = sum(1 for r in results if r["state"] == "success")
    lengths = []
    steps_list = []

    for r in results:
        st = r["state"]
        counts[st] = counts.get(st, 0) + 1
        if r["steps"] > 0:
            lengths.append(r["length"])
            steps_list.append(r["steps"])

    success_rate = n_success / total if total > 0 else 0.0

    report = {
        "clase_solicitada": requested_class,
        "maps_generated_total": total + n_discarded,
        "mapas_descartados_degenerated": n_discarded,
        "mapas_descartados_vacios": n_empty,
        "mapas_descartados_colapsados": n_collapsed,
        "mapas_descartados_pocos": n_few,
        "mean_num_obs_evaluated": round(float(np.mean([r["num_obs"] for r in results if "num_obs" in r])), 1) if results else None,
        "mean_num_obs_success": round(float(np.mean([r["num_obs"] for r in results if r["state"] == "success" and "num_obs" in r])), 1) if n_success else None,
        "mapas_evaluados": total,
        "verified_success_rate": round(success_rate, 4),
        "success_rate_over_total_generated": round(n_success / (total + n_discarded), 4) if (total + n_discarded) > 0 else None,
        "count_by_status": counts,
        "mean_length": round(np.mean(lengths), 4) if lengths else None,
        "length_std": round(np.std(lengths), 4) if lengths else None,
        "mean_steps": round(np.mean(steps_list), 1) if steps_list else None,
    }
    return report

# ==========================================================================
# Main
# ==========================================================================
