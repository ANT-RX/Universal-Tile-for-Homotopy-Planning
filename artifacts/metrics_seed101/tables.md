# Model metrics — tables for the paper


## 2. Compared cost (HPPM runs per useful tile)

| Method | HPPM runs / useful tile | acceptance rate | note |
|---|---|---|---|
| Local repair | 209.6 | 72.3% | inherited figure, not re-measured in this run |
| Pure rejection | 319.41 | 0.46% | measured in this run |
| **Diffusion + filter** | **26.62** | 24.51% | does not include training cost |

| Method | wall-clock s / successful tile (end to end) | worker-s / successful tile |
|---|---|---|
| Pure rejection | 0.0566 | 1.6976 |
| Diffusion + filter | 0.0172 | 0.1647 |

Diffusion+filter breakdown (s): sampling 7.973 | filter 1.158 | HPPM verification 4.246 (wall, 30 workers).

> The reported 'speedup' is the REDUCTION IN HPPM CALLS PER USEFUL TILE (cost.pure_rejection.hppm_runs_per_success / cost.diffusion_plus_filter.hppm_runs_per_success) under this experiment's conditions. It is NOT a speedup of the whole process: it excludes dataset generation and training, which are the four items above, in their own units, and are never summed here.

## 4. Generative quality

Real reference: held-out test split `test_holdout_dedup.npz` (NOT the training dataset).

| Metric | Value | 95% CI (Hanley-McNeil) | p-value (H0: AUC=0.5) | Interpretation |
|---|---|---|---|---|
| C2ST AUC | 0.5422 | [0.5185, 0.5659] | 0.0005 | 0.5 = indistinguishable from real; CI must contain 0.5 to claim it |
| C2ST AUC (without num_obs) | 0.5423 | [0.5186, 0.566] | 0.0005 | same, without the controlled variable |
| Chamfer diversity (gen) | 0.14782 | - | - | vs. real: 0.15132 |
| Memorization (frac < real p5) | 0.0573 | - | - | high = copying the training set (p5 threshold 0.2098) |

## 5. Complete mosaics

Verification: ONLY the corner-to-corner route (1 HPPM route), NOT the 12-route per-tile criterion. Wilson 95% CI; k/k is not a guarantee.

| Size | target obstacles | before repair | after repair [95% CI] | resolved 1st try / after repair | cell replacements (mean/max) | real obstacles | mean time |
|---|---|---|---|---|---|---|---|
| 3x3 | 135 | 5/30 (17% [7%, 34%]) | 30/30 100% [89%, 100%] | 5/25 | 2.27/6 | 134.9 | 4.7s |
| 5x5 | 375 | 5/30 (17% [7%, 34%]) | 30/30 100% [89%, 100%] | 5/25 | 3.5/10 | 374.9 | 9.1s |
| 8x8 | 900 | 0/30 (0% [0%, 11%]) | 30/30 100% [89%, 100%] | 0/30 | 6.97/17 | 899.8 | 20.0s |
