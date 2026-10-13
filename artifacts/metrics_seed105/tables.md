# Model metrics — tables for the paper


## 4. Generative quality

Real reference: held-out test split `test_holdout_dedup.npz` (NOT the training dataset).

| Metric | Value | 95% CI (Hanley-McNeil) | p-value (H0: AUC=0.5) | Interpretation |
|---|---|---|---|---|
| C2ST AUC | 0.5469 | [0.5228, 0.571] | 0.0001 | 0.5 = indistinguishable from real; CI must contain 0.5 to claim it |
| C2ST AUC (without num_obs) | 0.5546 | [0.5306, 0.5787] | 0.0 | same, without the controlled variable |
| Chamfer diversity (gen) | 0.14546 | - | - | vs. real: 0.15137 |
| Memorization (frac < real p5) | 0.0473 | - | - | high = copying the training set (p5 threshold 0.20707) |
