# Model metrics — tables for the paper


## 4. Generative quality

Real reference: held-out test split `test_holdout_dedup.npz` (NOT the training dataset).

| Metric | Value | 95% CI (Hanley-McNeil) | p-value (H0: AUC=0.5) | Interpretation |
|---|---|---|---|---|
| C2ST AUC | 0.5336 | [0.5097, 0.5576] | 0.0059 | 0.5 = indistinguishable from real; CI must contain 0.5 to claim it |
| C2ST AUC (without num_obs) | 0.54 | [0.5161, 0.5639] | 0.001 | same, without the controlled variable |
| Chamfer diversity (gen) | 0.14861 | - | - | vs. real: 0.15336 |
| Memorization (frac < real p5) | 0.0493 | - | - | high = copying the training set (p5 threshold 0.20644) |
