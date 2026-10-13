# Model metrics — tables for the paper


## 4. Generative quality

Real reference: held-out test split `test_holdout_dedup.npz` (NOT the training dataset).

| Metric | Value | 95% CI (Hanley-McNeil) | p-value (H0: AUC=0.5) | Interpretation |
|---|---|---|---|---|
| C2ST AUC | 0.537 | [0.5131, 0.5608] | 0.0024 | 0.5 = indistinguishable from real; CI must contain 0.5 to claim it |
| C2ST AUC (without num_obs) | 0.5464 | [0.5226, 0.5702] | 0.0001 | same, without the controlled variable |
| Chamfer diversity (gen) | 0.14496 | - | - | vs. real: 0.15075 |
| Memorization (frac < real p5) | 0.0561 | - | - | high = copying the training set (p5 threshold 0.20905) |
