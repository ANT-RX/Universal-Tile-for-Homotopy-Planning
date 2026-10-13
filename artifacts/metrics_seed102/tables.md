# Model metrics — tables for the paper


## 4. Generative quality

Real reference: held-out test split `test_holdout_dedup.npz` (NOT the training dataset).

| Metric | Value | 95% CI (Hanley-McNeil) | p-value (H0: AUC=0.5) | Interpretation |
|---|---|---|---|---|
| C2ST AUC | 0.5348 | [0.5107, 0.5589] | 0.0047 | 0.5 = indistinguishable from real; CI must contain 0.5 to claim it |
| C2ST AUC (without num_obs) | 0.5353 | [0.5112, 0.5594] | 0.0041 | same, without the controlled variable |
| Chamfer diversity (gen) | 0.14705 | - | - | vs. real: 0.15138 |
| Memorization (frac < real p5) | 0.0567 | - | - | high = copying the training set (p5 threshold 0.21104) |
