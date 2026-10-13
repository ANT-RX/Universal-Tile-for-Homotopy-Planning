"""Paper-oriented evaluation of the trained model.

Modules
-------
``paper_metrics``
    Runs the whole evaluation suite for the paper into a single output folder:
    success rate per ``n`` bin with Wilson confidence intervals, HPPM cost per
    useful tile across the three generation methods, DDIM-step ablation,
    diversity/realism (C2ST, Chamfer, memorisation) and full-mosaic success
    rate. Each test can be run alone with ``--solo``.
``final_evaluation``
    Consolidates the final comparison (model vs baseline, broken down by
    ``num_obs``) into a paper-ready markdown table with 95% Wilson binomial
    confidence intervals.
``visual_review``
    Human-in-the-loop inspection: a contact sheet of N random maps of a given
    status, and an optional re-verification pass over the same folder. Reads
    ``L`` / start / goal per map from ``results_detail.json``.

Depends on: ``src.common.*``, ``src.dataset.*`` and ``src.model.*``.
These scripts are read-only with respect to the artifacts: they never modify
the dataset or the checkpoints.
"""
