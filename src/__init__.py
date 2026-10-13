"""Universal Tiles — composable obstacle-set diffusion for HPPM path planning.

Canonical source tree. Run every entry point as a module from this directory
(the parent of ``src/``) so that the ``artifacts/...`` relative paths used as
CLI defaults resolve correctly::

    cd Universal_Tiles
    python -m src.model.set_diffusion train --data artifacts/tile-obstacle-maps.npz
    python -m src.evaluation.paper_metrics --help

Layout
------
``src/common``      shared primitives: domain scaling constants and maps I/O.
``src/model``       the diffusion model itself plus its cost benchmark.
``src/dataset``     dataset generation, verification against the C HPPM engine,
                    cleaning and consolidation.
``src/pipeline``    mosaic assembly and the auxiliary generation tools.
``src/evaluation``  paper-oriented evaluation (metrics, tables, visual review).
``engine``          the HPPM C source that is compiled to the runner binary.
``docs``            the consolidated phase report.
``artifacts``       data, checkpoints, logs, results and the compiled engine
                    binary. NOT tracked by git.

Pipeline order
--------------
``src/dataset/baseline_generator`` -> ``src/dataset/universal_generator``
(repair via ``src/dataset/repaired_generator``) -> ``src/dataset/verify_maps``
(HPPM) -> ``src/dataset/prepare_tiles`` / ``clean_tiles`` -> ``tiles .npz`` ->
``src/model/set_diffusion`` (train) -> sampling -> verification ->
``src/pipeline/assembler`` (mosaics) -> ``src/evaluation/paper_metrics``.
"""
