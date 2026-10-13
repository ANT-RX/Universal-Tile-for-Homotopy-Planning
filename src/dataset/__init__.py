"""Dataset generation, verification and consolidation for universal tiles.

A *universal* tile is a small map (``n`` in ``[1, 50]``, ``SIDE = 1.0`` fixed)
that solves the ``4 * P`` port-to-centre routes simultaneously, so it can be
composed into mosaics of any size.

Modules
-------
``baseline_generator``
    The original obstacle placement rule (exclusion margins, strict overlap
    margin, cluster size). Also the CLI for the pure-rejection baseline.
``repaired_generator``
    Plan B: the baseline generator plus local repair verified by HPPM
    (re-sample the culprit obstacle until all routes succeed).
``universal_generator``
    Filters during generation: a map is only accepted if all ``4 * P`` routes
    succeed on the *same* map. Repairs rather than rejecting.
``fast_generate``
    Generation without iterative repair: one candidate, pure geometric
    placement, one verification pass over the ``4 * P`` routes; discard on
    first failure. Priorises raw generation speed.
``verify_maps``
    HPPM invocations and trajectory evaluation: geometric filter, collapse
    detection, repulsion parameters, segment-to-obstacle collision check,
    report building.
``prepare_tiles``
    Consolidates generator batches into a single ``datos_tiles.npz``.
``clean_tiles``
    Re-verifies every tile against the current engine/collision check and
    writes ``tile-obstacle-maps.npz`` with only the tiles that still pass.
``check_engine``
    Re-verifies a sample of known-universal tiles against a given HPPM binary,
    to isolate "the model generates badly" from "the engine changed".

Depends on: ``src.common.*``.
Called by: ``src.model.set_diffusion`` (sampling path) and ``src.pipeline.*``.
"""
