"""Mosaic assembly and auxiliary generation/analysis tools.

Modules
-------
``assembler``
    Builds mosaics of any size by requesting tiles **live** from the trained
    checkpoint (never from a pre-generated catalogue), reusing the filtering
    and verification helpers of ``src.model.set_diffusion``.
``mosaic_no_hppm``
    Visual exploration only: applies the geometric filters but **never runs
    HPPM**, so the resulting mosaic has no navigability guarantee. Not valid
    for evaluation or training data.
``server_diagnostics``
    Environment check before a long run: real core count, CUDA availability
    and free VRAM, disk space and whether gcc is present to build the HPPM
    engine.

Depends on: ``src.model.*``, ``src.dataset.*``, ``src.common.*``.
"""
