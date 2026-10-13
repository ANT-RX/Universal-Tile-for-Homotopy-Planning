"""The diffusion model and its cost benchmark.

Modules
-------
``set_diffusion``
    Core architecture. ``SetDenoiser`` (a transformer without positional
    encoding, adaLN-Zero conditioning on ``t``, ``n`` and the class) plus the
    cosine ``alphas_cumprod`` schedule, the diffusion loss, the explicit
    training loop and multi-port DDIM sampling with HPPM verification.
    Direct diffusion over ``(x, y, r)`` — no autoencoder, no route
    conditioning.
``gpu_cost``
    Diagnostic: measures seconds per step and peak memory of the denoiser for
    several ``n`` and batch sizes, without needing data or checkpoints.

Depends on: ``src.common.domain_scaling``. In its sampling path it also
imports ``src.common.maps_io``, ``src.dataset.verify_maps`` and
``src.dataset.universal_generator``.

Called by: ``src.pipeline.assembler``, ``src.dataset.fast_generate``,
``src.pipeline.mosaic_no_hppm``, ``src.evaluation.*``.
"""
