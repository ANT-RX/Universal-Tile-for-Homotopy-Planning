"""Shared primitives used across the whole pipeline.

Modules
-------
``domain_scaling``
    Domain scaling ``[0, L] x [0, L]`` and the physical constants that scale
    linearly with ``L``. Single source of truth for ``MAX_DENSITY``,
    ``MIN_RADIUS`` and the four route families. Imports nothing local.
``maps_io``
    Compact maps I/O: one ``maps.npz`` per run instead of a ``mapa_*/``
    directory tree, plus a reader for the legacy layout. Imports nothing local.

Called by: ``src.model.set_diffusion``, ``src.dataset.*`` and
``src.pipeline.*``.
"""
