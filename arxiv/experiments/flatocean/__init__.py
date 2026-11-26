"""flatocean: shared infrastructure and analyses for the flat-ocean experiments.

Layout (dependency direction: drivers `exp_*.py` -> `flatocean.analyses` -> the infrastructure modules below; no module
here imports a driver):

    runs        run directories, checkpoint loading, training stages, matched stages
    linalg      ranks with an explicit reference scale, spans, assembled QR, SVD with the 64-bit LAPACK fallback
    blocks      the Jacobian block of a layer (MLP and tied CNN) behind one interface, per-unit column groups
    factors     stored spectral factors: paths, identity manifests, authenticated loading
    records     result files: identity from every consumed input, completeness per measurement, atomic saves
    directions  probes, scaling protocols, the ray evaluation of the transition residual
    cli         common command-line options and the explicit process configuration (threads, device)
    analyses    the scientific calculations (neuron-wise decomposition, CNN channels, ...)

Importing any module here has no process-wide side effect: threads, devices and seeds are set by the driver through
`flatocean.cli.configure`.
"""
__version__ = "0.1 (2026-09-24)"
