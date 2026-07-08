"""
epibarrett -- epigenetic early detection of Barrett's esophagus / EAC from
DNA methylation, mirroring the non-endoscopic capsule-sponge + methylation
paradigm (Cyted EndoSign; EsoGuard VIM+CCNA1).

Public API::

    from epibarrett.pipeline import run_offline_demo
    from epibarrett.config import RunConfig
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import RunConfig  # noqa: E402,F401
