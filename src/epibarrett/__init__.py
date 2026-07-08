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

# Optional Hugging Face Hub helpers are exposed lazily so the package remains
# importable when huggingface-hub is not installed.
try:
    from .hf_hub import load_bundle, load_model, load_preprocessor  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency path
    pass
