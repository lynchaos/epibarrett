"""Helpers for loading epibarrett artifacts from the Hugging Face Hub."""

from __future__ import annotations

from typing import Any

import joblib
from sklearn.pipeline import Pipeline

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "load_model requires huggingface-hub. "
        "Install with `pip install epibarrett[hf]` or `pip install huggingface-hub`."
    ) from exc

DEFAULT_REPO_ID = "kmlyyll/epibarrett-model"
BUNDLE_FILENAME = "epibarrett_model.joblib"


def load_model(repo_id: str = DEFAULT_REPO_ID, **kwargs: Any) -> Pipeline:
    """Load the epibarrett model bundle from the Hugging Face Hub.

    Parameters
    ----------
    repo_id : Hugging Face model repository id (default: ``kmlyyll/epibarrett-model``).
    **kwargs : forwarded to :func:`huggingface_hub.hf_hub_download`.

    Returns
    -------
    The ``lasso`` genome-wide L1 panel pipeline by default. Use
    ``load_bundle`` if you need the targeted or multimodal models, or the
    preprocessor.
    """
    bundle = load_bundle(repo_id=repo_id, **kwargs)
    return bundle["lasso"]


def load_bundle(repo_id: str = DEFAULT_REPO_ID, **kwargs: Any) -> dict[str, Any]:
    """Load the full epibarrett artifact bundle from the Hugging Face Hub.

    Returns a dictionary with keys:
        - ``lasso``: genome-wide L1 logistic panel
        - ``targeted``: VIM+CCNA1 two-gene panel
        - ``multimodal``: methylation score + clinical features model
        - ``preprocessor``: fitted beta->M preprocessor
        - ``probe_names``: list of probe columns expected at inference
        - ``clinical_features``: list of clinical column names
    """
    cache_dir = kwargs.pop("cache_dir", None)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=BUNDLE_FILENAME,
        cache_dir=cache_dir,
        **kwargs,
    )
    return joblib.load(local_path)


def load_preprocessor(repo_id: str = DEFAULT_REPO_ID, **kwargs: Any) -> Any:
    """Load only the fitted preprocessor from the Hugging Face Hub."""
    return load_bundle(repo_id=repo_id, **kwargs)["preprocessor"]
