"""Gradio demo for epibarrett BE/EAC methylation classifier."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "kmlyyll/epibarrett-model"
BUNDLE_PATH = Path("epibarrett_model.joblib")


def _load_bundle():
    if not BUNDLE_PATH.exists():
        hf_hub_download(REPO_ID, filename="epibarrett_model.joblib", local_dir=".")
    return joblib.load(BUNDLE_PATH)


BUNDLE = _load_bundle()
LASSO = BUNDLE["lasso"]
PREPROCESSOR = BUNDLE["preprocessor"]
PROBE_NAMES = BUNDLE["probe_names"]
CLINICAL_FEATURES = BUNDLE["clinical_features"]


def predict(csv_file):
    X = pd.read_csv(csv_file.name, index_col=0)
    missing = [p for p in PROBE_NAMES if p not in X.columns]
    if missing:
        raise gr.Error(
            f"Missing {len(missing)} expected probe columns (e.g. {missing[:5]})."
        )
    M = PREPROCESSOR.transform(X[PROBE_NAMES])
    proba = LASSO.predict_proba(M.to_numpy())[:, 1]
    out = pd.DataFrame(
        {"sample_id": X.index, "BE_EAC_probability": proba, "risk_call": (proba >= 0.5).astype(int)}
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    out.to_csv(tmp.name, index=False)
    return out, tmp.name


with gr.Blocks(title="epibarrett BE/EAC classifier") as demo:
    gr.Markdown(
        """
        # epibarrett demo
        Upload a CSV of HM450-style beta values (rows = samples, columns = CpG probes).
        The model returns a calibrated probability of Barrett's esophagus / EAC for each sample.
        """
    )
    file_in = gr.File(label="Upload beta-value CSV", file_types=[".csv"])
    btn = gr.Button("Predict")
    table_out = gr.Dataframe(label="Predictions")
    file_out = gr.File(label="Download predictions CSV")
    btn.click(fn=predict, inputs=file_in, outputs=[table_out, file_out])

if __name__ == "__main__":
    demo.launch()
