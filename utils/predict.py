from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib

DEFAULT_MODEL_PATH = Path("models") / "sentiment_model.joblib"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the persisted sentiment model pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model first with utils/train_model.py."
        )
    return joblib.load(path)


def predict_sentiment(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict[str, object]:
    """Predict sentiment and return the label with confidence."""
    model = load_model(model_path)
    probabilities = model.predict_proba([text])[0]
    labels = list(model.classes_)
    best_index = int(probabilities.argmax())
    label = labels[best_index]
    confidence = float(probabilities[best_index])
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {cls: float(prob) for cls, prob in zip(labels, probabilities)},
    }


def predict_sentiment_batch(texts: list[str], model_path: str | Path = DEFAULT_MODEL_PATH) -> list[Dict[str, object]]:
    """Predict sentiment for multiple text inputs."""
    model = load_model(model_path)
    probabilities = model.predict_proba(texts)
    labels = list(model.classes_)

    results = []
    for row in probabilities:
        best_index = int(row.argmax())
        results.append(
            {
                "label": labels[best_index],
                "confidence": float(row[best_index]),
                "probabilities": {cls: float(prob) for cls, prob in zip(labels, row)},
            }
        )
    return results
