from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils.data_preprocessing import load_dataset, save_cleaned_dataset

DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_PATH = Path("models") / "sentiment_model.joblib"
DEFAULT_METRICS_PATH = Path("models") / "metrics.json"
DEFAULT_CLEAN_DATA_PATH = Path("data") / "processed" / "cleaned_dataset.csv"


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Naive Bayes model pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=0.5),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> Dict[str, object]:
    """Evaluate the fitted model and return standard metrics."""
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["negative", "neutral", "positive"])

    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def train_model(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "sentiment",
    output_model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    cleaned_data_path: str | Path = DEFAULT_CLEAN_DATA_PATH,
    label_mapping: Optional[Dict[object, str]] = None,
) -> Dict[str, object]:
    """Train, evaluate, and persist a sentiment model."""
    dataset = load_dataset(csv_path, text_column=text_column, label_column=label_column, label_mapping=label_mapping)
    save_cleaned_dataset(dataset, cleaned_data_path)

    class_counts = dataset[label_column].value_counts()
    use_stratify = class_counts.min() >= 2 and len(dataset) >= 10

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[text_column],
        dataset[label_column],
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=dataset[label_column] if use_stratify else None,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    metrics = evaluate_model(pipeline, x_test, y_test)

    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "model_path": str(output_path),
        "metrics_path": str(metrics_file),
        "cleaned_data_path": str(Path(cleaned_data_path)),
        "metrics": metrics,
    }


def main() -> None:
    """Train the model from the 3-class dataset."""
    dataset_path = Path("data") / "raw" / "airline_sentiment_3class.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Required dataset not found: {dataset_path}")

    results = train_model(dataset_path)
    print("Training complete")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
