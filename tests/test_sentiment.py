from __future__ import annotations

from pathlib import Path

from utils.data_preprocessing import clean_text
from utils.predict import predict_sentiment
from utils.train_model import train_model


def test_clean_text_removes_noise() -> None:
    cleaned = clean_text("Check https://example.com NOW!!! This is AMAZING!!!")
    assert "http" not in cleaned
    assert "amazing" in cleaned
    assert cleaned == cleaned.lower()


def test_training_and_prediction_pipeline(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "text,sentiment\n"
        "I love this product,positive\n"
        "Amazing quality and great support,positive\n"
        "Fantastic experience and excellent service,positive\n"
        "Very happy with this purchase,positive\n"
        "Worst experience ever,negative\n"
        "Terrible service and bad attitude,negative\n"
        "I hate this and it is awful,negative\n"
        "Completely disappointing and frustrating,negative\n"
        "It is okay,neutral\n"
        "Average but acceptable,neutral\n"
        "Neither good nor bad just fine,neutral\n"
        "The product is standard and ordinary,neutral\n",
        encoding="utf-8",
    )

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    cleaned_path = tmp_path / "cleaned.csv"

    results = train_model(
        dataset_path,
        output_model_path=model_path,
        metrics_path=metrics_path,
        cleaned_data_path=cleaned_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert cleaned_path.exists()
    assert results["metrics"]["accuracy"] >= 0.0

    positive = predict_sentiment("I love this product", model_path)
    negative = predict_sentiment("Worst experience ever", model_path)
    neutral = predict_sentiment("It is okay", model_path)

    assert positive["label"] == "positive"
    assert negative["label"] == "negative"
    assert neutral["label"] == "neutral"
