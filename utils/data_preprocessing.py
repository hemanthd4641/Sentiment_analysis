from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Clean a single text value for sentiment analysis."""
    if pd.isna(text):
        return ""

    normalized = str(text).lower().strip()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = NON_ALPHA_PATTERN.sub(" ", normalized)
    tokens = [token for token in normalized.split() if token not in ENGLISH_STOP_WORDS]
    return MULTISPACE_PATTERN.sub(" ", " ".join(tokens)).strip()


def normalize_label(label: object, label_mapping: Optional[Dict[object, str]] = None) -> str:
    """Normalize label values to positive, negative, or neutral."""
    if pd.isna(label):
        return ""

    if label_mapping is not None and label in label_mapping:
        return label_mapping[label].strip().lower()

    normalized = str(label).strip().lower()

    if normalized in {"positive", "pos", "p", "4", "1", "yes", "true"}:
        return "positive"
    if normalized in {"negative", "neg", "n", "0", "-1", "no", "false"}:
        return "negative"
    if normalized in {"neutral", "neu", "2", "3", "mixed", "none", "ok"}:
        return "neutral"

    return normalized


def load_dataset(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "sentiment",
    label_mapping: Optional[Dict[object, str]] = None,
) -> pd.DataFrame:
    """Load and preprocess a sentiment dataset from a CSV file."""
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    required_columns = {text_column, label_column}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    processed = frame[[text_column, label_column]].copy()
    processed[text_column] = processed[text_column].fillna("").astype(str).map(clean_text)
    processed[label_column] = processed[label_column].map(lambda value: normalize_label(value, label_mapping))
    processed = processed.replace({"": pd.NA}).dropna(subset=[text_column, label_column])
    processed = processed[processed[text_column].str.len() > 0]
    processed = processed[processed[label_column].isin({"positive", "negative", "neutral"})]
    processed = processed.reset_index(drop=True)
    return processed


def save_cleaned_dataset(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Save a cleaned dataset for reproducible experiments."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
