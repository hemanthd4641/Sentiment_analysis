from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DEFAULT_PATH = Path("data") / "raw" / "training.1600000.processed.noemoticon.csv"
OUTPUT_DEFAULT_PATH = Path("data") / "raw" / "airline_sentiment.csv"

COLUMN_NAMES = ["target", "ids", "date", "flag", "user", "text"]
LABEL_MAP = {0: "negative", 2: "neutral", 4: "positive"}


def prepare_sentiment140(
    raw_path: str | Path = RAW_DEFAULT_PATH,
    output_path: str | Path = OUTPUT_DEFAULT_PATH,
) -> Path:
    """Convert raw Sentiment140 CSV into a normalized text/sentiment CSV."""
    source = Path(raw_path)
    if not source.exists():
        raise FileNotFoundError(f"Sentiment140 file not found: {source}")

    frame = pd.read_csv(source, header=None, names=COLUMN_NAMES, encoding="latin-1")
    frame = frame[["text", "target"]].copy()
    frame["sentiment"] = frame["target"].map(LABEL_MAP)
    frame = frame.dropna(subset=["text", "sentiment"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[["text", "sentiment"]].to_csv(output, index=False)
    return output


def main() -> None:
    output = prepare_sentiment140()
    frame = pd.read_csv(output)
    print(f"Saved normalized dataset to: {output}")
    print(frame["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
