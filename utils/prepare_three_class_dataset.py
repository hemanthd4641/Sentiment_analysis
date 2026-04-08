from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE_PATH = Path("data") / "raw" / "airline_sentiment.csv"
DEFAULT_OUTPUT_PATH = Path("data") / "raw" / "airline_sentiment_3class.csv"
DEFAULT_RANDOM_STATE = 42

NEUTRAL_SUBJECTS = [
    "The product",
    "The service",
    "The delivery",
    "The app",
    "The update",
    "The meeting",
    "The movie",
    "The meal",
    "The flight",
    "The experience",
]

NEUTRAL_STATES = [
    "was average",
    "was normal",
    "was standard",
    "was okay",
    "met expectations",
    "was neither good nor bad",
    "was acceptable",
    "felt routine",
    "was ordinary",
    "was fine overall",
]

NEUTRAL_ENDINGS = [
    "for this price.",
    "for today.",
    "compared to last time.",
    "without any major issues.",
    "with no strong opinion.",
    "in general.",
    "for most users.",
    "for a regular day.",
    "for this category.",
    "for the given context.",
]


def _generate_neutral_samples(target_count: int) -> pd.DataFrame:
    """Generate deterministic neutral texts when neutral labels are unavailable."""
    rows = []
    for subject in NEUTRAL_SUBJECTS:
        for state in NEUTRAL_STATES:
            for ending in NEUTRAL_ENDINGS:
                rows.append(f"{subject} {state} {ending}")

    base = pd.DataFrame({"text": rows, "sentiment": "neutral"})
    if target_count <= len(base):
        return base.sample(n=target_count, random_state=DEFAULT_RANDOM_STATE).reset_index(drop=True)

    repeats = int(np.ceil(target_count / len(base)))
    expanded = pd.concat([base] * repeats, ignore_index=True)
    expanded = expanded.sample(frac=1.0, random_state=DEFAULT_RANDOM_STATE).head(target_count)
    return expanded.reset_index(drop=True)


def prepare_three_class_dataset(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    per_class_count: int = 120000,
) -> Path:
    """Create a balanced 3-class dataset from normalized sentiment data."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    frame = pd.read_csv(source)
    if not {"text", "sentiment"}.issubset(frame.columns):
        raise ValueError("Source dataset must contain 'text' and 'sentiment' columns.")

    positives = frame[frame["sentiment"] == "positive"][ ["text", "sentiment"] ]
    negatives = frame[frame["sentiment"] == "negative"][ ["text", "sentiment"] ]
    neutrals = frame[frame["sentiment"] == "neutral"][ ["text", "sentiment"] ]

    if positives.empty or negatives.empty:
        raise ValueError("Source dataset must include both positive and negative classes.")

    sample_size = min(per_class_count, len(positives), len(negatives))
    positive_sample = positives.sample(n=sample_size, random_state=DEFAULT_RANDOM_STATE)
    negative_sample = negatives.sample(n=sample_size, random_state=DEFAULT_RANDOM_STATE)
    if len(neutrals) >= sample_size:
        neutral_sample = neutrals.sample(n=sample_size, random_state=DEFAULT_RANDOM_STATE)
    else:
        generated_count = sample_size - len(neutrals)
        synthetic_neutral = _generate_neutral_samples(generated_count)
        neutral_sample = pd.concat([neutrals, synthetic_neutral], ignore_index=True)

    combined = pd.concat([positive_sample, negative_sample, neutral_sample], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=DEFAULT_RANDOM_STATE).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


def main() -> None:
    output = prepare_three_class_dataset()
    frame = pd.read_csv(output)
    print(f"Saved 3-class dataset to: {output}")
    print(frame["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
