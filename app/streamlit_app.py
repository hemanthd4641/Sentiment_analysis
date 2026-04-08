from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.predict import predict_sentiment

APP_TITLE = "Sentiment Analysis"
THREE_CLASS_DATASET = Path("data") / "raw" / "airline_sentiment_3class.csv"
DEFAULT_MODEL_PATH = Path("models") / "sentiment_model.joblib"

st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fafc 0%, #eef2ff 100%);
        color: #111827;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem;
        border-radius: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.22);
    }
    .card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 1rem;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
    }
    .result-positive {
        color: #15803d;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .result-negative {
        color: #b91c1c;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .result-neutral {
        color: #a16207;
        font-weight: 800;
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom: 0.35rem;">Sentiment Analysis Demo</h1>
        <p style="margin: 0; font-size: 1rem; opacity: 0.9;">
            Predict tweet sentiment with a TF-IDF + Naive Bayes model trained on airline_sentiment_3class.csv.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Model Status")
st.sidebar.write(f"Dataset: {THREE_CLASS_DATASET}")

if DEFAULT_MODEL_PATH.exists():
    st.sidebar.success("Loaded saved model")
else:
    st.sidebar.error("No saved model found. Train model with data/raw/airline_sentiment_3class.csv first.")

st.markdown('<div class="card"><strong>1) Enter Text</strong><br/>Type a sentence, review, or tweet and click Analyze Sentiment.</div>', unsafe_allow_html=True)

with st.form("sentiment_form", clear_on_submit=False):
    user_text = st.text_area(
        "Text Input",
        placeholder="Example: The service was quick and the staff were polite.",
        height=160,
    )
    analyze_clicked = st.form_submit_button("Analyze Sentiment")

if analyze_clicked:
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    elif not DEFAULT_MODEL_PATH.exists():
        st.warning("Model file missing. Run training first using data/raw/airline_sentiment_3class.csv.")
    else:
        result = predict_sentiment(user_text, DEFAULT_MODEL_PATH)
        label = str(result["label"])
        confidence = float(result["confidence"])
        palette = {
            "positive": "result-positive",
            "negative": "result-negative",
            "neutral": "result-neutral",
        }
        st.markdown(
            f"""
            <div class="card">
                <div><strong>2) Your Input</strong></div>
                <div style="margin: 0.45rem 0 0.9rem 0;">{user_text}</div>
                <div><strong>3) Predicted Sentiment</strong></div>
                <div class="{palette.get(label, 'result-neutral')}">{label.title()}</div>
                <div>Confidence Score: {confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Show class probabilities"):
            probabilities = result["probabilities"]
            for sentiment_label, probability in probabilities.items():
                st.write(f"{sentiment_label.title()}: {probability:.2%}")

st.markdown(
    """
    <div class="card">
        <strong>Example inputs</strong>
        <ul>
            <li>I love this product</li>
            <li>Worst experience ever</li>
            <li>It is okay</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
