"""
src/predict.py
--------------
Module for real-time inference: predicts whether a new raw email or SMS
message is SPAM or NOT SPAM, along with a probability-based confidence score.

Key ML Concepts Explained:
1. Inference Pipeline Consistency:
   - Real-world raw text MUST pass through the exact same preprocessing pipeline
     (lowercasing, punctuation removal, stopword filtering, stemming) as the training set.
   - The text must be transformed using the ALREADY FITTED vectorizer (`transform`, NOT `fit_transform`).
2. Confidence Score & Probability Calibration:
   - For classification, we don't just want a hard label (0 or 1); we want the predicted
     probability P(Spam | Text) via `predict_proba`.
   - A message with 99.8% spam probability indicates high certainty, while a message near 51%
     suggests borderline ambiguity.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
from src.preprocess import clean_text

# Cached references for loaded model and vectorizer
_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def load_artifacts(models_dir: str = None) -> Tuple[Any, Any]:
    """
    Loads the serialized model and vectorizer from disk.
    Caches instances in memory to avoid repeated disk reads.

    Parameters
    ----------
    models_dir : str, optional
        Path to directory containing saved .joblib files.

    Returns
    -------
    model, vectorizer
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER

    if _CACHED_MODEL is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_MODEL, _CACHED_VECTORIZER

    if models_dir is None:
        models_dir = os.path.join(ROOT_DIR, "models")

    model_path = os.path.join(models_dir, "best_model.joblib")
    vectorizer_path = os.path.join(models_dir, "vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Trained artifacts not found in '{models_dir}'. "
            "Please train the model first by running `python src/train.py` or `python main.py`."
        )

    _CACHED_MODEL = joblib.load(model_path)
    _CACHED_VECTORIZER = joblib.load(vectorizer_path)
    return _CACHED_MODEL, _CACHED_VECTORIZER


def predict_email(
    text: str,
    model: Any = None,
    vectorizer: Any = None,
    use_stemming: bool = True
) -> Dict[str, Any]:
    """
    Predicts whether a given email or SMS text is SPAM or NOT SPAM.

    Parameters
    ----------
    text : str
        The raw email or SMS string to classify.
    model : optional
        Pre-loaded classifier. If None, loads from disk.
    vectorizer : optional
        Pre-loaded vectorizer. If None, loads from disk.
    use_stemming : bool, default=True
        Whether to apply stemming during preprocessing.

    Returns
    -------
    dict
        {
            "prediction": "SPAM" or "NOT SPAM",
            "confidence": float (percentage, e.g. 99.45),
            "probabilities": {
                "ham": float (0.0 to 1.0),
                "spam": float (0.0 to 1.0)
            },
            "cleaned_text": str
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "prediction": "NOT SPAM",
            "confidence": 100.0,
            "probabilities": {"ham": 1.0, "spam": 0.0},
            "cleaned_text": ""
        }

    # Load model and vectorizer if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    # Step 1: Preprocess raw text using identical pipeline
    cleaned = clean_text(text, use_stemming=use_stemming)

    # Step 2: Transform text into vector using fitted vocabulary
    features = vectorizer.transform([cleaned])

    # Step 3: Compute class probabilities and predicted label
    probabilities = model.predict_proba(features)[0]
    ham_prob = float(probabilities[0])
    spam_prob = float(probabilities[1])

    # Class 1 = SPAM, Class 0 = HAM (NOT SPAM)
    pred_label = int(model.predict(features)[0])
    prediction_str = "SPAM" if pred_label == 1 else "NOT SPAM"
    confidence = (spam_prob if pred_label == 1 else ham_prob) * 100.0

    return {
        "prediction": prediction_str,
        "confidence": round(confidence, 2),
        "probabilities": {
            "ham": round(ham_prob, 4),
            "spam": round(spam_prob, 4)
        },
        "cleaned_text": cleaned
    }


def run_sample_predictions():
    """
    Tests the prediction pipeline on sample emails/SMS messages covering
    clear spam, clear legitimate, and ambiguous scenarios.
    """
    sample_emails = [
        # Sample 1: Clear obvious spam (lottery prize)
        "CONGRATULATIONS! You have been selected to receive a FREE £1,000 Walmart Gift Card! Claim your reward now by texting CLAIM to 87121 or visiting http://free-gift.fake",

        # Sample 2: Clear legitimate email (work/meeting communication)
        "Hi Alex, please find attached the revised presentation slides and financial figures for our team sync meeting tomorrow at 10 AM. Let me know your thoughts.",

        # Sample 3: Casual personal message (legitimate)
        "Hey, are we still meeting up for coffee this afternoon? Let me know what time works best for you.",

        # Sample 4: Urgent phishing scam
        "URGENT ALERT: Your bank account has been temporarily locked due to suspicious login attempts. Verify your credentials immediately at http://secure-bank-login.com to avoid account termination."
    ]

    print("=" * 75)
    print("RUNNING SAMPLE EMAIL PREDICTIONS")
    print("=" * 75)

    model, vectorizer = load_artifacts()

    for idx, email in enumerate(sample_emails, start=1):
        result = predict_email(email, model=model, vectorizer=vectorizer)

        tag = "[SPAM]" if result["prediction"] == "SPAM" else "[NOT SPAM]"
        print(f"\nSample #{idx}:")
        print(f"  Raw Text:       \"{email[:80]}...\"")
        print(f"  Preprocessed:   \"{result['cleaned_text']}\"")
        print(f"  Result:         {tag} {result['prediction']}")
        print(f"  Confidence:     {result['confidence']}%")
        print(f"  Probabilities:  Ham: {result['probabilities']['ham']:.4f} | Spam: {result['probabilities']['spam']:.4f}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict whether an email or SMS message is spam.")
    parser.add_argument("--text", type=str, help="Text string of the message to classify.")
    args = parser.parse_args()

    if args.text:
        res = predict_email(args.text)
        print(f"\nMessage: \"{args.text}\"")
        print(f"Prediction: {res['prediction']} (Confidence: {res['confidence']}%)")
        print(f"Probabilities: {res['probabilities']}")
    else:
        # Run standard battery of tests
        run_sample_predictions()

