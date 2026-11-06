"""
src/train.py
------------
Module for training, evaluating, comparing, and persisting spam classification models.

Key ML Concepts Explained:
1. Stratified Train-Test Split (80/20):
   - In spam datasets, class distribution is naturally imbalanced (~86% ham, ~14% spam).
   - Stratification guarantees that both the training and test sets preserve the exact
     same proportion of classes, avoiding sample bias.
2. Multinomial Naive Bayes (MultinomialNB):
   - Computes posterior class probabilities using Bayes' Theorem: P(Class|Words) ~ P(Class) * Prod(P(Word|Class)).
   - Assumes feature independence ("naive"), which works surprisingly well for text classification.
   - Fast, resilient to overfitting with small datasets, and highly interpretable.
3. Logistic Regression:
   - Fits a linear decision boundary using the sigmoid (logistic) activation function.
   - Learns positive feature weights for spam-indicative words (e.g., "prize", "urgent", "claim")
     and negative weights for legitimate conversation tokens.
4. Evaluation Metrics & Confusion Matrix:
   - Accuracy: Overall correct predictions / total predictions. (Can be misleading with class imbalance).
   - Precision = TP / (TP + FP): When the model predicts SPAM, how often is it actually spam?
   - Recall = TP / (TP + FN): How many of the real spam messages did the model successfully catch?
   - F1-Score: Harmonic mean of Precision and Recall (2 * P * R / (P + R)).
   - Confusion Matrix: [TN, FP; FN, TP] tabular breakdown of prediction outcomes.

*** WHICH METRIC MATTERS MOST FOR SPAM DETECTION? ***
In spam filtering, PRECISION (for the Spam class) is the single most critical metric.
Reason:
- False Positive (FP): A legitimate, critical email (e.g., job offer, bank alert, medical update)
  is incorrectly flagged as SPAM and lost in the junk folder. This causes real-world disruption.
- False Negative (FN): A spam email reaches the inbox. While annoying, the user can easily delete it.
Therefore, a production spam filter must achieve near-perfect Precision (approaching 100%)
while keeping Recall as high as possible.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so modules can be run directly or imported as a package
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocess import load_and_inspect_data, preprocess_dataset
from src.features import extract_features

RANDOM_STATE = 42


def split_data(
    df: pd.DataFrame,
    text_column: str = "clean_message",
    target_column: str = "label",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Performs a stratified 80/20 train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.
    text_column : str
        Column containing cleaned text.
    target_column : str
        Column containing numeric target labels (0=ham, 1=spam).
    test_size : float, default=0.2
        Fraction of data reserved for test set.
    random_state : int, default=42
        Seed for deterministic reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : Tuple of pandas Series
    """
    print(f"\n[INFO] Performing stratified {int((1-test_size)*100)}/{int(test_size*100)} train-test split (random_state={random_state})...")
    X = df[text_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"  - Training samples: {len(X_train):,} (Ham: {(y_train == 0).sum():,}, Spam: {(y_train == 1).sum():,})")
    print(f"  - Testing samples:  {len(X_test):,} (Ham: {(y_test == 0).sum():,}, Spam: {(y_test == 1).sum():,})")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """
    Evaluates a trained classifier across key classification metrics and confusion matrix.

    Parameters
    ----------
    model : Trained scikit-learn classifier
    X_test : Transformed feature matrix
    y_test : True binary target labels
    model_name : str

    Returns
    -------
    Dict[str, Any]
        Dictionary storing all metric scores and the confusion matrix.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 60)
    print(f"EVALUATION RESULTS: {model_name}")
    print("-" * 60)
    print(f"  Accuracy:        {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision (Spam):{prec:.4f} ({prec * 100:.2f}%)  <-- [Crucial: avoids false alarms]")
    print(f"  Recall (Spam):   {rec:.4f} ({rec * 100:.2f}%)")
    print(f"  F1-Score (Spam): {f1:.4f} ({f1 * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(f"                  Predicted Ham (0)    Predicted Spam (1)")
    print(f"  Actual Ham (0)       {cm[0, 0]:<15}      {cm[0, 1]:<15} (False Positives)")
    print(f"  Actual Spam (1)      {cm[1, 0]:<15}      {cm[1, 1]:<15} (True Positives)")
    print("-" * 60)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }


def train_and_compare_models(
    X_train_vec,
    X_test_vec,
    y_train,
    y_test
) -> Tuple[Dict[str, Any], Any]:
    """
    Trains Multinomial Naive Bayes and Logistic Regression, prints a side-by-side
    comparison table, and returns the top-performing model.

    Parameters
    ----------
    X_train_vec : Feature matrix of training set
    X_test_vec : Feature matrix of test set
    y_train : Training target labels
    y_test : Test target labels

    Returns
    -------
    best_results : dict
        Metrics of the best model.
    best_model : scikit-learn classifier
        The best trained model object.
    """
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train_vec, y_train)
        trained_models[name] = model

        eval_res = evaluate_model(model, X_test_vec, y_test, name)
        results.append(eval_res)

    # Comparison summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model_name']:<28} | {r['accuracy']:<10.4f} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['f1_score']:<10.4f}")
    print("=" * 70)

    # Explanation of trade-off and decision criteria
    print("\n[NOTE ON METRICS SELECTION]")
    print("Spam classification prioritizes HIGH PRECISION to prevent legitimate messages")
    print("from being diverted to the spam folder. When Precision is tied or comparable,")
    print("F1-score determines the overall best balance between Precision and Recall.")

    # Select best model: highest precision first, broken by f1_score
    sorted_results = sorted(results, key=lambda x: (x["precision"], x["f1_score"]), reverse=True)
    best_res = sorted_results[0]
    best_name = best_res["model_name"]
    best_model = trained_models[best_name]

    print(f"\n>>> Selected Best Model: {best_name} (Precision: {best_res['precision']:.4f}, F1: {best_res['f1_score']:.4f})")
    return best_res, best_model


def save_artifacts(model, vectorizer, metadata: dict, output_dir: str = "models") -> None:
    """
    Persists the best trained model, vectorizer, and metadata to disk using joblib.

    Parameters
    ----------
    model : Trained classifier
    vectorizer : Fitted feature vectorizer
    metadata : dict
        Model evaluation scores and details.
    output_dir : str, default='models'
        Target directory to save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.joblib")
    vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[SUCCESS] Saved best model to:       {model_path}")
    print(f"[SUCCESS] Saved vectorizer to:       {vectorizer_path}")
    print(f"[SUCCESS] Saved model metadata to:   {metadata_path}")


def train_pipeline(data_path: str = "data/spam.csv", feature_method: str = "tfidf") -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Executes the entire training pipeline end to end:
    1. Loads dataset
    2. Cleans text
    3. Splits data (stratified 80/20)
    4. Extracts features without leakage
    5. Trains and compares MultinomialNB vs LogisticRegression
    6. Persists best model and vectorizer

    Returns
    -------
    best_model, vectorizer, best_results
    """
    df = load_and_inspect_data(data_path)
    df_clean = preprocess_dataset(df, text_column="message", use_stemming=True)

    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=RANDOM_STATE)

    X_train_vec, X_test_vec, vectorizer = extract_features(
        X_train,
        X_test,
        method=feature_method,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    best_results, best_model = train_and_compare_models(
        X_train_vec,
        X_test_vec,
        y_train,
        y_test
    )

    # Save artifacts into models/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    save_artifacts(best_model, vectorizer, best_results, output_dir=models_dir)

    return best_model, vectorizer, best_results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "spam.csv")
    train_pipeline(data_path=data_file, feature_method="tfidf")
