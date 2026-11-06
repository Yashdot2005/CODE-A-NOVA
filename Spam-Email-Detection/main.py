"""
main.py
-------
Code-A-Nova Machine Learning Internship
Project: Level 1(b) - Email & SMS Spam Detection
------------------------------------------------
Orchestrates the entire end-to-end NLP & Machine Learning pipeline:
1. Verifies and prepares the dataset (data/spam.csv)
2. Loads, inspects, and cleans text data (src/preprocess.py)
3. Extracts numeric features (TF-IDF & Count) without data leakage (src/features.py)
4. Trains and evaluates Multinomial Naive Bayes vs. Logistic Regression (src/train.py)
5. Explains why Precision is paramount for spam filtering
6. Persists the winning model and vectorizer (models/)
7. Executes real-time sample predictions with confidence scores (src/predict.py)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.download_data import download_dataset
from src.preprocess import load_and_inspect_data, preprocess_dataset
from src.features import extract_features
from src.train import split_data, train_and_compare_models, save_artifacts, RANDOM_STATE
from src.predict import load_artifacts, predict_email, run_sample_predictions


def print_banner():
    banner = """
    ======================================================================
                         CODE-A-NOVA ML INTERNSHIP
                LEVEL 1(b): EMAIL & SMS SPAM DETECTION
    ======================================================================
    Pipeline:
      1. Data Ingestion & Inspection
      2. NLP Preprocessing (Tokenize, Filter, Stem)
      3. Feature Extraction (TF-IDF Vectorization)
      4. Model Training & Comparison (Naive Bayes vs Logistic Regression)
      5. Metrics Evaluation & Metric Priority Analysis
      6. Model Artifact Persistence
      7. Sample Inference & Confidence Scoring
    ======================================================================
    """
    print(banner)


def main():
    print_banner()

    # Paths configuration
    data_dir = os.path.join(ROOT_DIR, "data")
    dataset_path = os.path.join(data_dir, "spam.csv")
    models_dir = os.path.join(ROOT_DIR, "models")

    # Step 1: Ensure dataset is available
    if not os.path.exists(dataset_path):
        print("[STEP 1/7] Dataset not found locally. Initiating download...")
        download_dataset()
    else:
        print(f"[STEP 1/7] Found dataset at {dataset_path}")

    # Step 2: Load and inspect dataset
    print("\n[STEP 2/7] Loading and inspecting dataset...")
    df = load_and_inspect_data(dataset_path)

    # Step 3: Text Preprocessing
    print("\n[STEP 3/7] Preprocessing text (Lowercasing, Cleaning, Stopwords, Stemming)...")
    df_clean = preprocess_dataset(df, text_column="message", use_stemming=True)

    # Step 4: Stratified Train-Test Split (80/20)
    print("\n[STEP 4/7] Splitting data into stratified Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        text_column="clean_message",
        target_column="label",
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Step 5: Feature Extraction (TF-IDF)
    print("\n[STEP 5/7] Extracting TF-IDF Features (Fitting ONLY on training set)...")
    X_train_vec, X_test_vec, vectorizer = extract_features(
        X_train,
        X_test,
        method="tfidf",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    # Step 6: Train and Compare Models
    print("\n[STEP 6/7] Training & Evaluating Classification Models...")
    best_results, best_model = train_and_compare_models(
        X_train_vec,
        X_test_vec,
        y_train,
        y_test
    )

    # Save artifacts
    save_artifacts(best_model, vectorizer, best_results, output_dir=models_dir)

    # Step 7: Sample Real-Time Predictions
    print("\n[STEP 7/7] Demonstrating Inference on Sample Messages...")
    run_sample_predictions()

    print("\n" + "=" * 70)
    print("[SUCCESS] Full pipeline completed successfully!")
    print(f"Top model saved in: {models_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

