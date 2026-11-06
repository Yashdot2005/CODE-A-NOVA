"""
src/preprocess.py
-----------------
Module for loading, inspecting, and preprocessing email / SMS text data.

Key ML Concepts Explained:
1. Label Encoding: Machine learning models require numerical targets. We map
   categorical labels ('ham' and 'spam') to binary values (0 and 1).
2. Lowercasing: Standardizes vocabulary so that words like "Free", "FREE",
   and "free" map to the same token, avoiding redundant feature dimensions.
3. Noise & Punctuation Removal: Strips symbols and numbers that add variance
   without carrying generalizable semantic meaning.
4. Tokenization: Splits continuous strings into discrete word tokens.
5. Stopword Removal: Drops high-frequency English functional words (e.g., 'the',
   'is', 'in') that do not help distinguish spam from legitimate mail.
6. Stemming (PorterStemmer): Collapses inflected or derived words down to their
   base root (e.g., 'winner', 'winning', 'wins' -> 'win') to combat vocabulary
   sparsity.
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Combine scikit-learn's standard stopword list with a set for O(1) lookup
STOPWORDS = set(ENGLISH_STOP_WORDS)


def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file, handles missing values,
    normalizes column names, encodes binary labels, and prints dataset statistics.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with standardized columns: 'label' (int: 0/1) and 'message' (str).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Read CSV with fallback encodings (Kaggle/UCI spam files often use latin-1 or utf-8)
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")

    # Kaggle dataset often names columns ['v1', 'v2', 'Unnamed: 2', ...]
    # Standardize column names to 'label' and 'message'
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
        # Drop extra unnamed columns if present
        df = df[["label", "message"]]
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]
    else:
        # Use first two columns if names differ
        df = df.iloc[:, [0, 1]]
        df.columns = ["label", "message"]

    # 1. Handle Missing Values
    initial_rows = len(df)
    df = df.dropna(subset=["label", "message"]).copy()
    dropped_na = initial_rows - len(df)

    # 2. Convert text columns to string to ensure consistent types
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    # 3. Label Encoding: 0 for ham (legitimate), 1 for spam
    # Using explicit mapping ensures determinism
    label_mapping = {"ham": 0, "spam": 1}
    df["label"] = df["label"].map(label_mapping)

    # Drop any unexpected labels that could not be mapped
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # 4. Display Class Distribution & Inspection
    total_samples = len(df)
    ham_count = (df["label"] == 0).sum()
    spam_count = (df["label"] == 1).sum()

    print("=" * 60)
    print("DATASET INSPECTION & SUMMARY")
    print("=" * 60)
    print(f"Total valid samples: {total_samples:,}")
    if dropped_na > 0:
        print(f"Dropped rows with missing values: {dropped_na}")
    print(f"  - HAM  (Legitimate, label 0): {ham_count:,} ({ham_count / total_samples * 100:.2f}%)")
    print(f"  - SPAM (Junk/Fraud, label 1): {spam_count:,} ({spam_count / total_samples * 100:.2f}%)")
    print("\nSample records:")
    print(df.head(3))
    print("=" * 60)

    return df


def clean_text(text: str, use_stemming: bool = True) -> str:
    """
    Cleans an input text string through a sequential NLP pipeline:
    1. Lowercases the text.
    2. Removes URLs and email addresses.
    3. Strips punctuation, symbols, and numeric digits.
    4. Tokenizes into individual words.
    5. Filters out English stopwords.
    6. (Optional) Applies Porter Stemming to reduce words to root form.

    Parameters
    ----------
    text : str
        Raw input email or SMS string.
    use_stemming : bool, default=True
        Whether to apply Porter Stemming.

    Returns
    -------
    str
        Processed and cleaned space-separated string of tokens.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase to standardize case variations
    text = text.lower()

    # Step 2: Remove URLs (e.g., http://..., www...) and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Step 3: Remove punctuation, numbers, and special characters, retaining only alphabetic words
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Step 4: Tokenization - extract alphabetical tokens of length >= 2
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text)

    # Step 5: Remove Stopwords (non-informative words like 'and', 'the', 'is')
    filtered_tokens = [word for word in tokens if word not in STOPWORDS]

    # Step 6: Stemming - reduce tokens to their morphological root
    if use_stemming:
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    else:
        processed_tokens = filtered_tokens

    # Reassemble tokens back into a single whitespace-delimited string
    return " ".join(processed_tokens)


def preprocess_dataset(df: pd.DataFrame, text_column: str = "message", use_stemming: bool = True) -> pd.DataFrame:
    """
    Applies text cleaning across an entire dataframe column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing raw text.
    text_column : str, default='message'
        Column name holding the text to clean.
    use_stemming : bool, default=True
        Whether to apply stemming.

    Returns
    -------
    pd.DataFrame
        Dataframe with a new 'clean_message' column added.
    """
    print(f"\n[INFO] Preprocessing {len(df):,} text records (Stemming={use_stemming})...")
    df = df.copy()
    df["clean_message"] = df[text_column].apply(lambda x: clean_text(x, use_stemming=use_stemming))
    print("[INFO] Text preprocessing complete.")
    return df


if __name__ == "__main__":
    # Self-test block: execute standalone to verify data loading and preprocessing
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "spam.csv")

    data = load_and_inspect_data(dataset_path)

    print("\nTesting text cleaning on sample spam message:")
    sample_raw = "CONGRATULATIONS!! You won a £1,000 Walmart Gift Card! Claim now at http://win.fake or call 0800123."
    print("Raw:     ", sample_raw)
    print("Cleaned: ", clean_text(sample_raw))

    data_cleaned = preprocess_dataset(data.head(5))
    print("\nFirst 3 cleaned rows:")
    for idx, row in data_cleaned.head(3).iterrows():
        print(f"[{'SPAM' if row['label'] == 1 else 'HAM'}] Original: {row['message'][:60]}...")
        print(f"       Cleaned:  {row['clean_message']}")

