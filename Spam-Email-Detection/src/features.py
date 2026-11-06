"""
src/features.py
---------------
Module for text feature extraction using Bag-of-Words (CountVectorizer)
and TF-IDF (TfidfVectorizer).

Key ML Concepts Explained:
1. Feature Extraction: Machine learning models (e.g., Logistic Regression, Naive Bayes)
   cannot directly consume raw strings. We must convert text documents into numeric feature matrices.
2. Bag-of-Words (CountVectorizer):
   - Represents each text by the frequency count of each word in the vocabulary.
   - Ignores word order and grammar, focusing purely on token occurrences.
3. TF-IDF (TfidfVectorizer):
   - Computes: Term Frequency (TF) x Inverse Document Frequency (IDF).
   - TF: How frequently a word appears inside a single document.
   - IDF: Logarithmic measure of how rare the word is across the whole corpus.
   - Result: Penalizes common non-discriminative words while rewarding keywords that are
     uniquely characteristic of spam or ham.
4. Avoiding Data Leakage:
   - CRITICAL: Vectorizers must be fitted ONLY on the training split (fit_transform).
   - The test split (and unseen real-world predictions) must ONLY be converted using transform().
   - Fitting on test data would leak global vocabulary frequency and IDF weights into the training
     phase, creating overly optimistic and deceptive performance metrics.
"""

from typing import Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_count_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 2
) -> CountVectorizer:
    """
    Instantiates a CountVectorizer (Bag-of-Words) with standard NLP parameters.

    Parameters
    ----------
    max_features : int, default=5000
        Maximum number of most frequent vocabulary tokens to track.
    ngram_range : tuple of (int, int), default=(1, 1)
        The lower and upper boundary of n-grams (e.g., (1, 1) for unigrams, (1, 2) for bigrams).
    min_df : int, default=2
        Ignore tokens that appear in fewer than `min_df` documents (filters rare typos).

    Returns
    -------
    CountVectorizer
        Configured CountVectorizer instance.
    """
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )


def get_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 2
) -> TfidfVectorizer:
    """
    Instantiates a TfidfVectorizer with standard NLP parameters.

    Parameters
    ----------
    max_features : int, default=5000
        Maximum number of vocabulary tokens by TF-IDF ranking.
    ngram_range : tuple of (int, int), default=(1, 1)
        Lower and upper range of n-values for extracted n-grams.
    min_df : int, default=2
        Ignore tokens appearing in fewer than `min_df` documents.

    Returns
    -------
    TfidfVectorizer
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )


def extract_features(
    train_texts,
    test_texts,
    method: str = "tfidf",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> Tuple[Any, Any, Any]:
    """
    Extracts numerical features from text splits while strictly preventing data leakage.

    Parameters
    ----------
    train_texts : Iterable[str]
        Text samples for the training set.
    test_texts : Iterable[str]
        Text samples for the testing/evaluation set.
    method : str, default='tfidf'
        Feature representation: 'tfidf' or 'count'.
    max_features : int, default=5000
        Maximum vocabulary size.
    ngram_range : tuple of (int, int), default=(1, 2)
        Use unigrams and bigrams by default (e.g., 'free' and 'free entry').
    min_df : int, default=2
        Cutoff for rare tokens.

    Returns
    -------
    X_train_vec : scipy.sparse matrix
        Feature matrix for training data.
    X_test_vec : scipy.sparse matrix
        Feature matrix for test data.
    vectorizer : CountVectorizer or TfidfVectorizer
        Fitted vectorizer object (fitted on training data only).
    """
    method = method.lower().strip()
    if method == "tfidf":
        vectorizer = get_tfidf_vectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df
        )
    elif method in ["count", "bow"]:
        vectorizer = get_count_vectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df
        )
    else:
        raise ValueError(f"Unsupported extraction method '{method}'. Choose 'tfidf' or 'count'.")

    # Fit on training data ONLY, then transform both train and test
    print(f"[INFO] Fitting {vectorizer.__class__.__name__} on training data (max_features={max_features})...")
    X_train_vec = vectorizer.fit_transform(train_texts)

    # Transform test set using the learned training vocabulary (NO fit on test data!)
    print(f"[INFO] Transforming test data using fitted vocabulary (preventing data leakage)...")
    X_test_vec = vectorizer.transform(test_texts)

    print(f"[INFO] Feature matrix shapes -> Train: {X_train_vec.shape}, Test: {X_test_vec.shape}")
    print(f"[INFO] Total vocabulary size: {len(vectorizer.vocabulary_):,} features")

    return X_train_vec, X_test_vec, vectorizer


if __name__ == "__main__":
    # Self-test block: verify feature extraction on mock data
    sample_train = [
        "congratul win free cash prize call now",
        "hi are we meet for lunch today",
        "urgent bank account suspend verify immedi",
        "project report deadline extend till monday"
    ]
    sample_test = [
        "win cash prize today",
        "lunch meeting project report"
    ]

    print("--- Testing TF-IDF Extraction ---")
    X_tr_tfidf, X_te_tfidf, tfidf_vec = extract_features(sample_train, sample_test, method="tfidf", min_df=1)
    print("TF-IDF Sample Features:", list(tfidf_vec.vocabulary_.keys())[:8])

    print("\n--- Testing Count Extraction ---")
    X_tr_cnt, X_te_cnt, cnt_vec = extract_features(sample_train, sample_test, method="count", min_df=1)
    print("Count Sample Features:", list(cnt_vec.vocabulary_.keys())[:8])

