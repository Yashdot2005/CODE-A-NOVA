"""
movie_recommender/src/content_based.py
--------------------------------------
Module for Content-Based Movie Recommendation using TF-IDF and Cosine Similarity.

Key ML Concepts Explained:
1. Content-Based Filtering:
   - Evaluates the intrinsic properties/attributes of items (genres, title tokens, metadata).
   - "If you liked Movie A, you will also like Movie B because they share similar themes and genres."
   - Cold-Start Advantage: Can recommend new or niche movies even if no user has rated them yet!
2. TF-IDF on Metadata:
   - Converts genre lists and keywords into weighted sparse vector representations.
   - Distinctive genres (e.g., "Film-Noir", "Musical", "Animation") receive appropriate weighting.
3. Cosine Similarity:
   - Measures the cosine of the angle between two multi-dimensional feature vectors:
     Cosine(A, B) = (A . B) / (||A|| * ||B||)
   - Value ranges from 0 (completely orthogonal/unrelated genres) to 1 (identical metadata profile).
4. Memory-Efficient Vector Querying:
   - Rather than storing an entire N x N pairwise dense similarity matrix (which requires massive RAM),
     we compute the dot product between the query movie's vector and the pre-normalized TF-IDF matrix:
     similarities = tfidf_matrix.dot(query_vector.T)
   - Runs in single-digit milliseconds and scales smoothly.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root and workspace root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(WORKSPACE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from movie_recommender.src.preprocess import clean_movie_metadata, search_movies
except (ImportError, ModuleNotFoundError):
    from src.preprocess import clean_movie_metadata, search_movies


class ContentBasedRecommender:
    """
    Content-Based Recommendation Engine based on movie metadata and Cosine Similarity.
    """

    def __init__(self, movies_df: pd.DataFrame):
        """
        Initializes the recommender with cleaned movie metadata.

        Parameters
        ----------
        movies_df : pd.DataFrame
            DataFrame containing 'movieId', 'title', 'genres', etc.
        """
        if "content_metadata" not in movies_df.columns:
            self.movies_df = clean_movie_metadata(movies_df)
        else:
            self.movies_df = movies_df.copy()

        # Build reverse index: normalized title -> dataframe index
        self.title_to_idx = {
            row["clean_title"].lower(): idx
            for idx, row in self.movies_df.iterrows()
        }
        self.movie_id_to_idx = {
            row["movieId"]: idx
            for idx, row in self.movies_df.iterrows()
        }

        self.vectorizer = None
        self.tfidf_matrix = None
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        """
        Fits a TfidfVectorizer on movie metadata (title keywords + formatted genres).
        """
        print("[INFO] Building Content-Based feature space using TF-IDF...")
        # Use sublinear TF scaling to dampen the effect of very long titles/tokens
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["content_metadata"])
        print(f"[INFO] Content TF-IDF Matrix shape: {self.tfidf_matrix.shape}")

    def recommend(
        self,
        query: str,
        top_n: int = 5,
        min_rating_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Recommends top N movies most similar to the queried title.

        Parameters
        ----------
        query : str
            Movie title to base recommendations on (e.g., 'Toy Story', 'The Matrix').
        top_n : int, default=5
            Number of recommendations to return.
        min_rating_count : int, default=0
            Filter out movies with fewer than this many ratings.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked list of recommended movies with similarity scores and metadata.
        """
        query_norm = query.strip().lower()

        # Find target movie index
        target_idx = None
        if query_norm in self.title_to_idx:
            target_idx = self.title_to_idx[query_norm]
        else:
            # Fuzzy substring search
            matches = search_movies(query, self.movies_df, limit=1)
            if not matches.empty:
                target_idx = matches.index[0]

        if target_idx is None:
            raise ValueError(f"Movie '{query}' not found in catalog. Try searching with a different keyword.")

        target_row = self.movies_df.iloc[target_idx]
        target_title = target_row["title"]
        target_genres = target_row["genres"]

        # Compute cosine similarity between the target vector and all movie vectors
        target_vector = self.tfidf_matrix[target_idx]
        # Since TF-IDF vectors are L2-normalized by scikit-learn, the dot product equals Cosine Similarity
        similarity_scores = self.tfidf_matrix.dot(target_vector.T).toarray().ravel()

        # Sort indices by descending similarity
        ranked_indices = np.argsort(similarity_scores)[::-1]

        recommendations = []
        for idx in ranked_indices:
            if idx == target_idx:
                continue  # Skip the query movie itself

            row = self.movies_df.iloc[idx]
            sim_score = float(similarity_scores[idx])

            # Stop if similarity is zero
            if sim_score <= 0.0:
                break

            # Filter by minimum rating count if provided
            if min_rating_count > 0 and "rating_count" in row:
                if row["rating_count"] < min_rating_count:
                    continue

            recommendations.append({
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "clean_title": row["clean_title"],
                "year": row.get("year", ""),
                "genres": row["genres"],
                "similarity_score": round(sim_score * 100.0, 2),
                "rating_mean": float(row.get("rating_mean", 0.0)) if "rating_mean" in row else None
            })

            if len(recommendations) >= top_n:
                break

        return recommendations


if __name__ == "__main__":
    from src.preprocess import load_movielens_data, compute_movie_stats

    movies_raw, ratings_raw = load_movielens_data()
    movies_cleaned = clean_movie_metadata(movies_raw)
    movies_with_stats = compute_movie_stats(movies_cleaned, ratings_raw)

    recommender = ContentBasedRecommender(movies_with_stats)

    test_queries = ["The Matrix", "Toy Story", "The Dark Knight", "Inception"]

    for q in test_queries:
        print("\n" + "=" * 65)
        print(f"CONTENT-BASED RECOMMENDATIONS FOR: '{q}'")
        print("=" * 65)
        try:
            recs = recommender.recommend(q, top_n=5)
            for i, r in enumerate(recs, 1):
                print(f" {i}. {r['title']} ({r['genres']})")
                print(f"    Similarity: {r['similarity_score']}%")
        except Exception as e:
            print(f"Error: {e}")

