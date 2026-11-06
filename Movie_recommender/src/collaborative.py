"""
movie_recommender/src/collaborative.py
--------------------------------------
Module for Collaborative Filtering using User-Item Matrix Factorization (TruncatedSVD).

Key ML Concepts Explained:
1. Collaborative Filtering:
   - Recommends items based on user behavioral consensus and rating correlation.
   - Core assumption: If User A and User B both loved "Inception" and "Interstellar",
     and User A loved "The Prestige", then User B will likely love "The Prestige" as well.
2. User-Item Interaction Matrix (Sparse Matrix):
   - Rows = Users (610 users), Columns = Movies (9,742 movies).
   - Values = Explicit star ratings (0.5 to 5.0).
   - The matrix is >98% sparse because most users rate only a tiny fraction of catalog movies.
3. Matrix Factorization (Truncated SVD / Latent Factor Model):
   - Decomposes the user-item rating matrix R into lower-rank user embeddings U and item embeddings V:
     R ~ U * Sigma * V^T
   - Uncovers hidden (latent) thematic dimensions (e.g., sci-fi complexity, comedic tone, emotional depth).
4. Predicting Unseen Ratings:
   - Dot product between user latent vector (u) and movie latent vector (v) predicts the expected rating:
     Predicted_Rating(u, m) ~ u . v^T
   - We rank all unwatched movies by predicted rating to generate personalized recommendations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Ensure project root and workspace root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(WORKSPACE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from movie_recommender.src.preprocess import load_movielens_data
except (ImportError, ModuleNotFoundError):
    from src.preprocess import load_movielens_data


class CollaborativeRecommender:
    """
    Collaborative Filtering Engine using Truncated Singular Value Decomposition (SVD).
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        n_components: int = 20,
        random_state: int = 42
    ):
        """
        Initializes the collaborative recommender.

        Parameters
        ----------
        ratings_df : pd.DataFrame
            Contains 'userId', 'movieId', 'rating'.
        movies_df : pd.DataFrame
            Contains 'movieId', 'title', 'genres'.
        n_components : int, default=20
            Number of latent dimensions for matrix factorization.
        random_state : int, default=42
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.n_components = n_components
        self.random_state = random_state

        # Movie details lookup dictionary
        self.movie_info = self.movies_df.set_index("movieId")[["title", "genres"]].to_dict("index")

        # Create user and movie mapping indices
        self.user_ids = sorted(self.ratings_df["userId"].unique())
        self.movie_ids = sorted(self.ratings_df["movieId"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.idx_to_user = {i: u for i, u in enumerate(self.user_ids)}

        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for i, m in enumerate(self.movie_ids)}

        self.user_item_matrix = None
        self.user_mean_ratings = None
        self.svd = None
        self.user_factors = None
        self.item_factors = None

        self._fit()

    def _fit(self):
        """
        Builds sparse normalized user-item matrix and trains TruncatedSVD.
        """
        print(f"[INFO] Building User-Item Matrix ({len(self.user_ids)} users x {len(self.movie_ids)} movies)...")

        # Map IDs to consecutive integers
        user_indices = self.ratings_df["userId"].map(self.user_to_idx).values
        movie_indices = self.ratings_df["movieId"].map(self.movie_to_idx).values
        ratings = self.ratings_df["rating"].values

        # Construct CSR sparse matrix
        num_users = len(self.user_ids)
        num_movies = len(self.movie_ids)
        raw_matrix = csr_matrix((ratings, (user_indices, movie_indices)), shape=(num_users, num_movies))

        # Compute per-user mean rating to center ratings (de-biasing lenient vs strict raters)
        user_sums = np.array(raw_matrix.sum(axis=1)).ravel()
        user_counts = np.diff(raw_matrix.indptr)
        self.user_mean_ratings = np.where(user_counts > 0, user_sums / user_counts, 3.5)

        # De-meaned ratings for non-zero entries
        demeaned_ratings = ratings - self.user_mean_ratings[user_indices]
        centered_matrix = csr_matrix((demeaned_ratings, (user_indices, movie_indices)), shape=(num_users, num_movies))

        # Fit Truncated SVD
        print(f"[INFO] Training TruncatedSVD with {self.n_components} latent factors...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        # user_factors shape: (num_users, n_components)
        self.user_factors = self.svd.fit_transform(centered_matrix)
        # item_factors shape: (n_components, num_movies)
        self.item_factors = self.svd.components_

        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"[INFO] SVD Factorization complete! Total explained variance: {explained_var * 100:.2f}%")

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 5,
        min_movie_ratings: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Generates personalized recommendations for a given user.

        Parameters
        ----------
        user_id : int
            The ID of the user (e.g., 1 to 610 in MovieLens).
        top_n : int, default=5
            Number of top predicted movies to return.
        min_movie_ratings : int, default=15
            Filter out obscure movies with fewer than this many total ratings.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked list of recommended movies with predicted star ratings.
        """
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found in ratings database.")

        user_idx = self.user_to_idx[user_id]
        user_mean = self.user_mean_ratings[user_idx]

        # Predict ratings for all movies: user_factors[u] . item_factors
        # Shape: (1, n_components) x (n_components, num_movies) -> (num_movies,)
        predicted_deltas = np.dot(self.user_factors[user_idx], self.item_factors)
        predicted_ratings = user_mean + predicted_deltas
        # Clip predicted ratings to valid 0.5 - 5.0 range
        predicted_ratings = np.clip(predicted_ratings, 0.5, 5.0)

        # Identify movies already rated by this user
        watched_movie_ids = set(self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"])

        # Count total ratings per movie for threshold filtering
        movie_counts = self.ratings_df["movieId"].value_counts().to_dict()

        # Rank unwatched movies by predicted score
        candidate_indices = np.argsort(predicted_ratings)[::-1]

        recommendations = []
        for m_idx in candidate_indices:
            m_id = self.idx_to_movie[m_idx]

            # Don't recommend movies the user already watched
            if m_id in watched_movie_ids:
                continue

            # Minimum popularity filter to ensure high quality
            if movie_counts.get(m_id, 0) < min_movie_ratings:
                continue

            movie_meta = self.movie_info.get(m_id, {"title": f"Movie #{m_id}", "genres": "Unknown"})

            recommendations.append({
                "movieId": m_id,
                "title": movie_meta["title"],
                "genres": movie_meta["genres"],
                "predicted_rating": round(float(predicted_ratings[m_idx]), 2),
                "total_ratings": movie_counts.get(m_id, 0)
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    def get_user_top_rated(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Returns the movies previously given the highest ratings by this user.
        """
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        top = user_ratings.sort_values(by="rating", ascending=False).head(top_n)

        results = []
        for _, row in top.iterrows():
            m_id = int(row["movieId"])
            meta = self.movie_info.get(m_id, {"title": f"Movie #{m_id}", "genres": "Unknown"})
            results.append({
                "movieId": m_id,
                "title": meta["title"],
                "genres": meta["genres"],
                "rating": row["rating"]
            })
        return results


if __name__ == "__main__":
    from src.preprocess import load_movielens_data

    movies, ratings = load_movielens_data()
    recommender = CollaborativeRecommender(ratings, movies, n_components=25)

    test_users = [1, 2, 3]

    for uid in test_users:
        print("\n" + "=" * 65)
        print(f"COLLABORATIVE FILTERING FOR USER #{uid}")
        print("=" * 65)

        print("\nUser's Past Favorite Movies:")
        history = recommender.get_user_top_rated(uid, top_n=3)
        for h in history:
            print(f"  * {h['title']} ({h['genres']}) - Rated: {h['rating']} ★")

        print("\nTop Personalized Recommendations:")
        recs = recommender.recommend_for_user(uid, top_n=5)
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']} ({r['genres']})")
            print(f"     Predicted Rating: {r['predicted_rating']} ★ (from {r['total_ratings']} reviews)")

