"""
movie_recommender/src/engine.py
-------------------------------
Unified Recommendation Engine combining Content-Based and Collaborative Filtering.

Key ML Concepts Explained:
1. Hybrid Recommendation Systems:
   - Content-Based Filtering excels at finding similar items and solving the Cold-Start problem
     for newly added items.
   - Collaborative Filtering excels at discovering serendipitous, unexpected recommendations
     derived from collective human community behavior.
   - A Hybrid System blends both signals:
     Score_Hybrid = alpha * Score_Content + (1 - alpha) * Score_Collaborative
2. Recommender Trade-offs:
   - Diversity vs. Relevance: Balancing exact genre duplicates against broader community favorites.
   - Popularity Bias: Guarding against simply recommending the same top 5 blockbusters to every user.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

# Ensure project root and workspace root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(WORKSPACE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from movie_recommender.src.preprocess import load_movielens_data, clean_movie_metadata, compute_movie_stats
    from movie_recommender.src.content_based import ContentBasedRecommender
    from movie_recommender.src.collaborative import CollaborativeRecommender
except (ImportError, ModuleNotFoundError):
    from src.preprocess import load_movielens_data, clean_movie_metadata, compute_movie_stats
    from src.content_based import ContentBasedRecommender
    from src.collaborative import CollaborativeRecommender


class MovieRecommenderEngine:
    """
    Unified Movie Recommendation Engine providing:
    - Content-Based Recommendations (by movie title)
    - Collaborative Filtering Recommendations (by user ID)
    - Hybrid Recommendations (personalized for a user given a context movie)
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initializes and fits both Content-Based and Collaborative engines.
        """
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, "data")

        # 1. Load data
        self.movies_raw, self.ratings_raw = load_movielens_data(data_dir=data_dir)

        # 2. Preprocess & enrich metadata
        self.movies_cleaned = clean_movie_metadata(self.movies_raw)
        self.movies_enriched = compute_movie_stats(self.movies_cleaned, self.ratings_raw)

        # 3. Instantiate sub-engines
        print("\n[INFO] Initializing Content-Based Engine...")
        self.content_engine = ContentBasedRecommender(self.movies_enriched)

        print("\n[INFO] Initializing Collaborative Filtering Engine...")
        self.collaborative_engine = CollaborativeRecommender(
            self.ratings_raw,
            self.movies_enriched,
            n_components=25
        )

    def recommend_by_movie(self, title: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Finds movies similar to the given title using Content-Based filtering.
        """
        return self.content_engine.recommend(title, top_n=top_n)

    def recommend_for_user(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Finds personalized recommendations for a user using SVD Collaborative Filtering.
        """
        return self.collaborative_engine.recommend_for_user(user_id, top_n=top_n)

    def hybrid_recommend(
        self,
        user_id: int,
        context_movie_title: str,
        top_n: int = 5,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generates hybrid recommendations that balance content similarity to a context movie
        with the user's personalized collaborative taste profile.

        Parameters
        ----------
        user_id : int
            Target user ID.
        context_movie_title : str
            A movie the user currently browses or selected as inspiration.
        top_n : int, default=5
        alpha : float, default=0.5
            Weight for content similarity (0.0 = pure collaborative, 1.0 = pure content-based).

        Returns
        -------
        List[Dict[str, Any]]
        """
        # Step 1: Retrieve candidate pool of content-similar movies
        content_candidates = self.content_engine.recommend(context_movie_title, top_n=30)
        if not content_candidates:
            return []

        # Step 2: Compute collaborative predicted ratings for these candidates
        if user_id in self.collaborative_engine.user_to_idx:
            user_idx = self.collaborative_engine.user_to_idx[user_id]
            user_mean = self.collaborative_engine.user_mean_ratings[user_idx]
            predicted_deltas = np.dot(
                self.collaborative_engine.user_factors[user_idx],
                self.collaborative_engine.item_factors
            )
            pred_ratings = user_mean + predicted_deltas
        else:
            pred_ratings = np.full(len(self.collaborative_engine.movie_ids), 3.5)

        # Step 3: Combine scores
        hybrid_results = []
        for cand in content_candidates:
            m_id = cand["movieId"]
            content_sim_norm = cand["similarity_score"] / 100.0  # normalize to 0..1

            if m_id in self.collaborative_engine.movie_to_idx:
                m_idx = self.collaborative_engine.movie_to_idx[m_id]
                pred_rating = float(np.clip(pred_ratings[m_idx], 0.5, 5.0))
                collab_norm = (pred_rating - 0.5) / 4.5  # normalize 0.5..5.0 to 0..1
            else:
                pred_rating = 3.5
                collab_norm = 0.67

            # Blended weighted score
            hybrid_score = (alpha * content_sim_norm) + ((1.0 - alpha) * collab_norm)

            hybrid_results.append({
                "movieId": m_id,
                "title": cand["title"],
                "genres": cand["genres"],
                "content_similarity": f"{cand['similarity_score']}%",
                "predicted_user_rating": round(pred_rating, 2),
                "hybrid_score": round(hybrid_score * 100.0, 2)
            })

        # Sort by hybrid score descending
        hybrid_results = sorted(hybrid_results, key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_n]

    def save_engine(self, output_dir: Optional[str] = None):
        """
        Serializes and saves engine components into models/ using joblib.
        """
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(output_dir, exist_ok=True)

        content_path = os.path.join(output_dir, "content_engine.joblib")
        collab_path = os.path.join(output_dir, "collab_engine.joblib")

        joblib.dump(self.content_engine, content_path)
        joblib.dump(self.collaborative_engine, collab_path)

        print(f"[SUCCESS] Saved Content engine to:       {content_path}")
        print(f"[SUCCESS] Saved Collaborative engine to: {collab_path}")


if __name__ == "__main__":
    engine = MovieRecommenderEngine()

    print("\n" + "=" * 70)
    print("HYBRID RECOMMENDATION DEMO: USER #1 + CONTEXT 'The Matrix'")
    print("=" * 70)
    hybrid_recs = engine.hybrid_recommend(user_id=1, context_movie_title="The Matrix", top_n=5)
    for i, r in enumerate(hybrid_recs, 1):
        print(f" {i}. {r['title']} ({r['genres']})")
        print(f"    Content Match: {r['content_similarity']} | Predicted User Rating: {r['predicted_user_rating']} ★ | Hybrid: {r['hybrid_score']}")

    # Save models
    engine.save_engine()

