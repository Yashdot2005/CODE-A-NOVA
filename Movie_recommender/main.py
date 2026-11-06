"""
movie_recommender/main.py
-------------------------
Code-A-Nova Machine Learning Internship
Project: Level 2(b) - Movie Recommendation System
-------------------------------------------------
Orchestrates and demonstrates:
1. MovieLens Dataset Ingestion & EDA (movies.csv, ratings.csv)
2. Content-Based Recommendation (TF-IDF + Cosine Similarity)
3. Collaborative Filtering (User-Item Matrix + SVD Matrix Factorization)
4. Hybrid Recommendation Engine
5. CLI Interface for interactive recommendations
"""

import os
import sys
import argparse
from pathlib import Path

# Add both workspace root and movie_recommender to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for p in [str(WORKSPACE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Support both namespaced and local imports to satisfy all IDEs and execution contexts
try:
    from movie_recommender.src.preprocess import load_movielens_data, search_movies
    from movie_recommender.src.engine import MovieRecommenderEngine
except (ImportError, ModuleNotFoundError):
    from src.preprocess import load_movielens_data, search_movies
    from src.engine import MovieRecommenderEngine


def print_banner():
    banner = """
    ======================================================================
                         CODE-A-NOVA ML INTERNSHIP
                LEVEL 2(b): MOVIE RECOMMENDATION SYSTEM
    ======================================================================
    Recommender Approaches:
      1. Content-Based Filtering: Cosine similarity on TF-IDF genre metadata
      2. Collaborative Filtering: Truncated SVD Matrix Factorization
      3. Hybrid Recommendation:   Weighted blend of content and user taste
    ======================================================================
    """
    print(banner)


def run_pipeline():
    print_banner()

    # Step 1: Initialize unified engine
    print("[STEP 1/4] Loading MovieLens dataset and initializing recommendation engines...")
    engine = MovieRecommenderEngine()

    # Step 2: Content-Based Filtering Demo
    print("\n" + "=" * 70)
    print("[STEP 2/4] DEMONSTRATING CONTENT-BASED FILTERING (By Movie Similarity)")
    print("=" * 70)

    demo_movies = ["Toy Story", "The Dark Knight", "Inception"]
    for title in demo_movies:
        print(f"\n--- Recommendations based on: '{title}' ---")
        try:
            recs = engine.recommend_by_movie(title, top_n=4)
            for i, r in enumerate(recs, 1):
                print(f"  {i}. {r['title']}")
                print(f"     Genres: {r['genres']} | Similarity: {r['similarity_score']}%")
        except Exception as e:
            print(f"  Error: {e}")

    # Step 3: Collaborative Filtering Demo
    print("\n" + "=" * 70)
    print("[STEP 3/4] DEMONSTRATING COLLABORATIVE FILTERING (Personalized by User ID)")
    print("=" * 70)

    demo_users = [1, 2]
    for uid in demo_users:
        print(f"\n--- User #{uid} Past Rated Favorites ---")
        history = engine.collaborative_engine.get_user_top_rated(uid, top_n=3)
        for h in history:
            print(f"  * {h['title']} ({h['genres']}) - Rated: {h['rating']} ★")

        print(f"\n--- User #{uid} Top Recommended Unwatched Movies ---")
        user_recs = engine.recommend_for_user(uid, top_n=4)
        for i, r in enumerate(user_recs, 1):
            print(f"  {i}. {r['title']}")
            print(f"     Genres: {r['genres']} | Predicted Rating: {r['predicted_rating']} ★ ({r['total_ratings']} reviews)")

    # Step 4: Hybrid Recommendation Demo
    print("\n" + "=" * 70)
    print("[STEP 4/4] DEMONSTRATING HYBRID RECOMMENDATION (User Taste + Movie Context)")
    print("=" * 70)
    print("Target: User #1 browsing 'The Matrix'")
    hybrid_recs = engine.hybrid_recommend(user_id=1, context_movie_title="The Matrix", top_n=4)
    for i, r in enumerate(hybrid_recs, 1):
        print(f"  {i}. {r['title']}")
        print(f"     Genres: {r['genres']}")
        print(f"     Content Match: {r['content_similarity']} | Predicted User Rating: {r['predicted_user_rating']} ★ | Hybrid Score: {r['hybrid_score']}")

    # Save models
    models_dir = os.path.join(PROJECT_ROOT, "models")
    engine.save_engine(models_dir)

    print("\n" + "=" * 70)
    print("[SUCCESS] Movie Recommendation Pipeline completed successfully!")
    print(f"Models and similarity matrices saved in: {models_dir}")
    print("=" * 70)
    return engine


def interactive_cli(engine: MovieRecommenderEngine):
    """
    Optional interactive loop allowing users to test custom movie queries in real-time.
    """
    print("\n" + "*" * 70)
    print("INTERACTIVE RECOMMENDATION TOOL")
    print("Type a movie title to get content recommendations, or 'user:<id>' for user recommendations.")
    print("Type 'exit' or 'quit' to finish.")
    print("*" * 70)

    while True:
        try:
            user_input = input("\nEnter query > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode. Goodbye!")
                break

            if user_input.lower().startswith("user:"):
                uid_str = user_input.split(":", 1)[1].strip()
                try:
                    uid = int(uid_str)
                    recs = engine.recommend_for_user(uid, top_n=5)
                    print(f"\nPersonalized Recommendations for User #{uid}:")
                    for i, r in enumerate(recs, 1):
                        print(f"  {i}. {r['title']} ({r['genres']}) - Predicted: {r['predicted_rating']} ★")
                except ValueError:
                    print("Invalid user ID format. Please use integers e.g. 'user:1'.")
            else:
                recs = engine.recommend_by_movie(user_input, top_n=5)
                print(f"\nRecommendations based on '{user_input}':")
                for i, r in enumerate(recs, 1):
                    print(f"  {i}. {r['title']} ({r['genres']}) - Similarity: {r['similarity_score']}%")

        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens Recommendation System")
    parser.add_argument("--movie", type=str, help="Get content recommendations for a movie title.")
    parser.add_argument("--user", type=int, help="Get collaborative recommendations for a user ID.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive search shell.")
    args = parser.parse_args()

    if args.movie:
        eng = MovieRecommenderEngine()
        recs = eng.recommend_by_movie(args.movie, top_n=5)
        print(f"\nRecommendations for '{args.movie}':")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']} ({r['genres']}) - Match: {r['similarity_score']}%")
    elif args.user is not None:
        eng = MovieRecommenderEngine()
        recs = eng.recommend_for_user(args.user, top_n=5)
        print(f"\nRecommendations for User #{args.user}:")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']} ({r['genres']}) - Predicted: {r['predicted_rating']} ★")
    else:
        eng = run_pipeline()
        if args.interactive:
            interactive_cli(eng)

