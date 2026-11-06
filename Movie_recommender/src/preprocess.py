"""
movie_recommender/src/preprocess.py
-----------------------------------
Module for loading, inspecting, aggregating, and preprocessing MovieLens data.

Key ML Concepts Explained:
1. Content-Based vs. Collaborative Metadata:
   - Content-Based Filtering relies on item features (genres, titles, descriptions).
   - Collaborative Filtering relies on user interactions (explicit ratings from 0.5 to 5.0).
2. The Cold-Start & Long-Tail Problem:
   - Obscure movies with a single 5.0-star rating can distort recommendations.
   - We aggregate rating statistics (vote count and mean rating) and apply minimum-vote
     thresholds to identify reliable, high-confidence titles.
3. Feature Tokenization for Metadata:
   - MovieLens encodes genres as pipe-delimited strings (e.g., "Action|Adventure|Sci-Fi").
   - We clean and tokenize these strings into whitespace-separated keywords suitable
     for TF-IDF feature extraction.
4. Title Normalization & Fuzzy Search:
   - Titles often include release years, e.g., "Matrix, The (1999)".
   - We extract years and build a normalized lookup index so user searches like
     "the matrix" or "dark knight" resolve accurately.
"""

import os
import re
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Ensure project root is accessible in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_movielens_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads movies.csv and ratings.csv, performs data validation, and prints dataset summary.

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing MovieLens CSV files. Defaults to 'movie_recommender/data'.

    Returns
    -------
    movies_df, ratings_df : Tuple[pd.DataFrame, pd.DataFrame]
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")

    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"MovieLens CSVs not found in '{data_dir}'. "
            "Please run `python movie_recommender/data/download_movielens.py` first."
        )

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    # Clean missing values
    movies_df = movies_df.dropna(subset=["movieId", "title", "genres"]).copy()
    ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"]).copy()

    # Ensure types
    movies_df["movieId"] = movies_df["movieId"].astype(int)
    movies_df["title"] = movies_df["title"].astype(str).str.strip()
    movies_df["genres"] = movies_df["genres"].astype(str).str.strip()

    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(int)
    ratings_df["rating"] = ratings_df["rating"].astype(float)

    print("=" * 65)
    print("MOVIELENS DATASET INSPECTION & SUMMARY")
    print("=" * 65)
    print(f"Total Movies in Catalog:  {len(movies_df):,}")
    print(f"Total User Ratings:       {len(ratings_df):,}")
    print(f"Unique Active Users:      {ratings_df['userId'].nunique():,}")
    print(f"Rating Scale:             {ratings_df['rating'].min()} to {ratings_df['rating'].max()} stars")
    print(f"Global Average Rating:    {ratings_df['rating'].mean():.2f} stars")
    print("=" * 65)

    return movies_df, ratings_df


def clean_movie_metadata(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans movie titles, extracts release years, and formats genres
    into whitespace-separated text tokens for TF-IDF processing.

    Parameters
    ----------
    movies_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with added columns: 'year', 'clean_title', 'genre_features'
    """
    df = movies_df.copy()

    # Extract 4-digit release year from title (e.g. "Toy Story (1995)" -> 1995)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").fillna("")

    # Clean title: remove parenthesized year and standardize articles ("Matrix, The" -> "The Matrix")
    def standardize_title(t: str) -> str:
        # Remove (YYYY)
        clean = re.sub(r"\s*\(\d{4}\).*", "", t).strip()
        # Handle trailing ", The", ", A", ", An"
        match = re.match(r"^(.*?),\s*(The|A|An)$", clean, re.IGNORECASE)
        if match:
            clean = f"{match.group(2)} {match.group(1)}"
        return clean

    df["clean_title"] = df["title"].apply(standardize_title)

    # Convert pipe-separated genres "Action|Adventure|Sci-Fi" to space-delimited "Action Adventure Sci-Fi"
    # Replace hyphens with underscore to treat "Sci-Fi" or "Film-Noir" as a single token "scifi", "filmnoir"
    def format_genres(g: str) -> str:
        if g == "(no genres listed)":
            return ""
        tokens = g.lower().replace("-", "").split("|")
        return " ".join(tokens)

    df["genre_features"] = df["genres"].apply(format_genres)

    # Combined content metadata string (combining clean title tokens + genre tokens)
    df["content_metadata"] = df["clean_title"].str.lower() + " " + df["genre_features"]

    return df


def compute_movie_stats(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rating statistics for each movie:
    - rating_count: number of user reviews
    - rating_mean: average rating score
    - weighted_score: Bayesian weighted rating formula (IMDb formula) to balance count and mean.

    Formula:
      Weighted Score = (v / (v + m)) * R + (m / (v + m)) * C
      where:
        R = average rating for the movie
        v = number of votes for the movie
        m = minimum votes required to be listed (90th percentile)
        C = mean vote across the whole report
    """
    stats = ratings_df.groupby("movieId").agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean")
    ).reset_index()

    merged = movies_df.merge(stats, on="movieId", how="left")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["rating_mean"] = merged["rating_mean"].fillna(0.0).round(2)

    # Compute weighted rating
    C = ratings_df["rating"].mean()
    m = stats["rating_count"].quantile(0.85)  # threshold: top 15% most rated

    v = merged["rating_count"]
    R = merged["rating_mean"]

    # Calculate weighted score
    merged["weighted_score"] = np.where(
        v >= m,
        (v / (v + m)) * R + (m / (v + m)) * C,
        R * 0.7 + C * 0.3  # slight shrinkage for lower volume
    ).round(3)

    return merged


def search_movies(query: str, movies_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """
    Helper function to search movies by title substring or keyword.

    Parameters
    ----------
    query : str
        Search phrase (e.g. "dark knight", "matrix", "toy story")
    movies_df : pd.DataFrame
    limit : int, default=5

    Returns
    -------
    pd.DataFrame of matching movies
    """
    query_clean = query.strip().lower()
    matches = movies_df[
        movies_df["title"].str.lower().str.contains(query_clean, regex=False) |
        movies_df["clean_title"].str.lower().str.contains(query_clean, regex=False)
    ]
    return matches.head(limit)


if __name__ == "__main__":
    # Self-test block: verify data loading, metadata cleaning, and movie stats
    movies, ratings = load_movielens_data()
    movies_cleaned = clean_movie_metadata(movies)
    movies_with_stats = compute_movie_stats(movies_cleaned, ratings)

    print("\nTop 5 Most Rated Movies in Catalog:")
    top_rated = movies_with_stats.sort_values(by="rating_count", ascending=False).head(5)
    print(top_rated[["movieId", "title", "genres", "rating_count", "rating_mean", "weighted_score"]])

    print("\nSearch Test: 'matrix'")
    print(search_movies("matrix", movies_cleaned)[["movieId", "clean_title", "genres"]])

