# CODE-A-NOVA Machine Learning Internship
## Task Level 2(b): Movie Recommendation System

A dual-engine and hybrid movie recommendation system built using the official **MovieLens** dataset. It implements **Content-Based Filtering (TF-IDF + Cosine Similarity)**, **Collaborative Filtering (Matrix Factorization via TruncatedSVD)**, and a **Hybrid Recommender**.

---

## 📌 Project Overview

Recommender systems are the algorithmic backbone of platforms like Netflix, Spotify, and YouTube.

This project delivers three distinct recommendation strategies:
1. **Content-Based Filtering**: Recommends films with matching genres, themes, and metadata to a target movie using vector space similarity.
2. **Collaborative Filtering**: Decomposes the sparse User-Item interaction matrix using **Truncated Singular Value Decomposition (SVD)** to uncover latent user preferences and predict personalized star ratings.
3. **Hybrid Recommender**: Blends content similarity with user taste profiles to balance novelty, relevance, and user alignment.

---

## 📂 Architecture

```
movie_recommender/
├── data/
│   ├── download_movielens.py   # Automated downloader for MovieLens ml-latest-small
│   ├── movies.csv              # Movie catalog (9,742 movies with genres)
│   └── ratings.csv             # 100,836 ratings from 610 users
├── notebooks/
│   └── recommender_walkthrough.ipynb # Interactive Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Ingestion, EDA, genre tokenization, title normalization
│   ├── content_based.py        # TF-IDF Vectorization & Cosine Similarity engine
│   ├── collaborative.py        # Sparse User-Item Matrix & TruncatedSVD Latent Factor model
│   └── engine.py               # Unified Hybrid Recommender engine
├── models/
│   ├── content_engine.joblib   # Persisted Content-Based model
│   └── collab_engine.joblib    # Persisted SVD Collaborative model
├── README.md                   # Full documentation & equations
└── main.py                     # CLI orchestrator & interactive recommender
```

---

## 📊 Dataset: MovieLens (GroupLens Research)

- **Total Movies**: 9,742
- **Total User Reviews**: 100,836
- **Unique Users**: 610
- **Rating Scale**: 0.5 to 5.0 stars
- **Global Mean Rating**: 3.50 stars

---

## 🚀 How to Run

### 1. Run the Full Pipeline & Demo
From the project root:
```bash
python movie_recommender/main.py
```

### 2. Get Recommendations for a Movie (Content-Based)
```bash
python movie_recommender/main.py --movie "The Dark Knight"
```

### 3. Get Personalized Recommendations for a User (Collaborative)
```bash
python movie_recommender/main.py --user 1
```

### 4. Interactive Search Shell
```bash
python movie_recommender/main.py --interactive
```

---

## 🔬 Algorithms & Mathematical Concepts

### 1. Content-Based Filtering (Cosine Similarity)
Transforms genre lists and metadata into TF-IDF vector embeddings:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

* **Sample Output (`Toy Story`)**:
  - `Toy Story 2 (1999)`: 100.0% similarity
  - `Toy Story 3 (2010)`: 91.67% similarity
  - `Monsters, Inc. (2001)`: 41.57% similarity

### 2. Collaborative Filtering (Matrix Factorization via SVD)
The sparse User-Item rating matrix $R \in \mathbb{R}^{610 \times 9724}$ is centered and decomposed into rank-$k$ latent factors:
$$R \approx U \cdot \Sigma \cdot V^T$$

Each user $u$ and movie $m$ is represented as a dense latent vector $\mathbf{p}_u, \mathbf{q}_m \in \mathbb{R}^{25}$. The predicted rating for an unrated movie is:
$$\hat{r}_{u, m} = \mu_u + \mathbf{p}_u \cdot \mathbf{q}_m^T$$

* **Sample Output (User #1 - Crime & Thriller Fan)**:
  - Past Favorites: *Se7en* (5.0★), *The Usual Suspects* (5.0★), *Bottle Rocket* (5.0★)
  - Top Recommendations:
    1. *The Shawshank Redemption (1994)* — Predicted: 4.65 ★
    2. *The Godfather (1972)* — Predicted: 4.65 ★
    3. *Twelve Monkeys (1995)* — Predicted: 4.58 ★
    4. *Trainspotting (1996)* — Predicted: 4.56 ★

### 3. Hybrid Recommender
Combines content relevance with personalized user affinity:
$$\text{Score}_{\text{Hybrid}} = \alpha \cdot \text{Score}_{\text{Content}} + (1 - \alpha) \cdot \text{Score}_{\text{Collaborative}}$$

---

## 📝 What I Learned

- [x] **Item-Feature vs. Behavioral-Feedback Paradigms**: Understood the fundamental differences and trade-offs between Content-Based and Collaborative Filtering.
- [x] **Vector Space Modeling**: Implemented TF-IDF tokenization on multi-genre metadata and computed memory-efficient Cosine Similarity.
- [x] **Handling Matrix Sparsity**: Constructed compressed sparse row (`scipy.sparse.csr_matrix`) matrices to efficiently process 100,000+ ratings with minimal memory footprint.
- [x] **Latent Factor Modeling**: Applied TruncatedSVD to uncover hidden thematic dimensions in movie ratings.
- [x] **Cold-Start Problem**: Designed hybrid blending to overcome cold-start limitations for new movies and users.

