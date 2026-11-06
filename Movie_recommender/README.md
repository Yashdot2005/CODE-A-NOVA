# 🎬 Movie Recommendation System
### CODE-A-NOVA | Machine Learning Internship – Level 2(b)

A Machine Learning-based Movie Recommendation System that suggests relevant movies based on movie characteristics, user preferences, and rating patterns using Content-Based Filtering, Collaborative Filtering, and Hybrid Recommendation techniques.

---

## 📌 Project Overview

With thousands of movies available across different platforms, finding movies that match individual preferences can be challenging.

This project implements a recommendation engine using multiple Machine Learning techniques to generate relevant movie suggestions.

The system explores three recommendation approaches:

- **Content-Based Filtering:** Recommends movies based on genres, descriptions, keywords, and other movie attributes.
- **Collaborative Filtering:** Recommends movies based on user rating patterns and similarities between users.
- **Hybrid Recommendation:** Combines content-based and collaborative filtering to improve recommendation relevance.

### Key Features

- Movie dataset loading and preprocessing.
- Movie metadata analysis and feature extraction.
- TF-IDF-based text vectorization.
- Cosine Similarity for movie-to-movie comparison.
- SVD Matrix Factorization for collaborative recommendations.
- Hybrid Recommendation Engine.
- Personalized movie recommendation generation.
- Modular and reusable architecture.

---

## 📂 Project Structure

```text
movie_recommender/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── notebooks/
│   └── movie_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── content_based.py
│   ├── collaborative.py
│   └── hybrid.py
│
├── models/
│   └── recommendation_model.pkl
│
├── requirements.txt
├── README.md
└── main.py
```

*Note: This is a suggested modular structure. Update filenames according to your actual repository implementation.*

---

## 📊 Dataset Information

The recommendation system uses movie metadata and user-rating information to generate recommendations.

### Dataset Components

**1. Movies Dataset (`movies.csv`)**

Contains information about movies, such as:

- Movie ID
- Movie Title
- Genres
- Movie Description
- Keywords (if available)

**2. Ratings Dataset (`ratings.csv`)**

Contains user interaction information, such as:

- User ID
- Movie ID
- Rating
- Timestamp (if available)

*The actual dataset fields depend on the dataset selected for implementation.*

---

## ⚙️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Feature extraction and similarity calculation |
| TF-IDF | Text feature representation |
| Cosine Similarity | Measuring movie similarity |
| Surprise | SVD-based collaborative filtering, if used |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Pickle / Joblib | Model persistence |

---

## 🧠 Recommendation Algorithms

### 1. Content-Based Filtering

Content-Based Filtering recommends movies based on their characteristics and similarities with movies that a user likes.

**Techniques Used:**

- TF-IDF Vectorization
- Movie Metadata Processing
- Cosine Similarity

**Working:**

1. Extract movie information such as genres and descriptions.
2. Combine relevant movie attributes into a feature representation.
3. Convert textual information into numerical vectors using TF-IDF.
4. Calculate similarity between movie vectors using Cosine Similarity.
5. Recommend movies with the highest similarity scores.

### 2. Collaborative Filtering

Collaborative Filtering generates recommendations using user-rating patterns and preferences.

**Technique: Singular Value Decomposition (SVD)**

SVD is a Matrix Factorization technique that decomposes the user-item interaction matrix into lower-dimensional latent factor matrices.

**Working:**

1. Load user-movie rating information.
2. Construct the user-item interaction matrix.
3. Apply SVD-based matrix factorization.
4. Learn latent representations of users and movies.
5. Predict ratings for movies that users have not rated.
6. Recommend movies with higher predicted ratings.

### 3. Hybrid Recommendation Engine

The Hybrid Recommendation Engine combines Content-Based Filtering and Collaborative Filtering.

**Working:**

1. Generate recommendations using Content-Based Filtering.
2. Generate recommendations using Collaborative Filtering.
3. Combine recommendation scores using a suitable weighting strategy.
4. Rank the movies based on the combined scores.
5. Return the most relevant movie recommendations.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd movie_recommender
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the Complete Recommendation System

```bash
python main.py
```

The application executes the recommendation workflow:

1. Load movie and rating datasets.
2. Inspect and preprocess movie information.
3. Extract movie features using TF-IDF.
4. Calculate movie similarities using Cosine Similarity.
5. Train the SVD-based collaborative filtering model.
6. Generate content-based recommendations.
7. Generate collaborative recommendations.
8. Combine recommendations using the hybrid engine.

### Run Individual Modules

**Data Preprocessing**

```bash
python src/preprocess.py
```

**Content-Based Recommendation**

```bash
python src/content_based.py
```

**Collaborative Filtering**

```bash
python src/collaborative.py
```

**Hybrid Recommendation**

```bash
python src/hybrid.py
```

---

## 🔍 Sample Recommendation

**Example Input:**

```text
Movie Title: Toy Story (1995)
```

**Recommendation Output:**

```text
Recommended Movies:

1. Toy Story 2 (1999)
2. A Bug's Life (1998)
3. Monsters, Inc. (2001)
4. Finding Nemo (2003)
5. Shrek (2001)
```

*Illustrative example only. Actual recommendations depend on the selected dataset and model output.*

---

## 📈 Model Evaluation

Recommendation systems require different evaluation metrics from traditional classification models.

The following metrics can be used to evaluate recommendation quality:

| Metric | Purpose |
|---|---|
| RMSE | Measures rating prediction error |
| MAE | Measures average absolute prediction error |
| Precision@K | Measures relevant recommendations among the top K results |
| Recall@K | Measures how many relevant items are retrieved |
| NDCG@K | Evaluates the quality of recommendation ranking |

### Comparison of Recommendation Approaches

| Approach | Main Advantage | Limitation |
|---|---|---|
| Content-Based | Recommends similar movies using movie attributes | May produce less diverse recommendations |
| Collaborative Filtering | Learns preferences from user-rating patterns | Suffers from cold-start problems |
| Hybrid Recommendation | Combines multiple recommendation approaches | More complex to implement and tune |

*Actual evaluation scores should be added after running experiments on the selected dataset.*

---

## 🎯 Applications

- Movie streaming platforms.
- Personalized movie discovery.
- Entertainment recommendation engines.
- OTT platform recommendations.
- User preference analysis.
- Intelligent content filtering.

---

## 📝 What I Learned

- Understanding Recommendation System fundamentals.
- Implementing Content-Based Filtering.
- Applying TF-IDF for movie feature extraction.
- Measuring similarity using Cosine Similarity.
- Understanding Collaborative Filtering.
- Implementing SVD Matrix Factorization.
- Combining recommendation approaches using a Hybrid Engine.
- Understanding cold-start problems.
- Evaluating recommendation systems using ranking and prediction metrics.
- Developing modular Machine Learning applications.

---

## 👨‍💻 Author & Acknowledgments

- **Internship Program:** CODE-A-NOVA Technology Initiative
- **Task:** Level 2(b) – Movie Recommendation System
- **Domain:** Machine Learning & Recommender Systems
- **Programming Language:** Python
- **Repository:** CODE-A-NOVA Machine Learning Internship

---

*Developed as part of the CODE-A-NOVA Machine Learning Internship Program.*
