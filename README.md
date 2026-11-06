# 🚀 CODE-A-NOVA Machine Learning Internship
### Comprehensive Machine Learning Repository

Welcome to the **CODE-A-NOVA Machine Learning Internship** repository. This repository contains two Machine Learning projects developed as part of the internship program, demonstrating practical applications of Natural Language Processing, supervised learning, and recommendation systems.

The projects follow a modular development approach with separate components for data preprocessing, feature engineering, model training, evaluation, and prediction.

---

## 📌 Projects Overview

| Level & Task | Domain | Algorithms & Techniques | Location |
|---|---|---|---|
| **Level 1(b): Email & SMS Spam Detection** | Natural Language Processing (NLP) | Multinomial Naive Bayes, Logistic Regression, TF-IDF, Bag-of-Words, Porter Stemmer | Root directory (`main.py`, `src/`) |
| **Level 2(b): Movie Recommendation System** | Recommender Systems | Content-Based Filtering, TF-IDF, Cosine Similarity, SVD Matrix Factorization, Hybrid Recommendation | `movie_recommender/` |

---

# 📧 Project 1: Level 1(b) – Email & SMS Spam Detection

## Project Overview

A modular Machine Learning and Natural Language Processing project that classifies text messages into two categories:

- **SPAM:** Unwanted promotional, fraudulent, or suspicious messages.
- **HAM (NOT SPAM):** Genuine and legitimate messages.

The system uses text preprocessing, feature extraction, and supervised Machine Learning algorithms to identify spam messages.

### Key Features

- Text cleaning and normalization.
- Tokenization, stopword removal, and Porter Stemming.
- TF-IDF and Bag-of-Words feature extraction.
- Multinomial Naive Bayes and Logistic Regression.
- Model comparison using evaluation metrics.
- Real-time message classification with confidence scores.
- Modular and reusable ML pipeline.

## Dataset Information

**Dataset:** SMS Spam Collection Dataset  
**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

| Category | Records | Percentage |
|---|---:|---:|
| Ham | 4,825 | 86.59% |
| Spam | 747 | 13.41% |
| **Total** | **5,572** | **100%** |

## Technologies Used

- Python
- Pandas and NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Joblib
- Matplotlib and Seaborn

## Project Structure

```text
CODE-A-NOVA/
│
├── data/
│   ├── download_data.py
│   └── spam.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── best_model.joblib
│   ├── vectorizer.joblib
│   └── model_metadata.json
│
├── requirements.txt
├── main.py
│
└── README.md
```

## Model Performance

Using an 80:20 stratified train-test split:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.86% | 100% | 76.51% | 86.69% |
| Logistic Regression | 96.86% | 100% | 76.51% | 86.69% |

*Results are based on the reported experiment and should be verified against actual training output.*

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

For custom predictions:

```bash
python src/predict.py --text "Congratulations! You won a free gift card!"
```

---

# 🎬 Project 2: Level 2(b) – Movie Recommendation System

## Project Overview

A Machine Learning-based Movie Recommendation System designed to suggest movies based on user preferences, movie characteristics, and similarity between movie profiles.

The project explores different recommendation approaches and combines their strengths to provide relevant movie suggestions.

### Key Features

- Content-Based Movie Recommendations.
- TF-IDF-based movie feature extraction.
- Cosine Similarity for measuring movie similarity.
- Collaborative Filtering using SVD Matrix Factorization.
- Hybrid Recommendation Engine.
- Personalized movie suggestions.
- Ranking and recommendation generation.

## Recommendation Techniques

### 1. Content-Based Filtering

Recommends movies based on their characteristics, such as genres, descriptions, keywords, and other available metadata.

**Techniques:**
- TF-IDF Vectorization.
- Cosine Similarity.
- Movie Feature Representation.

### 2. Collaborative Filtering

Recommends movies using user-item interaction patterns and similarities in user preferences.

**Technique:**
- Singular Value Decomposition (SVD).
- User-Movie Rating Matrix.
- Latent Factor Modeling.

### 3. Hybrid Recommendation System

Combines content-based and collaborative filtering approaches to generate recommendations using both movie characteristics and user preference patterns.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Surprise (SVD), if used in the implementation
- Matplotlib and Seaborn

## Project Structure

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
├── main.py
└── README.md
```

*The directory structure is a suggested organization; adjust it to match your actual implementation.*

## How to Run

Navigate to the project directory:

```bash
cd movie_recommender
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the recommendation system:

```bash
python main.py
```

---

# 🛠️ Overall Technologies & Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Natural Language Processing | NLTK, TF-IDF |
| Classification | Naive Bayes, Logistic Regression |
| Recommendation Systems | Content-Based, Collaborative Filtering, Hybrid |
| Matrix Factorization | SVD |
| Similarity Measurement | Cosine Similarity |
| Model Persistence | Joblib, Pickle |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git and GitHub |

---

# 🎓 Key Learning Outcomes

Through these projects, I gained practical experience in:

- Developing end-to-end Machine Learning pipelines.
- Understanding Natural Language Processing techniques.
- Implementing supervised classification algorithms.
- Applying feature extraction techniques to textual data.
- Building content-based recommendation systems.
- Understanding collaborative filtering and matrix factorization.
- Combining multiple recommendation approaches.
- Evaluating model performance and recommendation quality.
- Organizing projects using modular Python architecture.
- Managing code and projects using Git and GitHub.

---

# 👨‍💻 Internship Information

| Details | Information |
|---|---|
| Organization | CODE-A-NOVA Technology Initiative |
| Program | Machine Learning Internship |
| Level 1(b) | Email & SMS Spam Detection |
| Level 2(b) | Movie Recommendation System |
| Domain | Machine Learning, NLP & Recommender Systems |
| Programming Language | Python |

---

**Developed as part of the CODE-A-NOVA Machine Learning Internship Program.**

⭐ If you find this repository useful, feel free to explore the individual projects and their implementations.
