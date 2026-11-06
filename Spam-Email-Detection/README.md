# CODE-A-NOVA Machine Learning Internship
### Comprehensive Machine Learning Repository

Welcome to the **CODE-A-NOVA** Machine Learning Internship repository. This repository hosts completed, production-ready machine learning solutions developed to professional software engineering standards.

| Level & Task | Domain | Algorithms & Techniques | Location |
| :--- | :--- | :--- | :--- |
| **Level 1(b): Email & SMS Spam Detection** | Natural Language Processing (NLP) | Multinomial Naive Bayes, Logistic Regression, TF-IDF, Porter Stemmer | Root directory (`main.py`, `src/`) |
| **Level 2(b): Movie Recommendation System** | Recommender Systems | Content-Based (TF-IDF + Cosine Sim), SVD Matrix Factorization, Hybrid Engine | [`movie_recommender/`](file:///movie_recommender/) |

---

# 🚀 Task 1: Level 1(b) - Email & SMS Spam Detection

A modular, production-ready Natural Language Processing (NLP) and Machine Learning project that classifies incoming emails and SMS messages as **SPAM** or **NOT SPAM (HAM)** with probabilistic confidence scores.

---

## 📌 Project Overview

Spam messages constitute a large portion of digital communication, ranging from annoying commercial advertisements to dangerous phishing and financial fraud. 

This project implements a complete, end-to-end Machine Learning pipeline using industry best practices:
- **Modular Architecture**: Separate modules for data preprocessing, feature engineering, model training, and inference.
- **Robust NLP Cleaning**: Case normalization, noise and punctuation stripping, tokenization, stopword removal, and Porter Stemming.
- **Leak-Free Feature Extraction**: TF-IDF and Bag-of-Words representations fitted strictly on training data.
- **Model Comparison**: Multinomial Naive Bayes vs. Logistic Regression evaluated with stratified validation.
- **Production Inference**: Real-time confidence scores and probability calibration for new unseen messages.

---

## 📂 Repository Structure

```
CODE-A-NOVA/
├── data/
│   ├── download_data.py       # Automated dataset downloader & offline bootstrap generator
│   └── spam.csv               # SMS Spam Collection dataset (5,572 records)
├── notebooks/
│   └── exploration.ipynb      # Interactive step-by-step Jupyter Notebook
├── src/
│   ├── __init__.py            # Package initialization
│   ├── preprocess.py          # Data ingestion, inspection, and NLP cleaning pipeline
│   ├── features.py            # CountVectorizer & TfidfVectorizer feature extraction
│   ├── train.py               # Model training, stratified split, evaluation & joblib saving
│   └── predict.py             # Real-time inference function with confidence scoring
├── models/
│   ├── best_model.joblib      # Persisted winning classifier
│   ├── vectorizer.joblib      # Persisted fitted TF-IDF vectorizer
│   └── model_metadata.json    # Evaluation scores and metadata
├── requirements.txt           # Pinned dependency versions
├── README.md                  # Project documentation and results
└── main.py                    # Runs the full pipeline end to end
```

---

## 📊 Dataset Information

- **Dataset**: [SMS Spam Collection Dataset (UCI Machine Learning Repository & Kaggle)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Total Records**: 5,572 messages
  - **Ham (Legitimate, 0)**: 4,825 samples (86.59%)
  - **Spam (Junk/Fraud, 1)**: 747 samples (13.41%)
- **Class Balance**: Moderately imbalanced, requiring **Stratified K-Fold / Stratified Split** to ensure test sets reflect real-world proportions.

### How to Download / Verify the Dataset
The dataset is downloaded automatically when running `main.py`. Alternatively, you can run:
```bash
python data/download_data.py
```
If working fully offline, `download_data.py` automatically generates a representative bootstrap dataset to ensure tests never fail.

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Repository
```bash
cd CODE-A-NOVA
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run the Full Pipeline End to End
To inspect the data, preprocess, extract features, train and evaluate both models, save the best model, and test sample predictions:
```bash
python main.py
```

### Run Individual Modules
- **Test Preprocessing & Data Inspection:**
  ```bash
  python src/preprocess.py
  ```
- **Test Feature Extraction:**
  ```bash
  python src/features.py
  ```
- **Train Models & Compare Metrics:**
  ```bash
  python src/train.py
  ```
- **Run Real-Time Predictions:**
  ```bash
  # Run sample test suite:
  python src/predict.py

  # Classify a custom message:
  python src/predict.py --text "CONGRATULATIONS! You won a $1000 gift card. Call now to claim!"
  ```

---

## 📈 Model Performance & Evaluation

The dataset was split using a stratified 80/20 train-test split (`random_state=42`):
- **Training Set**: 4,457 messages (3,859 Ham, 598 Spam)
- **Testing Set**: 1,115 messages (966 Ham, 149 Spam)

### Comparison Table

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **96.86%** | **100.00%** | 76.51% | 86.69% | **0** |
| **Logistic Regression** | **96.86%** | **100.00%** | 76.51% | 86.69% | **0** |

### Confusion Matrix (Multinomial Naive Bayes on Test Set)

| | Predicted Ham (0) | Predicted Spam (1) |
| :--- | :---: | :---: |
| **Actual Ham (0)** | 966 *(True Negatives)* | **0** *(False Positives)* |
| **Actual Spam (1)** | 35 *(False Negatives)* | 114 *(True Positives)* |

---

## 🎯 Which Metric Matters Most for Spam Detection?

In spam detection, **PRECISION (for the Spam class)** is the most critical metric.

### Why?
$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

- **False Positive (FP) Cost (Very High)**: A legitimate, critical email (such as an employment offer, university admission notice, hospital test result, or banking two-factor authentication code) is wrongly classified as Spam and routed to the junk folder or deleted. The user misses vital information, causing direct real-world harm.
- **False Negative (FN) Cost (Low to Moderate)**: A spam message leaks through into the primary inbox. While annoying, the recipient can spot and delete it in one click.

Therefore, spam classification systems must strive for **near-zero False Positives (approaching 100% Precision)** while maximizing Recall as much as possible.

---

## 🔍 Sample Real-Time Predictions

```
Sample #1:
  Raw Text:     "CONGRATULATIONS! You have been selected to receive a FREE £1,000 Walmart Gift Card! Claim your reward now..."
  Preprocessed: "congratul select receiv free walmart gift card claim reward text claim visit"
  Result:       [SPAM] SPAM
  Confidence:   90.69% (Spam Prob: 0.9069 | Ham Prob: 0.0931)

Sample #2:
  Raw Text:     "Hi Alex, please find attached the revised presentation slides and financial figures for our team sync meeting..."
  Preprocessed: "hi alex attach revis present slide financi figur team sync meet tomorrow let know thought"
  Result:       [NOT SPAM] NOT SPAM
  Confidence:   99.28% (Ham Prob: 0.9928 | Spam Prob: 0.0072)

Sample #3:
  Raw Text:     "Hey, are we still meeting up for coffee this afternoon? Let me know what time works best for you."
  Preprocessed: "hey meet coffe afternoon let know time work best"
  Result:       [NOT SPAM] NOT SPAM
  Confidence:   99.90% (Ham Prob: 0.9990 | Spam Prob: 0.0010)
```

---

## 📝 What I Learned

> *This section reflects key insights gained while completing the Code-A-Nova Level 1(b) Machine Learning Task:*

- [x] **NLP Text Cleaning Pipeline**: Learned why lowercasing, punctuation stripping, tokenization, stopword removal, and stemming (PorterStemmer) are foundational for reducing vocabulary dimensionality and feature sparsity.
- [x] **Feature Representation (TF-IDF vs. BoW)**: Understood how TF-IDF discounts ubiquitously repeated words while emphasizing distinguishing tokens.
- [x] **Preventing Data Leakage**: Discovered why vectorizers must only be fitted on the training split (`fit_transform`) and never on the test split (`transform`).
- [x] **Handling Class Imbalance**: Gained practical experience applying stratified splitting to preserve realistic class ratios in testing.
- [x] **Business-Oriented Metric Selection**: Learned that high Accuracy can be deceptive on imbalanced datasets, and why Precision takes priority over Recall in spam filtering.
- [x] **Model Serialization**: Mastered saving and loading model artifacts (`.joblib`) for reproducible real-time inference.

---

## 👥 Author & Acknowledgments

- **Internship Program**: Code-A-Nova Technology Initiative
- **Task**: Level 1(b) - Email Spam Detection (Month 3)
- **Repository**: [CODE-A-NOVA](https://github.com/)

