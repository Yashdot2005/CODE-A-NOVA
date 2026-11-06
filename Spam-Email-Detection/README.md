# 📧 Email & SMS Spam Detection
### CODE-A-NOVA | Machine Learning Internship – Level 1(b)

A modular Machine Learning and Natural Language Processing (NLP) project that identifies spam and legitimate emails or SMS messages using text preprocessing, feature extraction, and supervised classification algorithms.

---

## 📌 Project Overview

Spam messages are a common problem in digital communication. They include unwanted advertisements, fraudulent offers, phishing attempts, and potentially harmful links.

This project implements an end-to-end Machine Learning pipeline to classify incoming text messages into two categories:

- **SPAM:** Unwanted, promotional, or potentially fraudulent messages.
- **HAM (NOT SPAM):** Genuine and legitimate messages.

### Key Features

- **Text Preprocessing:** Lowercasing, tokenization, stopword removal, punctuation removal, and Porter Stemming.
- **Feature Extraction:** TF-IDF and Bag-of-Words (BoW).
- **Model Training:** Multinomial Naive Bayes and Logistic Regression.
- **Model Comparison:** Evaluation using accuracy, precision, recall, and F1-score.
- **Prediction System:** Classifies new messages with probability-based confidence scores.
- **Modular Architecture:** Separate components for preprocessing, feature extraction, training, and prediction.

---

## 📂 Project Structure

```text
Email-SMS-Spam-Detection/
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
├── README.md
└── main.py
```

---

## 📊 Dataset Information

**Dataset:** SMS Spam Collection Dataset

**Source:** UCI Machine Learning Repository

Dataset Link: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

- Total Records: 5,572 messages
- Ham Messages: 4,825 (86.59%)
- Spam Messages: 747 (13.41%)

The dataset contains labeled SMS messages categorized as spam or legitimate messages.

A stratified train-test split is used to maintain the original class distribution during model evaluation.

---

## ⚙️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| NLTK | NLP preprocessing |
| Scikit-learn | Machine Learning algorithms |
| TF-IDF | Text feature extraction |
| Matplotlib / Seaborn | Data visualization |
| Joblib | Model serialization |
| Jupyter Notebook | Exploratory data analysis |

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Email-SMS-Spam-Detection
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

### Run the Complete Machine Learning Pipeline

```bash
python main.py
```

This executes the complete workflow:

1. Load the dataset.
2. Inspect and preprocess text data.
3. Extract numerical features using TF-IDF.
4. Train classification models.
5. Evaluate and compare model performance.
6. Save the trained model and vectorizer.
7. Perform sample predictions.

### Run Individual Modules

**Data Preprocessing**
```bash
python src/preprocess.py
```

**Feature Extraction**
```bash
python src/features.py
```

**Model Training**
```bash
python src/train.py
```

**Spam Prediction**
```bash
python src/predict.py
```

**Custom Message Prediction**
```bash
python src/predict.py --text "Congratulations! You won a free gift card. Claim now!"
```

---

## 🧠 Machine Learning Algorithms

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It is widely used in text classification because it works effectively with word-frequency and TF-IDF features.

### 2. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a message belonging to the spam or ham category.

### Feature Extraction Techniques

- **Bag-of-Words (BoW):** Represents text using word occurrence frequencies.
- **TF-IDF:** Assigns importance to words based on their frequency and uniqueness across documents.
- **Porter Stemmer:** Reduces words to their root forms to minimize vocabulary variations.

---

## 📈 Model Performance & Evaluation

The project uses an 80:20 stratified train-test split.

| Dataset | Records |
|---|---:|
| Training Set | 4,457 |
| Testing Set | 1,115 |
| Total | 5,572 |

### Model Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.86% | 100% | 76.51% | 86.69% |
| Logistic Regression | 96.86% | 100% | 76.51% | 86.69% |

*Note: These are the reported experimental results. Actual performance should be verified by running the training pipeline.*

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 966 | 0 |
| Spam | 35 | 114 |

### Evaluation Metrics

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Precision:** Measures how many messages predicted as spam are actually spam.
- **Recall:** Measures how many actual spam messages are correctly identified.
- **F1-Score:** Represents the harmonic mean of precision and recall.

---

## 🔍 Sample Predictions

**Example 1: Spam Message**

Input:
```text
CONGRATULATIONS! You won a FREE gift card. Claim now!
```

Expected Classification: SPAM

**Example 2: Legitimate Message**

Input:
```text
Hi, please send me the updated project presentation.
```

Expected Classification: NOT SPAM (HAM)

**Example 3: Legitimate Message**

Input:
```text
Are we still meeting for coffee this afternoon?
```

Expected Classification: NOT SPAM (HAM)

---

## 🎯 Applications

- Email spam filtering.
- SMS spam detection.
- Fraudulent message identification.
- Unwanted promotional message filtering.
- Basic phishing message screening.

---

## 📚 What I Learned

- Understanding Natural Language Processing fundamentals.
- Implementing text preprocessing techniques.
- Applying TF-IDF and Bag-of-Words feature extraction.
- Training supervised Machine Learning classification models.
- Comparing Naive Bayes and Logistic Regression.
- Evaluating classification models using performance metrics.
- Understanding class imbalance and stratified sampling.
- Saving and loading trained models using Joblib.
- Building a reusable prediction pipeline.

---

## 👨‍💻 Author & Acknowledgments

**Internship Program:** CODE-A-NOVA Technology Initiative  
**Task:** Level 1(b) – Email & SMS Spam Detection  
**Domain:** Machine Learning & Natural Language Processing  
**Programming Language:** Python

---

*Developed as part of the CODE-A-NOVA Machine Learning Internship Program.*
