# 📘 Sentiment Analysis on Kindle Reviews

A reproducible sentiment-analysis project that classifies Kindle product reviews into **Positive (1)** or **Negative (0)**. This repository demonstrates classical text representations (BOW, TF-IDF) and embedding-based representations (Word2Vec — custom & pretrained) and includes an inference script (`predict.py`) to classify new sentences.

---

## 🔍 Project overview

This project loads a dataset of Kindle reviews, converts numerical ratings into binary sentiment labels (negative for rating < 3, positive for rating ≥ 3), balances the classes, applies a cleaning and normalization pipeline, trains multiple models using different text representations, evaluates them, and exposes an inference-only script for quick predictions.

Key goals:

* Provide clear, reproducible baselines using classical and embedding-based representations.
* Offer an easy-to-use `predict.py` tool for single-sentence inference.
* Make the project easy to reproduce and extend.

---

## 📁 Repository structure (recommended)

```
Sentiment-Analysis/
├─ implementation.ipynb       # Notebook: data prep, training, evaluation (your main work)
├─ predict.py                 # Prediction-only script (loads saved models)
├─ requirements.txt           # Python dependencies
├─ all_kindle_review.csv      # Dataset (not included in repo)
├─ w2v_nb_model.pkl           # Saved Naive Bayes classifier (after training)
├─ w2v_model.model           # Saved gensim Word2Vec model OR use GoogleNews .bin.gz
├─ README.md                  # This file
```

> **Note:** `all_kindle_review.csv` is not included in this repository. Place it in the repo root before running the training notebook.

---

## ⚙️ Installation

**1. Create & activate a virtual environment (recommended):**

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
pandas
numpy
scikit-learn
gensim
nltk
beautifulsoup4
joblib
```

**3. Download NLTK data (run once):**

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

**4. (Optional) Google pretrained Word2Vec**

* If you plan to use GoogleNews pretrained embeddings, download `GoogleNews-vectors-negative300.bin.gz` and place it in the project root. The file is \~1.5 GB and requires substantial RAM to load.

---

## 🧹 Data & Preprocessing

**Data used**

* `all_kindle_review.csv` — columns used: `reviewText`, `rating`.

**Labeling rule**

* `rating < 3` → `0` (Negative)
* `rating >= 3` → `1` (Positive)

**Balancing**

* The notebook downsamples the majority class to match minority class size for a balanced training set.

**Preprocessing pipeline**

1. Cast to string and lowercase
2. Remove non-alphanumeric characters (preserve spaces)
3. Remove URLs and HTML tags
4. Remove stopwords (`nltk.corpus.stopwords`)
5. Lemmatize with `WordNetLemmatizer`
6. Tokenize with `nltk.word_tokenize`

This normalized text works reliably with `CountVectorizer`, `TfidfVectorizer`, and Word2Vec.

---

## 🧠 Models implemented

### 1. Bag-of-Words (CountVectorizer) + GaussianNB

* Simple and fast baseline. Converts each review into a sparse counts vector.

### 2. TF-IDF (TfidfVectorizer) + GaussianNB

* Reweights token counts by inverse document frequency.

### 3. Word2Vec embeddings + GaussianNB

* Two options supported:

  * **Custom Word2Vec** trained on the training split using `gensim.models.Word2Vec`.
  * **Pretrained GoogleNews** Word2Vec (`KeyedVectors`) loaded from `GoogleNews-vectors-negative300.bin.gz`.
* Each document is represented by the **mean** of its word vectors (mean pooling).
* Classifier: Gaussian Naive Bayes trained on dense vectors.

> **Why mean pooling?** Mean pooling is simple, fast, and an effective baseline. For better results later you may use TF-IDF weighted pooling or sentence encoders (e.g., Sentence-BERT).

---

## 🏋️ Training (in `implementation.ipynb`)

A standard training flow in the notebook includes:

1. Load CSV and filter columns.
2. Convert ratings into binary sentiment and downsample for balance.
3. Preprocess text with the pipeline above.
4. Train/test split: `test_size=0.2`, `random_state=42`.
5. Train models:

   * `CountVectorizer` + `GaussianNB`
   * `TfidfVectorizer` + `GaussianNB`
   * Word2Vec (custom or Google) → mean vectors → `GaussianNB`
6. Evaluate using `accuracy_score`, `confusion_matrix`, and `classification_report`.

**Saving artifacts** (required for `predict.py`):






