import joblib

# Save classifier
joblib.dump(nb_model_w2v, "w2v_nb_model.pkl")

# Save Word2Vec model
w2v_model.save("w2v_model.model")

import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

# Download NLTK data (first time only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# Load saved models
# ===============================
nb_model_w2v = joblib.load("w2v_nb_model.pkl")
w2v_model = Word2Vec.load("w2v_model.model")

# ===============================
# Preprocessing function
# ===============================
def preprocess(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z0-9 ]+', '', text)  # remove special chars
    text = " ".join([w for w in text.split() if w not in stopwords.words('english')])
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = BeautifulSoup(text, 'lxml').get_text()  # remove HTML
    text = " ".join(text.split())  # remove extra spaces
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# ===============================
# Convert sentence → vector
# ===============================
def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[doc], axis=0)

# ===============================
# Prediction function
# ===============================
def predict_sentence(sentence):
    clean = preprocess(sentence)
    tokens = word_tokenize(clean)
    vector = document_vector(tokens, w2v_model).reshape(1, -1)
    prediction = nb_model_w2v.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    text = input("Enter a sentence for Sentiment Analysis:")
    print("Predicted Sentiment:", predict_sentence(text))
