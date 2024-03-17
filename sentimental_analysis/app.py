from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import string
import pandas as pd
import emoji
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

def clean_text(doc):
    doc = doc.replace("</br>", " ")
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = doc.lower()
    tokens = nltk.word_tokenize(doc)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))
model_folder = "resources"
model_file_path = os.path.join(model_folder, 'sentimental_analysis.pkl')
model_count = os.path.join(model_folder, 'count_vect.pkl')

with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
with open(model_count, 'rb') as f:
    count_vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = text.lower().strip()  
        text_vectorized = count_vectorizer.transform([cleaned_text])
        sentiment = model.predict(text_vectorized)[0]  
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
