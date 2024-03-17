from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)
import os
from sklearn.feature_extraction.text import CountVectorizer
from text_prep import clean
# Get the absolute path to the current script
script_path = os.path.abspath(__file__)

# Set the working directory to the script's directory
os.chdir(os.path.dirname(script_path))

# Specify the path to the model folder relative to the script's directory
model_folder = "resources"

# Construct the full path to the model file
model_file_path = os.path.join(model_folder, 'sentimental_analysis.pkl')
model_count = os.path.join(model_folder, 'count_vect.pkl')
# Load sentimental analysis model from the pickle file
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
with open(model_count, 'rb') as f:
    count_vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = text.lower().strip()  # Assuming simple text cleaning
        text_vectorized = count_vectorizer.transform([cleaned_text])
        sentiment = model.predict(text_vectorized)[0]  # Get sentiment prediction
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)