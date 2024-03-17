from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os
from text_prep import clean  # Import the clean function from the cleaning module

app = Flask(__name__)

# Set the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load sentimental analysis model from the pickle file
with open('resources/sentimental_analysis.pkl', 'rb') as f:
    model = pickle.load(f)

with open('resources/count_vect.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean(text)  # Clean the text using the clean function
        text_vectorized = count_vectorizer.transform([cleaned_text])
        sentiment = model.predict(text_vectorized)[0]
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
