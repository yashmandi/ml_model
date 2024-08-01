from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import PyPDF2
import os

app = Flask(__name__)
CORS(app)  # Allow requests from any origin. Adjust if needed.

# Loading the model and vectorizer
model = joblib.load('final_saved_model.pkl')
vectorizer = joblib.load('final_saved_vectorizer.pkl')

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Corrected regex
    text = text.lower()
    return text

def predict_rating(text):
    text = preprocess_text(text)
    X = vectorizer.transform([text])
    rating = model.predict(X)[0]
    return round(rating * 2) / 2  # Round to nearest 0.5

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Log the extracted text
    print("Extracted text: ", text[:500])  # Print the first 500 characters for brevity

    rating = predict_rating(text)
    
    # Log the rating
    print("Predicted rating: ", rating)

    return jsonify({"rating": rating})

if __name__ == '__main__':
    # Set host to '0.0.0.0' to allow external connections, and use environment variable for port if set
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)