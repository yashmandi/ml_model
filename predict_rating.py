import PyPDF2
import re
import os

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

def predict_rating(file_path):
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            text = file.read()
    else:
        raise ValueError("Unsupported file format")
    
    text = preprocess_text(text)
    X = vectorizer.transform([text])
    rating = model.predict(X)[0]
    return rating

# Example usage
file_path = 'resume_uploaded.pdf'  # Path to the uploaded file
rating = predict_rating(file_path)
print(f"Resume Rating: {rating}")