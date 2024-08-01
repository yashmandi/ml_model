import os
import re
import pdfplumber
import pytesseract
from PIL import Image
from io import BytesIO
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure Tesseract is installed and configured
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust this path based on your OS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    img = page.to_image()
                    page_text = pytesseract.image_to_string(img.original)
                text += page_text
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
    return text

def read_text_file(txt_path):
    try:
        with open(txt_path, "r", encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {txt_path}: {str(e)}")
        return ""

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    return text

# Process all .pdf and .txt files in a directory
directory = "../backend/uploads"  # Change this to your directory containing .pdf and .txt files
extracted_texts = []

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    print(f"Processing file: {filename}")
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif filename.endswith(".txt"):
        text = read_text_file(file_path)
    else:
        continue
    
    if text.strip():  # Only add non-empty texts
        extracted_texts.append(text)
        print(f"Extracted text from {filename} (length: {len(text)})")
    else:
        print(f"No text extracted from {filename}")

# Preprocess texts
preprocessed_texts = [preprocess_text(text) for text in extracted_texts]

if not preprocessed_texts:
    print("No texts to process. Check your input files and extraction process.")
    exit()

# Save preprocessed texts to a file
with open('preprocessed_texts.pkl', 'wb') as f:
    pickle.dump(preprocessed_texts, f)

# Load and inspect preprocessed texts
with open('preprocessed_texts.pkl', 'rb') as f:
    preprocessed_texts = pickle.load(f)
    print(f"Preprocessed texts sample: {preprocessed_texts[:2]}")  # Print the first two preprocessed texts

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(preprocessed_texts)

# Save the vectorizer and TF-IDF matrix
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_X.pkl', 'wb') as f:
    pickle.dump(X, f)

# Load and inspect TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    print(f"Feature names sample: {vectorizer.get_feature_names_out()[:10]}")  # Print the first 10 feature names

# Load and inspect TF-IDF matrix
with open('tfidf_X.pkl', 'rb') as f:
    X = pickle.load(f)
    print(f"TF-IDF matrix shape: {X.shape}")  # Print the shape of the matrix
    
    print(f"Number of files processed: {len(extracted_texts)}")
    print(f"Sample of extracted texts: {extracted_texts[:2]}")
    print(f"Sample of preprocessed texts: {preprocessed_texts[:2]}")