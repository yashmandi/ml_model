import joblib
import pickle

# Load the trained model and vectorizer
model = joblib.load('final_saved_model.pkl')
vectorizer = joblib.load('final_saved_vectorizer.pkl')