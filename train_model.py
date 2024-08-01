from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pickle
import numpy as np

# Load preprocessed texts and ratings
print("Loading data...")
with open('tfidf_X.pkl', 'rb') as f:
    X = pickle.load(f)

with open('ratings.pkl', 'rb') as f:
    y = pickle.load(f)

print(f"X shape: {X.shape}")
print(f"y shape: {np.array(y).shape}")

if X.shape[0] != len(y):
    print("Error: Number of samples in X and y do not match.")
    exit()

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor
print("Training model...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Make predictions and evaluate
print("Evaluating model...")
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model mean squared error: {mse}")

# Save the model and vectorizer
print("Saving model and vectorizer...")
joblib.dump(reg, 'final_saved_model.pkl')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
joblib.dump(vectorizer, 'final_saved_vectorizer.pkl')

print("Done!")