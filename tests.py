import joblib

# Load the vectorizer (if you have the original object in memory)
vectorizer = joblib.load('model/vectorizer.pkl')

# Save again to ensure it's correctly written to disk
joblib.dump(vectorizer, 'model/vectorizer.pkl')

# Test loading it again to verify
loaded_vectorizer = joblib.load('model/vectorizer.pkl')
print("Vectorizer loaded successfully:", loaded_vectorizer)
