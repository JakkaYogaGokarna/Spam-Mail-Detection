# backend/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import chardet

# --- Step 1: Detect file encoding ---
csv_file = 'data/spam.csv'

with open(csv_file, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

# --- Step 2: Load CSV ---
df = pd.read_csv(csv_file, encoding=encoding)
print("CSV loaded successfully")
print(df.head())

# --- Step 3: Prepare data ---
# Adjust column names according to your CSV
# For example, 'label' = spam/ham, 'message' = email content
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
  # rename columns if necessary
X = df['message']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Build ML pipeline ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),      # Convert text to features
    ('clf', MultinomialNB())           # Naive Bayes classifier
])

# --- Step 5: Train model ---
pipeline.fit(X_train, y_train)
print("Model trained successfully")

# --- Step 6: Evaluate ---
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- Step 7: Save trained model ---
model_file = 'backend/model/spam_pipeline.joblib'
joblib.dump(pipeline, model_file)
print(f"Trained model saved to {model_file}")
