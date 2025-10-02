import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# 1. Load dataset
df = pd.read_csv("twitter_disaster (1).csv")
X = df["text"]
y = df["target"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, solver="liblinear"))
])

# 4. Train model
pipeline.fit(X_train, y_train)

# 5. Test accuracy
print("✅ Train Accuracy:", pipeline.score(X_train, y_train))
print("✅ Test Accuracy:", pipeline.score(X_test, y_test))

# 6. Save pipeline
with open("tweet_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("🎉 Model saved successfully as tweet_pipeline.pkl")
