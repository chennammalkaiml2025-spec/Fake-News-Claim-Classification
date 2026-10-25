
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Prepare data
data = data.dropna(subset=["text", "label"])
data["label"] = data["label"].str.strip().str.title()
data = data[data["label"].isin(["Real", "Fake"])]

X = data["text"].astype(str)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create machine learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Fake News Classification System")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Predict new claims
while True:
    news = input("\nEnter news claim (or type exit): ")

    if news.strip().lower() == "exit":
        break

    result = model.predict([news])[0]
    print("Predicted Label:", result)
    print("Note: Prediction is not fact verification.")
  
