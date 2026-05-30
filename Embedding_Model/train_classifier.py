import pandas as pd
import pickle
import os
import re

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def preprocessor(text):
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

data = pd.read_csv(
    "data/perfume_query_dataset.csv"
)

data = data.drop_duplicates(
    subset="query"
)

data["query"] = data["query"].apply(
    preprocessor
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        data["query"],
        data["label"],
        random_state=48,
        test_size=0.2,
        stratify=data["label"]
    )
)

bow = CountVectorizer()

X_train_bow = bow.fit_transform(X_train)

nb = MultinomialNB(alpha=0.1)

nb.fit(X_train_bow, y_train)

test_data = bow.transform(X_test)

predictions = nb.predict(test_data)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

os.makedirs("models", exist_ok=True)

with open("models/bow.pkl", "wb") as f:
    pickle.dump(bow, f)

with open("models/nb.pkl", "wb") as f:
    pickle.dump(nb, f)

print("Models saved!")