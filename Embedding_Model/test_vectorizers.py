import pandas as pd
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------
# PREPROCESSOR
# ----------------------------

def preprocessor(text):
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


# ----------------------------
# LOAD DATA
# ----------------------------

data = pd.read_csv(
    "data/perfume_query_dataset.csv"
)

data = data.drop_duplicates(
    subset="query"
)

data["query"] = (
    data["query"]
    .astype(str)
    .apply(preprocessor)
)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        data["query"],
        data["label"],
        test_size=0.2,
        random_state=48,
        stratify=data["label"]
    )
)

# =========================================================
# BAG OF WORDS TEST
# =========================================================

print("\n========== BAG OF WORDS ==========\n")

bow = CountVectorizer()

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

nb_bow = MultinomialNB(alpha=0.1)

nb_bow.fit(
    X_train_bow,
    y_train
)

predictions_bow = nb_bow.predict(
    X_test_bow
)

print(
    "Accuracy:",
    accuracy_score(y_test, predictions_bow)
)

print(
    "Precision:",
    precision_score(
        y_test,
        predictions_bow,
        average="weighted"
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        predictions_bow,
        average="weighted"
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        predictions_bow,
        average="weighted"
    )
)

# =========================================================
# TF-IDF TEST
# =========================================================

print("\n========== TF-IDF ==========\n")

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(
    X_train
)

X_test_tfidf = tfidf.transform(
    X_test
)

nb_tfidf = MultinomialNB(alpha=0.1)

nb_tfidf.fit(
    X_train_tfidf,
    y_train
)

predictions_tfidf = nb_tfidf.predict(
    X_test_tfidf
)

print(
    "Accuracy:",
    accuracy_score(y_test, predictions_tfidf)
)

print(
    "Precision:",
    precision_score(
        y_test,
        predictions_tfidf,
        average="weighted"
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        predictions_tfidf,
        average="weighted"
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        predictions_tfidf,
        average="weighted"
    )
)
print(classification_report(y_test, predictions_bow))