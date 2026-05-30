import nltk
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords


nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")



try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

df = pd.read_csv(
    "Bow_model/final_perfume_data_labeled.csv",
    encoding="utf-8",
    encoding_errors="replace",
    sep=";"
).fillna("")

df = df[df["num_families"] > 0]

df["families_list"] = df["Families"].apply(lambda s: s.split("|"))

english_stopwords = set(stopwords.words("english"))
english_stopwords.discard("not")
english_stopwords.discard("no")

def preprocessor(text):
    text = str(text).lower()
    for punct in string.punctuation:
        text = text.replace(punct, "")
    tokens = nltk.word_tokenize(text, language="english"))
    return [w for w in tokens if w not in english_stopwords]

training_data = [
    ("bright and zesty", "citrus"),
    ("fresh lemon morning vibe", "citrus"),
    ("something citrusy", "citrus"),
    ("energetic summer scent", "citrus"),
    ("bergamot freshness", "citrus"),
]

queries_df = pd.DataFrame(training_data, columns=["query", "label"])
queries_df["tokens"] = queries_df["query"].apply(preprocessor)

X_train, X_test, y_train, y_test = train_test_split(
    queries_df["tokens"],
    queries_df["label"],
    test_size=0.2,
    random_state=42,
    stratify=queries_df["label"]
)

bow = CountVectorizer(analyzer=lambda x: x)
X_train_bow = bow.fit_transform(X_train)

nb = MultinomialNB(alpha=1)
nb.fit(X_train_bow, y_train)

def bow_retrieval(query, top_k=3):

    tokens = preprocessor(query)
    query_bow = bow.transform([tokens])
    pred_label = nb.predict(query_bow)[0]

    similarity_scores = np.array([
        1 if pred_label in df.iloc[i]["families_list"] else 0
        for i in range(len(df))
    ])

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []

    for i in top_indices:
        perfume = df.iloc[i]
        results.append({
            "name": perfume["Name"],
            "notes": perfume["Notes"],
            "description": perfume["Description"],
            "score": float(similarity_scores[i])
        })

    return results
