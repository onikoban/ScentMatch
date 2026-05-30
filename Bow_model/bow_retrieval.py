import nltk
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

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

all_families = set()
for families in df["families_list"]:
    for f in families:
        all_families.add(f)

family_vocab = sorted(all_families)

num_perfumes = len(df)
num_families = len(family_vocab)

perfume_family_matrix = np.zeros((num_perfumes, num_families), dtype=int)

for i, families in enumerate(df["families_list"]):
    for j, f in enumerate(family_vocab):
        if f in families:
            perfume_family_matrix[i, j] = 1

english_stopwords = set(stopwords.words("english"))

english_stopwords.discard("not")
english_stopwords.discard("no")

def preprocessor(text):
    text = str(text).lower()
    for punct in string.punctuation:
        text = text.replace(punct, "")
    tokenized_text = nltk.word_tokenize(text)
    clean_text = [word for word in tokenized_text if word not in english_stopwords]
    return clean_text

training_data = [
    ("bright and zesty", "citrus"),
    ("fresh lemon morning vibe", "citrus"),
    ("something citrusy", "citrus"),
    ("energetic summer scent", "citrus"),
    ("bergamot freshness", "citrus"),
    ("orange and grapefruit", "citrus"),
    ("sharp citrus opening", "citrus"),
    ("sunny zingy fragrance", "citrus"),
    ("lemonade on a hot day", "citrus"),
    ("refreshing citrus burst", "citrus"),
    ("tangy lime feeling", "citrus"),
    ("mandarin and yuzu", "citrus"),
    ("crisp italian cologne", "citrus"),
    ("juicy orange peel", "citrus"),
    ("sparkling fresh citrus", "citrus"),
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
training_data_bow = bow.fit_transform(X_train)
test_data_bow = bow.transform(X_test)

nb = MultinomialNB(alpha=1)
nb.fit(training_data_bow, y_train)

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
