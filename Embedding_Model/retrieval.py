import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Embedding_Model.classifier import (get_dynamic_weights, preprocessor)
import pickle
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "final_perfume_data.csv"
)



MODELS_DIR = os.path.join(BASE_DIR, "models")


try:

    df = pd.read_csv(

        DATA_PATH,

        encoding="utf-8",

        encoding_errors="replace"

    )

except Exception:

    df = pd.read_csv(

        DATA_PATH,

        encoding="latin1"

    )



df["Description"] = df["Description"].apply(preprocessor)

descriptions = df["Description"].tolist()
df["Notes"] = df["Notes"].fillna("")
df["Notes"] = df["Notes"].apply(preprocessor)
note = df["Notes"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(
    os.path.join(
        MODELS_DIR,
        "description_embeddings.pkl"
    )
):

    with open(
    os.path.join(
        MODELS_DIR,
        "description_embeddings.pkl"
    ),
    "rb"
) as f:

        description_embeddings = pickle.load(f)

else:

    description_embeddings = model.encode(
        descriptions
    )

    with open(

    os.path.join(

        MODELS_DIR,

        "description_embeddings.pkl"

    ),

    "wb"

) as f:

        pickle.dump(
            description_embeddings,
            f
        )
        

if os.path.exists(
    os.path.join(
    MODELS_DIR,
    "note_embeddings.pkl"
)):

    with open(
    os.path.join(
        MODELS_DIR,
        "note_embeddings.pkl"
    ),
    "rb"
)   as f:

        note_embeddings = pickle.load(f)

else:

    note_embeddings = model.encode(note)

    with open(
    os.path.join(
        MODELS_DIR,
        "note_embeddings.pkl"
    ),
    "wb"
    ) as f:

        pickle.dump(
            note_embeddings,
            f
        )




def description_embedder(query):
  query_embedding = model.encode([query])
  similarities = cosine_similarity(query_embedding, description_embeddings)
  similarity_scores = similarities[0]
  return similarity_scores



def notes_embedder(query):
  query_embedding = model.encode([query])
  similarities = cosine_similarity(query_embedding, note_embeddings)
  similarity_scores = similarities[0]
  return similarity_scores


def Retrieval(query):
    desc_weight, notes_weight = get_dynamic_weights(query)
    print(f"Description weight: {desc_weight}")
    print(f"Notes weight: {notes_weight}")
    description_scores = description_embedder(query)
    notes_scores = notes_embedder(query)
    final_scores = (description_scores * desc_weight+ notes_scores * notes_weight)
    top_indices = np.argsort(final_scores)[::-1][:3]
    results = []
    for i in top_indices:
        perfume = df.iloc[i]
        score_percentage = round(final_scores[i] * 100, 2)
        results.append({"name": perfume["Name"],"notes": perfume["Notes"],"description": perfume["Description"],"score": final_scores[i],"ux_score": score_percentage})
    print(results)
    return results

if __name__ == "__main__":

    query = input("Enter query: ")

    results = Retrieval(query)

