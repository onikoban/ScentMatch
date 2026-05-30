import re
import pickle
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def preprocessor(text):
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

with open(os.path.join(MODELS_DIR, "bow.pkl"), "rb") as f:
     bow = pickle.load(f)

with open(os.path.join(MODELS_DIR, "nb.pkl"), "rb") as f:
     nb = pickle.load(f)

def get_dynamic_weights(query):

    query = preprocessor(query)

    query_vector = bow.transform([query])

    probabilities = nb.predict_proba(query_vector)

    class_probs = dict(
        zip(nb.classes_, probabilities[0])
    )

    notes_weight = (
        class_probs["NOTE_BASED"]
        + class_probs["MIXED"] * 0.5
    )

    desc_weight = (
        class_probs["DESCRIPTION_BASED"]
        + class_probs["MIXED"] * 0.5
    )

    total = notes_weight + desc_weight

    notes_weight /= total
    desc_weight /= total

    return desc_weight, notes_weight