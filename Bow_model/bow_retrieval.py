import nltk
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

import nltk

nltk.download('punkt')
nltk.download('stopwords')

try:
    df = pd.read_csv("Bow_model/final_perfume_data_labeled.csv", encoding='utf-8', encoding_errors='replace', sep=";",)
except Exception:
    df = pd.read_csv("Bow_model/final_perfume_data_labeled.csv", encoding='latin1',sep=";",)

df = df[df["num_families"] > 0]

df["families_list"] = df["Families"].apply(lambda s: s.split("|"))

all_families = set()
for families in df["families_list"]:
  for f in families:
    all_families.add(f)

family_vocab = sorted(all_families)

num_perfumes = len(df)
num_families =len(family_vocab)

perfume_family_matrix = np.zeros((num_perfumes, num_families), dtype=int)

for i, families in enumerate(df["families_list"]):
  for j, f in enumerate(family_vocab):
    if f in families:
      perfume_family_matrix[i, j] = 1


english_stopwords = set(
    stopwords.words("english")
)

english_stopwords.remove("not")
english_stopwords.remove("no")

def preprocessor(text):
  text = text.lower()
  for punct in string.punctuation:
    text = text.replace(punct,"")
  tokenized_text = nltk.word_tokenize(text)
  clean_text = [word for word in tokenized_text if word not in english_stopwords]
  return clean_text

training_data = [
    # ---------- citrus ----------
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
    # ---------- floral ----------
    ("flowery and feminine", "floral"),
    ("rose garden in bloom", "floral"),
    ("delicate floral bouquet", "floral"),
    ("romantic flower scent", "floral"),
    ("spring florals", "floral"),
    ("elegant floral perfume", "floral"),
    ("feminine and pretty", "floral"),
    ("rose perfume", "floral"),
    ("a bouquet of flowers", "floral"),
    ("soft pink florals", "floral"),
    ("magnolia and freesia", "floral"),
    ("bulgarian rose extract", "floral"),
    ("blooming garden", "floral"),
    ("iris and violet", "floral"),
    ("geranium and peony", "floral"),
    # ---------- white_floral ----------
    ("heady white flowers", "white_floral"),
    ("intoxicating jasmine", "white_floral"),
    ("tuberose and gardenia", "white_floral"),
    ("narcotic white florals", "white_floral"),
    ("creamy white blooms", "white_floral"),
    ("orange blossom magic", "white_floral"),
    ("wedding bouquet vibe", "white_floral"),
    ("lush white floral", "white_floral"),
    ("neroli and jasmine", "white_floral"),
    ("sambac jasmine", "white_floral"),
    ("ylang ylang scent", "white_floral"),
    ("frangipani tropical floral", "white_floral"),
    ("honeysuckle and gardenia", "white_floral"),
    ("muguet lily of the valley", "white_floral"),
    ("heavy white flower scent", "white_floral"),
    # ---------- woody ----------
    ("warm woody scent", "woody"),
    ("smells like a forest", "woody"),
    ("earthy and grounding", "woody"),
    ("sandalwood and cedar", "woody"),
    ("dry woody base", "woody"),
    ("deep wooden notes", "woody"),
    ("vetiver and patchouli", "woody"),
    ("tree bark and forest floor", "woody"),
    ("masculine woody fragrance", "woody"),
    ("smooth sandalwood", "woody"),
    ("cedar and cypress", "woody"),
    ("woody with vetiver", "woody"),
    ("pine and fir", "woody"),
    ("rich rosewood", "woody"),
    ("dark mysterious wood", "woody"),
    # ---------- oud_smoky ----------
    ("smoky oud", "oud_smoky"),
    ("incense and frankincense", "oud_smoky"),
    ("middle eastern oud", "oud_smoky"),
    ("deep smoky scent", "oud_smoky"),
    ("smells like a temple", "oud_smoky"),
    ("myrrh and incense", "oud_smoky"),
    ("burning church candles", "oud_smoky"),
    ("campfire smoke", "oud_smoky"),
    ("dark mysterious smoke", "oud_smoky"),
    ("agarwood scent", "oud_smoky"),
    ("pure oud", "oud_smoky"),
    ("birch tar smoke", "oud_smoky"),
    ("olibanum and myrrh", "oud_smoky"),
    ("smoke and resin", "oud_smoky"),
    ("dense smoky base", "oud_smoky"),
    # ---------- amber_resinous ----------
    ("warm amber", "amber_resinous"),
    ("resinous and golden", "amber_resinous"),
    ("labdanum and benzoin", "amber_resinous"),
    ("rich amber base", "amber_resinous"),
    ("honeyed resin", "amber_resinous"),
    ("ambery cocoon", "amber_resinous"),
    ("dense warm amber", "amber_resinous"),
    ("opoponax and styrax", "amber_resinous"),
    ("balsamic and golden", "amber_resinous"),
    ("ambroxan note", "amber_resinous"),
    ("glowing amber warmth", "amber_resinous"),
    ("peru balsam resin", "amber_resinous"),
    ("amber and resin", "amber_resinous"),
    ("honey amber base", "amber_resinous"),
    ("luxurious amber feel", "amber_resinous"),
    # ---------- sweet_gourmand ----------
    ("sweet vanilla", "sweet_gourmand"),
    ("dessert-like cozy", "sweet_gourmand"),
    ("smells like a bakery", "sweet_gourmand"),
    ("caramel and vanilla", "sweet_gourmand"),
    ("warm cookie scent", "sweet_gourmand"),
    ("creamy gourmand", "sweet_gourmand"),
    ("tonka and praline", "sweet_gourmand"),
    ("honey and milk", "sweet_gourmand"),
    ("chocolate cocoa scent", "sweet_gourmand"),
    ("sugary comforting", "sweet_gourmand"),
    ("vanilla bean", "sweet_gourmand"),
    ("coconut and almond", "sweet_gourmand"),
    ("coffee and caramel", "sweet_gourmand"),
    ("marshmallow soft", "sweet_gourmand"),
    ("rum and sugar", "sweet_gourmand"),
    # ---------- fruity ----------
    ("juicy peach", "fruity"),
    ("fresh apple", "fruity"),
    ("berry sweet", "fruity"),
    ("raspberry and plum", "fruity"),
    ("tropical pineapple", "fruity"),
    ("fruity and playful", "fruity"),
    ("cherry and apricot", "fruity"),
    ("lychee fragrance", "fruity"),
    ("summer fruits cocktail", "fruity"),
    ("blackcurrant note", "fruity"),
    ("ripe fig", "fruity"),
    ("pear and apple", "fruity"),
    ("mango and passionfruit", "fruity"),
    ("strawberry and cream", "fruity"),
    ("tropical fruit blend", "fruity"),
    # ---------- spicy ----------
    ("warm spices", "spicy"),
    ("cinnamon and clove", "spicy"),
    ("pepper and cardamom", "spicy"),
    ("spicy oriental", "spicy"),
    ("ginger and saffron", "spicy"),
    ("indian spice market", "spicy"),
    ("tingly pepper note", "spicy"),
    ("exotic spices", "spicy"),
    ("nutmeg and clove", "spicy"),
    ("pink pepper spice", "spicy"),
    ("saffron and cumin", "spicy"),
    ("hot warm spices", "spicy"),
    ("fiery spicy scent", "spicy"),
    ("chai spice blend", "spicy"),
    ("peppery and warm", "spicy"),
    # ---------- aromatic_herbal ----------
    ("lavender fields", "aromatic_herbal"),
    ("mint and basil", "aromatic_herbal"),
    ("aromatic provence", "aromatic_herbal"),
    ("sage and rosemary", "aromatic_herbal"),
    ("calming lavender", "aromatic_herbal"),
    ("fresh herbs scent", "aromatic_herbal"),
    ("thyme and rosemary", "aromatic_herbal"),
    ("clary sage cologne", "aromatic_herbal"),
    ("eucalyptus and mint", "aromatic_herbal"),
    ("chamomile and lavender", "aromatic_herbal"),
    ("lavender and herbs", "aromatic_herbal"),
    ("herbal cologne", "aromatic_herbal"),
    ("anise and licorice", "aromatic_herbal"),
    ("tarragon and basil", "aromatic_herbal"),
    ("fresh herb garden", "aromatic_herbal"),
    # ---------- green ----------
    ("fresh cut grass", "green"),
    ("leafy green scent", "green"),
    ("oakmoss base", "green"),
    ("mossy forest floor", "green"),
    ("violet leaf freshness", "green"),
    ("green tea note", "green"),
    ("spring green outdoors", "green"),
    ("galbanum and moss", "green"),
    ("crushed leaves", "green"),
    ("dewy green plants", "green"),
    ("bamboo and tea", "green"),
    ("freshly mowed lawn", "green"),
    ("wet green leaves", "green"),
    ("tomato leaf", "green"),
    ("green stem snap", "green"),
    # ---------- leather ----------
    ("leathery scent", "leather"),
    ("smells like leather", "leather"),
    ("tobacco and leather", "leather"),
    ("horsey leather note", "leather"),
    ("old library leather", "leather"),
    ("leather jacket vibe", "leather"),
    ("dark suede", "leather"),
    ("rich pipe tobacco", "leather"),
    ("animalic leather", "leather"),
    ("birch tar leather", "leather"),
    ("saddle leather", "leather"),
    ("leather and tobacco", "leather"),
    ("smoke and leather", "leather"),
    ("worn leather smell", "leather"),
    ("luxurious leather", "leather"),
    # ---------- musk ----------
    ("clean musk", "musk"),
    ("skin scent musk", "musk"),
    ("soft musky base", "musk"),
    ("ambergris and musk", "musk"),
    ("cozy musky", "musk"),
    ("skin-like musk", "musk"),
    ("white musk freshness", "musk"),
    ("warm sensual musk", "musk"),
    ("ambrette and musk", "musk"),
    ("smells like clean skin", "musk"),
    ("cashmere musk", "musk"),
    ("civet animalic", "musk"),
    ("second skin musk", "musk"),
    ("intimate musky scent", "musk"),
    ("soft second skin", "musk"),
    # ---------- powdery ----------
    ("powdery iris", "powdery"),
    ("soft powder feel", "powdery"),
    ("baby powder scent", "powdery"),
    ("aldehydic powder", "powdery"),
    ("vintage powdery", "powdery"),
    ("makeup powder note", "powdery"),
    ("heliotrope powder", "powdery"),
    ("orris and iris", "powdery"),
    ("almond powder", "powdery"),
    ("soft powdery dust", "powdery"),
    ("violet powder", "powdery"),
    ("powdered sugar feel", "powdery"),
    ("talc and powder", "powdery"),
    ("chypre powder", "powdery"),
    ("lipstick powder", "powdery"),
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

print(f"Total data: {len(queries_df)}")
print(f"Training:{len(X_train)}")
print(f"Test: {len(X_test)}")

bow = CountVectorizer(analyzer=lambda x: x)
training_data_bow =bow.fit_transform(X_train)
test_data_bow = bow.transform(X_test)

nb = MultinomialNB(alpha=1)
nb.fit(training_data_bow, y_train)

def bow_retrieval(query, top_k=3):

    tokens = preprocessor(query)

    query_bow = bow.transform([tokens])

    probs = nb.predict_proba(query_bow)

    similarity_scores = cosine_similarity(
        probs,
        perfume_family_matrix
    )[0]

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

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

if __name__ == "__main__":

    query = input("Enter query: ")

    results = bow_retrieval(query)

    print(results)