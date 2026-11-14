import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF

# 1. Defining the dataset
documents = [
    "Football is a popular sport played with a ball on a field. Soccer is the most common name for football in many countries.",
    "Cricket is played with a bat and a ball. A leather ball is used, and it is popular in many regions."
]

# 2. Display setting
pd.set_option('display.max_colwidth', 100)
data = pd.DataFrame(documents, columns=['text'])

# 3. Preprocessing function (Tokenization, Stopwords, Simple Lemmatization)
def preprocess(text: str) -> str:
    tokens = re.findall(r'[a-z]+', text.lower())
    filtered = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    normalized = []
    for word in filtered:
        if word.endswith('ies') and len(word) > 4:
            normalized.append(word[:-3] + 'y')
        elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            normalized.append(word[:-1])
        else:
            normalized.append(word)
    return " ".join(normalized)

data['processed'] = data['text'].apply(preprocess)

# ==========================================
# APPROACH 1: LDA (Probabilistic + CountVectorizer)
# ==========================================
count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(data['processed'])

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(count_matrix)

print("--- LDA Topics (CountVectorizer) ---")
for index, topic in enumerate(lda.components_):
    top_words = [count_vect.get_feature_names_out()[i] for i in topic.argsort()[-5:][::-1]]
    print(f"Topic {index + 1}: {' '.join(top_words)}")

# ==========================================
# APPROACH 2: NMF (Linear Algebra + TfidfVectorizer)
# ==========================================
tfidf_vect = TfidfVectorizer()
tfidf_matrix = tfidf_vect.fit_transform(data['processed'])

nmf = NMF(n_components=2, random_state=42)
nmf.fit(tfidf_matrix)

print("\n--- NMF Topics (TfidfVectorizer) ---")
for index, topic in enumerate(nmf.components_):
    top_words = [tfidf_vect.get_feature_names_out()[i] for i in topic.argsort()[-5:][::-1]]
    print(f"Topic {index + 1}: {' '.join(top_words)}")