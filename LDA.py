# lda_tfidf.py

# Step 1: Import the specific tools we need
# Note: We are using TfidfVectorizer as you requested, though LDA technically prefers raw counts.
# LatentDirichletAllocation: The probability engine that performs LDA
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 2: Our raw data (Same 10 documents)
documents = [
    "apple banana orange fruit healthy",    # 0: Food
    "apple orange sweet fruit juice",       # 1: Food
    "banana fruit healthy eat diet",        # 2: Food
    "football soccer goal team win",        # 3: Sports
    "soccer team score match champion",     # 4: Sports
    "football win goal league",             # 5: Sports
    "bank money invest stock finance",      # 6: Finance
    "money stock market trade invest",      # 7: Finance
    "bank invest finance loan economy",     # 8: Finance
    "orange apple fruit sweet dessert"      # 9: Food
]

print("========== LDA WITH TF-IDF ==========\n")

# Step 3: Text to Numbers (TF-IDF Matrix)
# We initialize the vectorizer and tell it to ignore common English stop words
vectorizer = TfidfVectorizer(stop_words='english')

# fit_transform learns the vocabulary and calculates the TF-IDF score for each word.
tfidf_matrix = vectorizer.fit_transform(documents)

# We extract the actual words for later printing
words = vectorizer.get_feature_names_out()

# Step 4: The Core LDA Math
# n_components=3 tells the math to find 3 hidden topics (piles)
# random_state=42 ensures the probability math gives the same result every time
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)

# fit_transform here does two things:
# 1. fit: runs the probabilistic math to figure out word probabilities for each topic
# 2. transform: calculates the probability that each document belongs to each topic
# 'doc_topic_matrix' is a 10x3 grid of probabilities (they will add up to 1.0, e.g., [0.8, 0.1, 0.1])
doc_topic_matrix = lda_model.fit_transform(tfidf_matrix)

# Step 5: Extracting the Top Words for each Topic
# 'lda_model.components_' holds the probability weights of words for each topic
for topic_idx, topic_weights in enumerate(lda_model.components_):
    # argsort() sorts the weights and returns their indices.
    # [:-5:-1] grabs the top 4 highest weights (highest first).
    top_word_indices = topic_weights.argsort()[:-5:-1]
    
    # Look up the actual words using the indices
    top_words = [words[i] for i in top_word_indices]
    
    print(f"Topic {topic_idx + 1} Top Words: {top_words}")

# Step 6: Assigning Documents to Piles
print("\nDocument Assignments:")
for i, doc_weights in enumerate(doc_topic_matrix):
    # argmax() looks at the 3 probabilities for a document and returns the index of the highest one.
    # We add +1 so it prints as Pile 1, 2, or 3.
    assigned_pile = doc_weights.argmax() + 1
    
    first_word = documents[i].split()[0]
    print(f"Document {i+1} (starts with '{first_word}') -> Assigned to Pile {assigned_pile}")