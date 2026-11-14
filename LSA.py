# lsa_tfidf.py

# Step 1: Import the specific tools we need
# TfidfVectorizer: Converts text to numbers based on word importance
# TruncatedSVD: The math engine that performs LSA
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Step 2: Our raw data (10 independent text documents)
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

print("========== LSA WITH TF-IDF ==========\n")

# Step 3: Text to Numbers (TF-IDF Matrix)
# We initialize the vectorizer and tell it to ignore common English stop words
vectorizer = TfidfVectorizer(stop_words='english')

# fit_transform does two things:
# 1. fit: learns the vocabulary (finds all unique words)
# 2. transform: calculates the TF-IDF score for each word in each document
# The result 'tfidf_matrix' is a grid of numbers (10 rows for docs, N columns for words)
tfidf_matrix = vectorizer.fit_transform(documents)

# We extract the actual words so we can print them later instead of column numbers
words = vectorizer.get_feature_names_out()

# Step 4: The Core LSA Math (TruncatedSVD)
# n_components=3 tells the math to find 3 hidden topics (piles)
# random_state=42 ensures the math gives the exact same result every time you run it
lsa_model = TruncatedSVD(n_components=3, random_state=42)

# fit_transform here does two things:
# 1. fit: performs the Singular Value Decomposition (matrix math) to find the 3 topics
# 2. transform: calculates how much each of the 10 documents belongs to each of the 3 topics
# 'doc_topic_matrix' is a 10x3 grid of numbers (scores can be negative or positive)
doc_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

# Step 5: Extracting the Top Words for each Topic
# 'lsa_model.components_' holds a 3xN grid. It tells us the weight of every word for every topic.
for topic_idx, topic_weights in enumerate(lsa_model.components_):
    # argsort() sorts the weights from lowest to highest and returns their index positions.
    # [:-5:-1] is a Python trick to grab the top 4 highest weights and reverse them (highest first).
    top_word_indices = topic_weights.argsort()[:-5:-1]
    
    # We use the indices to look up the actual English words from our 'words' array
    top_words = [words[i] for i in top_word_indices]
    
    print(f"Topic {topic_idx + 1} Top Words: {top_words}")

# Step 6: Assigning Documents to Piles
print("\nDocument Assignments:")
for i, doc_weights in enumerate(doc_topic_matrix):
    # argmax() looks at the 3 scores for a document and returns the index (0, 1, or 2) of the highest score.
    # We add +1 so it prints as Pile 1, 2, or 3 instead of Pile 0, 1, or 2.
    assigned_pile = doc_weights.argmax() + 1
    
    # We print the document number, the first word of the document (just to verify), and its assigned pile
    first_word = documents[i].split()[0]
    print(f"Document {i+1} (starts with '{first_word}') -> Assigned to Pile {assigned_pile}")