# word2vec_demo.py

# --- WHAT IS WORD2VEC? ---
# Word2Vec is an algorithm that turns words into dense lists of numbers (vectors).
# It is used to find the mathematical relationship between words.
# Example: It can learn that "King" is to "Man" as "Queen" is to "Woman".

# --- WHY USE SKIP-GRAM? ---
# Skip-gram (sg=1) is a Word2Vec architecture that uses a center word to predict 
# the surrounding context words. It is great for small datasets and rare words.

from gensim.models import Word2Vec

# 1. Our Toy Dataset
# A real dataset would be millions of sentences. But for learning, 
# we use a tiny dataset where the context makes sense.
# Notice: Gensim expects the text to be tokenized (split into lists of words).
sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "kingdom"],
    ["the", "man", "is", "a", "male", "human"],
    ["the", "woman", "is", "a", "female", "human"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "puppy", "chased", "the", "kitten"],
    ["the", "boy", "likes", "the", "girl"],
    ["the", "prince", "likes", "the", "princess"]
]

print("Training the Word2Vec model...\n")

# 2. Initialize the Model
# --- EXPLANATION OF PARAMETERS ---
# sentences: The data we want to learn from.
# vector_size: How many numbers we want to represent each word (e.g., 10 dimensions).
# window: How many words to look at to the left and right (context size = 2).
# min_count: Ignore words that appear less than 1 time (keeps all words).
# sg=1: THIS IS THE CRUCIAL PART. sg=1 means "Use Skip-gram". 
#       (sg=0 would be CBOW).
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)

# 3. WHY DO WE USE IT? (Finding the Vector)
# Let's look at the math. Here is the actual vector (list of numbers) for "king".
# The model learned these numbers based on what words "king" appeared next to.
print("Vector for 'king':")
print(model.wv["king"])
print("-" * 50)

# 4. USING WORD2VEC TO FIND SIMILAR WORDS
# Because words that appear in the same context get similar numbers, 
# we can ask the model: "Which words are most similar to 'king'?"
# The model uses Cosine Similarity (from your Unit 2 syllabus) to find this.
print("Words most similar to 'king':")
similar_words = model.wv.most_similar("king", topn=3)
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
print("-" * 50)

# 5. THE MAGIC ANALOGY (King - Man + Woman = Queen)
# This is the most famous use of Word2Vec. It does actual vector math!
# It takes the vector for "king", subtracts "man", and adds "woman".
# Then it finds the closest word to that resulting math.
print("Analogy: King - Man + Woman = ?")
analogy = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
for word, score in analogy:
    print(f"Answer: {word} (Score: {score:.4f})")
print("-" * 50)

# 6. COMPARING ANIMALS
# It should also know that a dog and a puppy are related because they share contexts.
print("Words most similar to 'dog':")
similar_dog = model.wv.most_similar("dog", topn=3)
for word, score in similar_dog:
    print(f"{word}: {score:.4f}")