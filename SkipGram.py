# skipgram_logic.py

def generate_skipgram_pairs(sentence, window_size):
    words = sentence.split()
    pairs = []
    
    # Loop through every word in the sentence (this is the Center Word)
    for i in range(len(words)):
        center_word = words[i]
        
        # Look to the left and right up to the window_size
        for j in range(1, window_size + 1):
            # Look left
            if i - j >= 0:
                context_word = words[i - j]
                pairs.append((center_word, context_word))
            # Look right
            if i + j < len(words):
                context_word = words[i + j]
                pairs.append((center_word, context_word))
                
    return pairs

# Let's test it with a sentence
sentence = "The cat chased the mouse"
window = 1

print(f"Sentence: '{sentence}'")
print(f"Window Size: {window}\n")

pairs = generate_skipgram_pairs(sentence, window)

print("--- Skip-gram Training Pairs (Center -> Context) ---")
for center, context in pairs:
    print(f"Input: '{center}' \t-> Predict Output: '{context}'")