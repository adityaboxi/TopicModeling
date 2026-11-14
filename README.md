# Topic Modeling and Word Embeddings: LSA vs LDA vs Word2Vec

This repository contains implementations of the core Natural Language Processing (NLP) algorithms: **LSA**, **LDA**, **Word2Vec**, and **Skip-gram**. 

Topic Modeling is a way for computers to read a bunch of documents and automatically group them into hidden "topics" (piles). Word Embeddings are a way for computers to map individual words into a mathematical space to understand their meaning. 

Here is a simple breakdown of how they work, their math, and how they are different.

---

## 1. LSA (Latent Semantic Analysis)

**Simple Explanation:** 
Imagine a giant Excel sheet where rows are documents and columns are words. LSA is basically a mathematical blender. It takes that giant sheet and shrinks it down into a smaller sheet. It doesn't know what the words mean; it just uses **Linear Algebra** to find which words always show up together.

**The Math (Singular Value Decomposition - SVD):**
The core formula for LSA is:
\[
X = U \Sigma V^T
\]
*   **\(X\)**: The giant matrix of words vs. documents (usually using TF-IDF).
*   **\(U\)**: The Document-Topic matrix (how much a document belongs to a topic).
*   **\(\Sigma\)**: A diagonal matrix of "singular values" (how important each topic is).
*   **\(V^T\)**: The Topic-Word matrix (which words make up each topic).

*In plain English:* We take the big matrix \(X\) and break it into three smaller matrices. By keeping only the top few numbers in \(\Sigma\) (e.g., the top 3), we force the computer to find the top 3 hidden topics.

**The Catch:** Because LSA uses linear algebra, its outputs can be negative numbers (e.g., Document 1 is `-0.4` related to Topic A). This makes it hard to interpret (you can't be "-40% interested" in sports).

---

## 2. LDA (Latent Dirichlet Allocation)

**Simple Explanation:** 
LDA treats text like a **probability game**. It assumes that every document is a smoothie, and the topics are fruits. A document is made of a mix of topics (e.g., 70% Apple, 30% Banana). It also assumes every topic is a mix of words. LDA uses statistics to reverse-engineer the recipe of the smoothie.

**The Math (Dirichlet Distribution & Probability):**
LDA is based on the Dirichlet distribution. It assumes two main probabilities:
1.  **Document-Topic Probability (\(\theta\))**: What is the probability this document is about Topic A, B, or C?
    \[
\theta \sim \text{Dirichlet}(\alpha)
\]
2.  **Topic-Word Probability (\(\phi\))**: What is the probability this word belongs to Topic A, B, or C?
    \[
\phi \sim \text{Dirichlet}(\beta)
\]

*In plain English:* LDA goes through every single word in a document and asks: *"Based on the other words here, what are the odds this word belongs to Topic 1, and what are the odds this document is about Topic 1?"* It repeats this math millions of times until the probabilities stabilize.

**The Benefit:** LDA only gives positive probabilities (e.g., Document 1 is `90%` Topic A, `10%` Topic B). This is much easier for humans to read and understand.

---

## 3. Word2Vec

**Simple Explanation:** 
While LSA and LDA group *documents*, Word2Vec looks only at *individual words*. It turns a word into a dense list of numbers (a vector). It's like a language map. Words with similar meanings (like "cat" and "dog") end up placed very close to each other on this map.

**The Math (Vector Arithmetic & Cosine Similarity):**
Word2Vec maps words to a vector space, allowing us to do math with meaning:
\[
Vector("King") - Vector("Man") + Vector("Woman") \approx Vector("Queen")
\]
*   **\(Vector\)**: A dense list of numbers (e.g., 100 dimensions) learned by a shallow neural network.
*   **Cosine Similarity**: The math used to measure the angle between two word vectors to see how similar they are.

*In plain English:* Instead of a massive spreadsheet of 50,000 words with mostly zeros, Word2Vec squishes the meaning of a word into a tiny list of numbers. Because of how the numbers are learned, words that mean the same thing have almost identical numbers.

**The Benefit:** It completely solves the "meaning" problem. To TF-IDF, "car" and "automobile" are completely unrelated. To Word2Vec, their vectors are practically identical.

---

## 4. Skip-gram (Word2Vec Architecture)

**Simple Explanation:** 
Skip-gram is the specific engine under the hood of Word2Vec. It plays a guessing game. It takes one word (the center word), and tries to guess the words that are surrounding it in a sentence. 

**The Math (Softmax & Context Prediction):**
The goal of Skip-gram is to maximize the probability of the context words, given the center word:
\[
 \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} | w_t) 
\]
*   **\(w_t\)**: The center word (e.g., "cat").
*   **\(w_{t+j}\)**: The context words (e.g., "the", "chased").
*   **\(c\)**: The window size (how far left and right to look).

*In plain English:* The network looks at the word "cat". It outputs a probability for every other word in the dictionary. It asks, *"What are the odds that 'chased' is nearby? What are the odds that 'the' is nearby?"* It then adjusts its numbers to make those odds higher.

**The Benefit:** Because it focuses on one word predicting its neighbors, it is incredibly good at learning the meaning of **rare words** and works well even if you have a small dataset.

---

## 5. The Big Differences (Summary)

| Feature | LSA | LDA | Word2Vec (Skip-gram) |
| :--- | :--- | :--- | :--- |
| **Core Math** | **Linear Algebra** (SVD) | **Probability** (Bayesian) | **Neural Network** (Shallow ANN) |
| **What it groups** | **Documents** (Topics) | **Documents** (Topics) | **Words** (Meaning/Embeddings) |
| **Best Input** | **TF-IDF** | **Raw Word Counts** | Raw Text (Tokenized) |
| **Output Numbers** | Can be **negative** | Always **probabilities** | Dense Vectors (e.g., 100 floats) |
| **Interpretation** | Hard to explain negative scores. | Easy to explain (e.g., 80% Sports). | Easy (e.g., Cat is close to Dog). |
| **Unit in Syllabus** | Unit 4 | Unit 4 | Unit 2 |