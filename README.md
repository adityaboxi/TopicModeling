# Topic Modeling: LSA vs LDA

This repository contains implementations of the two most popular Topic Modeling algorithms: **LSA** and **LDA**. 

Topic Modeling is a way for computers to read a bunch of documents and automatically group them into hidden "topics" (piles) without a human telling it what the topics are.

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

## 3. The Big Differences (Summary)

| Feature | LSA (Latent Semantic Analysis) | LDA (Latent Dirichlet Allocation) |
| :--- | :--- | :--- |
| **Core Math** | **Linear Algebra** (Matrix Factorization) | **Probability** (Bayesian Statistics) |
| **How it thinks** | Shrinks a big matrix into smaller ones. | Plays a probability game with words. |
| **Output Numbers** | Can be **negative** (e.g., `-0.5`, `0.8`) | Always **probabilities** (e.g., `0.1`, `0.9`) |
| **Best Input** | Works best with **TF-IDF** | Works best with **Raw Word Counts** |
| **Interpretation** | Hard to explain what a negative number means. | Easy to explain (e.g., "This doc is 80% Sports"). |
| **Speed** | Extremely fast. | Slower (needs many math iterations). |