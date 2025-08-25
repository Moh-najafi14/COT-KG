# Noun-Level Semantic Similarity for Sentence Pair Evaluation Using Token Embeddings

This project evaluates the semantic similarity between sentence pairs by focusing on **nouns** (or noun phrases).  
It uses token embeddings to measure alignment between ground-truth and generated text, and visualizes results with **UMAP**.  

---

## What pairs are being compared?

For a sentence pair (**ground-truth vs. generated**):

1. **Extract nouns (or noun phrases)** from both sentences.  
2. **Embed each noun** using a SentenceTransformer (e.g., `all-MiniLM-L6-v2`).  
3. **Compute cosine similarity** between every noun in the ground-truth sentence and every noun in the generated sentence, forming a **similarity matrix**.  
4. For each ground-truth noun, take the **most similar noun** in the generated sentence (**max similarity per row**).  
5. **Average** those max similarities → yields a **single score per sentence pair**.  

---

## What is UMAP doing?

- **UMAP** (Uniform Manifold Approximation and Projection) is a **dimensionality reduction technique**.  
- It takes **high-dimensional vectors** (noun embeddings) and projects them to **2D** for visualization in a scatter plot.  

---

## What do the UMAP axes mean?

- The axes do **not** correspond to real-world values (like height or weight).  
- They are **abstract dimensions** chosen by UMAP to preserve structure in the data.  

**Interpretation:**  
- Points **close together** → embeddings are similar.  
- Points **far apart** → embeddings are dissimilar.  

---

## Insights from a UMAP plot of nouns

- **Clusters** → groups of nouns with similar meanings or contexts.  
- **Overlap** → if ground-truth and generated nouns cluster together, the generated text is semantically aligned with the ground truth.  
- **Separation** → distinct clusters suggest generated nouns differ semantically or may be inaccurate.  

---

## Example interpretations

- If the noun *"patient"* appears close to other medical terms in both ground-truth and generated sets → **good semantic alignment**.  
- If generated nouns like *"t"* or *"thing"* appear far away or isolated → likely **noise or irrelevant terms**.  
- If clusters reflect topics (e.g., *medical* vs. *travel nouns*) → indicates **semantic diversity** in the dataset.  

---

## Requirements

- Python 3.8+  
- [sentence-transformers](https://www.sbert.net/)  
- [umap-learn](https://umap-learn.readthedocs.io/)  
- matplotlib / seaborn for visualization  

---

## Example Usage

```python
from sentence_transformers import SentenceTransformer, util
import umap
import matplotlib.pyplot as plt

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences
ground_truth = "The patient was admitted to the hospital."
generated = "A sick person went to the clinic."

# (Steps: extract nouns → embed → compute similarities → visualize with UMAP)

## Acknowledgment
This code was developed by **Ivan Makohon**, Computer Science Department, Old Dominion University.

