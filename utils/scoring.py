import math
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer


class TfidfPositionScorer:
    def __init__(self, vectorizer: TfidfVectorizer, lambda_pos: float = 0.2):
        self.vect = vectorizer
        self.lambda_pos = lambda_pos

    def score(self, sentences):
        vecs = self.vect.transform(sentences)
        centroid = vecs.mean(axis=0)
        cos = cosine_similarity(vecs, centroid)
        pos = np.expand_dims(1 / (np.arange(len(sentences)) + 1), 1)
        return (cos + self.lambda_pos * pos).ravel()


class TextRankScorer:
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vect = vectorizer

    def score(self, sentences):
        vecs = self.vect.transform(sentences)
        sims = cosine_similarity(vecs)
        np.fill_diagonal(sims, 0)
        g = nx.from_numpy_array(sims)
        scores = nx.pagerank_numpy(g)
        return np.array(list(scores.values()))


class PhoBERTScorer:
    def __init__(self, model_name="vinai/phobert-base-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, sentences):
        emb = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        centroid = emb.mean(axis=0, keepdims=True)
        cos = cosine_similarity(emb, centroid).ravel()
        return cos

    def save(self, path):
        self.model.save(path)

# Wrapper
def load_scorer(name: str):
    if name == "tfidf":
        vect = joblib.load("results/tfidf_vect.pkl")
        return TfidfPositionScorer(vect)
    if name == "textrank":
        vect = joblib.load("results/tfidf_vect.pkl")
        return TextRankScorer(vect)
    # default phobert
    return PhoBERTScorer("results/phobert_scorer")


def eliminate_redundancy(ranked_idx, sentences, top_k=3, threshold=0.8):
    chosen = []
    emb_cache = None
    for idx in ranked_idx:
        if len(chosen) == 0:
            chosen.append(idx)
            if len(chosen) == top_k:
                break
            continue
        if emb_cache is None:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("vinai/phobert-base-v2")
            emb_cache = model.encode(sentences, convert_to_tensor=True)
        sims = cosine_similarity(
            [emb_cache[idx].cpu().numpy()], [emb_cache[j].cpu().numpy() for j in chosen]
        )[0]
        if np.max(sims) < threshold:
            chosen.append(idx)
        if len(chosen) == top_k:
            break
    return sorted(chosen)