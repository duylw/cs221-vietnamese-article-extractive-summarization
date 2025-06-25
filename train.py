import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.scoring import PhoBERTScorer

DATA_PATH = "data/clean.tsv"
MODELS_DIR = "results"


def train_tfidf(data_path=DATA_PATH):
    df = pd.read_csv(data_path, sep="\t")
    vect = TfidfVectorizer(min_df=3, max_df=0.9)
    vect.fit(df["sentence"].values)
    joblib.dump(vect, f"{MODELS_DIR}/tfidf_vect.pkl")
    print("TF‑IDF vectorizer saved.")


def train_phobert(data_path=DATA_PATH):
    scorer = PhoBERTScorer(data_path)
    scorer.save(f"{MODELS_DIR}/phobert_scorer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tfidf", "phobert"], required=True)
    ap.add_argument("--data", default=DATA_PATH, required=True)
    args = ap.parse_args()
    if args.method == "tfidf":
        train_tfidf(data_path=args.data)
    elif args.method == "phobert":
        train_phobert(data_path=args.data)


if __name__ == "__main__":
    main()