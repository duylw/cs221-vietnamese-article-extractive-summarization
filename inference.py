import argparse
from utils.crawl import extract_text
from utils.tokenization import sent_tokenize
from utils.scoring import load_scorer, eliminate_redundancy


def summarize(src: str, k: int = 3, scorer_name: str = "phobert"):
    raw = extract_text(src)
    sents = sent_tokenize(raw)
    scorer = load_scorer(scorer_name)
    scores = scorer.score(sents)
    ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    chosen = eliminate_redundancy(ranked, sents, top_k=k, threshold=0.8)
    summary = [sents[i] for i in chosen]
    return summary, chosen, sents


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="URL or raw text")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--scorer", default="phobert")
    opts = ap.parse_args()
    summ, idx, _ = summarize(opts.src, opts.k, opts.scorer)
    print("\n".join(summ))
