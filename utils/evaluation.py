from rouge_score import rouge_scorer  # ([huggingface.co](https://huggingface.co/vinai/phobert-base-v2?utm_source=chatgpt.com))
import matplotlib.pyplot as plt


def compute_rouge(refs, hyps):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    return scores


def plot_rouge(r1, r2, rL, labels, save_path="figs/rouge.png"):
    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.plot(x, r1, label="ROUGE‑1")
    plt.plot(x, r2, label="ROUGE‑2")
    plt.plot(x, rL, label="ROUGE‑L")
    plt.xticks(x, labels)
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)