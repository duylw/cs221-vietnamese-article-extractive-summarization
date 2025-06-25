import re
import argparse
import pandas as pd
from datasets import load_dataset
from utils.tokenization import sent_tokenize

def main(split: str = "train", out_path: str = "data/clean.tsv"):
    ds = load_dataset("nam194/vietnews", split=split)  # 144k articles
    rows = []
    for art in ds:
        art_id = art["guid"]
        print(f"Processing article {art_id}...")
        text = re.sub(r"\s+", " ", art["article"].strip())
        for idx, s in enumerate(sent_tokenize(text)):
            if len(s.split()) >= 5:  # bỏ qua những câu quá ngắn
                rows.append((art_id, idx, s))
    pd.DataFrame(rows, columns=["art_id", "sent_id", "sentence"]).to_csv(
        out_path, sep="\t", index=False
    )
    print(f"Saved {len(rows):,} sentences -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/clean.tsv")
    args = ap.parse_args()
    main(args.split, args.out)