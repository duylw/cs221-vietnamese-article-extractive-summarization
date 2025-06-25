import re
from functools import lru_cache

USE_VNCORE = False # Sử dụng VNCoreNLP nếu True, PyVi nếu False

if USE_VNCORE:
    from vncorenlp import VnCoreNLP
    annotator = VnCoreNLP("vncorenlp/VnCoreNLP‑1.1.1.jar", annotate_mode="wseg")
else:
    from pyvi import ViTokenizer

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def sent_tokenize(text: str):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


@lru_cache(maxsize=10_000)
def word_tokenize(sentence: str):
    if USE_VNCORE:
        return " ".join(annotator.tokenize(sentence)[0])
    return ViTokenizer.tokenize(sentence)