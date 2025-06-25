import logging
from newspaper import Article  # ([newspaper.readthedocs.io](https://newspaper.readthedocs.io/?utm_source=chatgpt.com))
from readability import Document  # ([pypi.org](https://pypi.org/project/readability-lxml/?utm_source=chatgpt.com))
import requests
from bs4 import BeautifulSoup


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SummarizerBot/1.0)"}


def extract_text(src: str) -> str:
    if src.startswith("http"):
        try:
            art = Article(src, language="vi")
            art.download()
            art.parse()
            if len(art.text.split()) > 100:
                return art.text
        except Exception as exc:
            logging.warning("newspaper3k failed: %s", exc)
        # Fallback
        html = requests.get(src, headers=HEADERS, timeout=10).text
        doc = Document(html)
        cleaned = BeautifulSoup(doc.summary(), "lxml").get_text(" ", strip=True)
        return cleaned
    # Assume raw text
    return src
