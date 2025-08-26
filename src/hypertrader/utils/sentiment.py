"""Sentiment and headline utilities with optional transformer support."""

from __future__ import annotations

from functools import lru_cache
import warnings
from typing import List

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news_headlines(api_key: str, query: str = "crypto", limit: int = 10) -> List[str]:
    """Fetch recent headlines from NewsAPI."""
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": limit,
        "sortBy": "publishedAt",
    }
    response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return [article["title"] for article in data.get("articles", [])]


@lru_cache()
def _hf_sentiment_pipeline():
    """Return a cached Hugging Face sentiment pipeline.

    Uses FinBERT by default but gracefully falls back if transformers is
    unavailable or the model cannot be loaded.
    """

    try:  # heavy imports only when requested
        from transformers import pipeline
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of the model checkpoint")
            warnings.filterwarnings("ignore", category=UserWarning, message="The following .* were not used when initializing")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
            return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:  # pragma: no cover - handled in tests via monkeypatch
        return None


def compute_sentiment_score(headlines: List[str], use_transformer: bool = False) -> float:
    """Return mean sentiment for a list of headlines.

    Parameters
    ----------
    headlines:
        Sequence of news headlines.
    use_transformer:
        If ``True`` attempt to score sentiment using a Hugging Face model.
        Falls back to VADER if the model is unavailable.
    """

    if not headlines:
        return 0.0

    if use_transformer:
        pipe = _hf_sentiment_pipeline()
        if pipe is not None:
            scores = []
            for h in headlines:
                try:
                    res = pipe(h)[0]
                    score = res["score"]
                    if res.get("label", "").upper().startswith("NEG"):
                        score = -score
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            return float(sum(scores) / len(scores))

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(sum(scores) / len(scores))


@lru_cache()
def _hf_summarizer():
    """Cached summarization pipeline using an open-source GPT-style model."""
    try:
        from transformers import pipeline
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of the model checkpoint")
            warnings.filterwarnings("ignore", category=UserWarning, message="The following .* were not used when initializing")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
            return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception:  # pragma: no cover - optional dependency
        return None


def summarize_headlines(headlines: List[str]) -> str:
    """Summarize a list of headlines using a transformer model.

    If the summarizer cannot be loaded the first three headlines are
    concatenated and returned.
    """

    if not headlines:
        return ""

    summarizer = _hf_summarizer()
    if summarizer is None:
        return " ".join(headlines[:3])

    text = " ".join(headlines)
    try:
        summary = summarizer(text, max_length=60, min_length=10, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return " ".join(headlines[:3])

