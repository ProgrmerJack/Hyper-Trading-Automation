"""Asynchronous sentiment and catalyst extraction using HuggingFace models.

This module implements a microservice‑like set of functions that ingest
news headlines and social media posts, run multiple sentiment models
from HuggingFace and return structured factors.  It supports FinBERT
variants for financial sentiment, Twitter RoBERTa for social flow
analysis, and BART‑MNLI for zero‑shot classification of catalyst
themes (earnings, guidance, M&A, regulatory, macro and geopolitics).

The functions are designed to operate asynchronously to avoid
blocking the main trading loop.  Pipeline objects are lazily cached
and re‑used across calls, with graceful fallback to zero values if
dependencies are missing.  When integrated into the trading engine
these factors can be combined with other signals to improve edge
estimation.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Sequence

# Attempt to import transformers.  These imports are optional; if
# transformers is not available or model downloads fail the returned
# pipeline functions will be ``None`` and scores will default to zero.
try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore


@lru_cache()
def _get_finbert_pipeline() -> Any:
    """Return a sentiment analysis pipeline tuned for financial text.

    The function first attempts to load the ProsusAI/FinBERT model.  If
    that fails (e.g. due to missing weights or network issues) it
    attempts to load the yiyanghkust/finbert‑tone variant.  If both
    fail, ``None`` is returned and FinBERT scores will default to zero.

    The pipeline is cached after the first call to avoid repeated
    downloads and initialisation.
    """
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        try:
            return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
        except Exception:
            return None


@lru_cache()
def _get_twitter_pipeline() -> Any:
    """Return a sentiment analysis pipeline tuned for tweets.

    The primary model is cardiffnlp/twitter‑roberta‑base‑sentiment‑latest.
    If loading fails the function falls back to the older
    twitter‑roberta‑base‑sentiment checkpoint.  If neither is available
    the result is ``None`` and social scores default to zero.
    """
    if pipeline is None:
        return None
    try:
        return pipeline(
            "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    except Exception:
        try:
            return pipeline(
                "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
            )
        except Exception:
            return None


@lru_cache()
def _get_zero_shot_pipeline() -> Any:
    """Return a zero‑shot classifier for catalyst tagging.

    The BART‑MNLI model from Facebook provides robust natural language
    inference capabilities.  It supports multi‑label classification
    across user‑defined candidate labels.  If the pipeline cannot be
    loaded this function returns ``None`` and catalyst tags will be
    zeros.
    """
    if pipeline is None:
        return None
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception:
        return None


async def _run_pipeline(pipe: Any, text: str) -> Any:
    """Helper to run a pipeline on a single input asynchronously.

    HuggingFace pipelines are synchronous by default.  To avoid
    blocking the event loop we offload execution to the default
    executor via ``run_in_executor``.  If ``pipe`` is ``None`` the
    function returns ``None`` immediately.
    """
    if pipe is None:
        return None
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: pipe(text))


async def compute_finbert_scores(headlines: Sequence[str]) -> Dict[str, float]:
    """Compute average FinBERT sentiment scores for a list of headlines.

    Parameters
    ----------
    headlines:
        Sequence of headline strings.  Empty sequences return zeros.

    Returns
    -------
    dict[str, float]
        Dictionary containing positive probability, negative probability and
        a signed logit score.  Values are averaged across headlines.
    """
    pipe = _get_finbert_pipeline()
    if not pipe or not headlines:
        return {"fin_sent_prob_pos": 0.0, "fin_sent_prob_neg": 0.0, "fin_sent_logit": 0.0}
    # Execute the pipeline concurrently on all headlines
    results = await asyncio.gather(*[_run_pipeline(pipe, h) for h in headlines])
    prob_pos = 0.0
    prob_neg = 0.0
    logit = 0.0
    count = 0
    for res in results:
        if not res:
            continue
        try:
            first = res[0]
            score = float(first.get("score", 0.0))
            label = str(first.get("label", "")).lower()
            if "neg" in label:
                prob_neg += score
                logit -= score
            else:
                prob_pos += score
                logit += score
            count += 1
        except Exception:
            continue
    if count == 0:
        return {"fin_sent_prob_pos": 0.0, "fin_sent_prob_neg": 0.0, "fin_sent_logit": 0.0}
    return {
        "fin_sent_prob_pos": prob_pos / count,
        "fin_sent_prob_neg": prob_neg / count,
        "fin_sent_logit": logit / count,
    }


async def compute_twitter_scores(tweets: Sequence[str]) -> Dict[str, float]:
    """Compute average Twitter sentiment scores for a list of tweets.

    The returned dictionary contains keys ``tw_pos``, ``tw_neu`` and
    ``tw_neg`` representing the mean probabilities of positive,
    neutral and negative sentiment across the provided tweets.  If no
    tweets or no pipeline is available all values are zero.
    """
    pipe = _get_twitter_pipeline()
    if not pipe or not tweets:
        return {"tw_pos": 0.0, "tw_neu": 0.0, "tw_neg": 0.0}
    results = await asyncio.gather(*[_run_pipeline(pipe, t) for t in tweets])
    counts = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total = 0
    for res in results:
        if not res:
            continue
        try:
            first = res[0]
            label = str(first.get("label", "")).lower()
            score = float(first.get("score", 0.0))
            counts[label] = counts.get(label, 0.0) + score
            total += 1
        except Exception:
            continue
    if total == 0:
        return {"tw_pos": 0.0, "tw_neu": 0.0, "tw_neg": 0.0}
    return {
        "tw_pos": counts.get("positive", 0.0) / total,
        "tw_neu": counts.get("neutral", 0.0) / total,
        "tw_neg": counts.get("negative", 0.0) / total,
    }


async def compute_catalyst_tags(
    texts: Sequence[str], candidate_labels: Sequence[str] | None = None
) -> Dict[str, float]:
    """Assign catalyst tag probabilities using zero‑shot classification.

    Parameters
    ----------
    texts:
        Sequence of textual inputs (headlines and tweets).  They are
        concatenated into a single string for classification.
    candidate_labels:
        Optional iterable of labels defining the catalyst classes.  When
        unspecified a default set of financial catalysts is used.

    Returns
    -------
    dict[str, float]
        Mapping from label to probability assigned by the classifier.  If
        the model cannot be loaded all probabilities are zero.
    """
    if candidate_labels is None:
        candidate_labels = (
            "earnings",
            "guidance cut",
            "M&A",
            "regulatory",
            "macro-CPI",
            "geopolitics",
        )
    pipe = _get_zero_shot_pipeline()
    if not pipe or not texts:
        return {label: 0.0 for label in candidate_labels}
    combined = " ".join(texts)
    try:
        result = await _run_pipeline(pipe, combined)  # type: ignore[no-untyped-call]
        # The pipeline returns lists of labels and scores; map them accordingly
        scores: Dict[str, float] = {label: 0.0 for label in candidate_labels}
        for label, score in zip(result.get("labels", []), result.get("scores", [])):
            scores[str(label)] = float(score)
        # Ensure all labels are present in the returned dict
        for label in candidate_labels:
            scores.setdefault(str(label), 0.0)
        return scores
    except Exception:
        return {label: 0.0 for label in candidate_labels}


async def compute_sentiment_and_catalyst(
    headlines: Sequence[str], tweets: Sequence[str]
) -> Dict[str, float]:
    """Return combined sentiment and catalyst factors.

    This convenience wrapper concurrently computes FinBERT sentiment,
    Twitter sentiment and catalyst tag probabilities.  The result
    dictionary merges all keys from the individual calls.  Missing
    inputs default to empty sequences.
    """
    # Use create_task to launch coroutines concurrently
    fin_task = asyncio.create_task(compute_finbert_scores(headlines))
    tw_task = asyncio.create_task(compute_twitter_scores(tweets))
    cat_task = asyncio.create_task(compute_catalyst_tags(headlines + list(tweets)))
    fin_res, tw_res, cat_res = await asyncio.gather(fin_task, tw_task, cat_task)
    # Merge the three dictionaries into one
    merged: Dict[str, float] = {}
    merged.update(fin_res)
    merged.update(tw_res)
    merged.update(cat_res)
    return merged
