"""Hugging Face sentiment & catalyst microservice (FinBERT, Twitter-RoBERTa, BART-MNLI).
Returns dict features ready for your meta-score.
"""
from typing import List, Dict, Any
from functools import lru_cache
from transformers import pipeline

@lru_cache(maxsize=1)
def _finbert():
    return pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True, truncation=True)

@lru_cache(maxsize=1)
def _twitter():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True)

@lru_cache(maxsize=1)
def _bart():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def finbert_features(texts: List[str]) -> List[Dict[str, float]]:
    if not texts: return []
    clf = _finbert()
    outs = []
    for res in clf(texts, batch_size=16):
        sc = {x["label"].lower(): float(x["score"]) for x in res}
        outs.append({
            "fin_sent_prob_pos": sc.get("positive", 0.0),
            "fin_sent_prob_neg": sc.get("negative", 0.0),
            "fin_sent_logit": sc.get("positive",0.0) - sc.get("negative",0.0)
        })
    return outs

def twitter_features(texts: List[str]) -> List[Dict[str, float]]:
    if not texts: return []
    clf = _twitter()
    outs = []
    for r in clf(texts, batch_size=32):
        # pipeline returns a dict with label/score for each item
        d = {"tw_pos":0.0, "tw_neu":0.0, "tw_neg":0.0}
        label = r.get("label","NEU").lower()
        score = float(r.get("score",0.0))
        if "pos" in label: d["tw_pos"] = score
        elif "neg" in label: d["tw_neg"] = score
        else: d["tw_neu"] = score
        outs.append(d)
    return outs

def zero_shot_labels(texts: List[str], labels=None) -> List[Dict[str, Any]]:
    if labels is None:
        labels = ["earnings","guidance cut","m&a","regulatory","macro-cpi","geopolitics"]
    z = _bart(); outs=[]
    for t in texts:
        r = z(t, candidate_labels=labels, multi_label=True)
        outs.append({f"zs_{lbl.replace(' ','_')}": float(sc) for lbl, sc in zip(r["labels"], r["scores"])})
    return outs
