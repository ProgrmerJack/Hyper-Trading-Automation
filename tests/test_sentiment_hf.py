from hypertrader.utils import sentiment


def test_compute_sentiment_vader():
    score = sentiment.compute_sentiment_score(["good", "bad"])
    assert -1.0 <= score <= 1.0


def test_compute_sentiment_transformer(monkeypatch):
    # Replace the heavy transformer pipeline with a lightweight stub
    def fake_pipe(text):
        return [{"label": "positive", "score": 0.8}]

    monkeypatch.setattr(
        sentiment, "_hf_sentiment_pipeline", lambda: fake_pipe, raising=False
    )
    score = sentiment.compute_sentiment_score(["great market"], use_transformer=True)
    assert score > 0.0


def test_summarize_headlines(monkeypatch):
    def fake_summarizer(text, max_length, min_length, do_sample):  # noqa: D401
        return [{"summary_text": "summary"}]

    monkeypatch.setattr(
        sentiment, "_hf_summarizer", lambda: fake_summarizer, raising=False
    )
    assert sentiment.summarize_headlines(["headline one", "headline two"]) == "summary"

