import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news_headlines(api_key: str, query: str = 'crypto', limit: int = 10) -> list[str]:
    params = {
        'q': query,
        'apiKey': api_key,
        'pageSize': limit,
        'sortBy': 'publishedAt',
    }
    response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return [article['title'] for article in data.get('articles', [])]


def compute_sentiment_score(headlines: list[str]) -> float:
    analyzer = SentimentIntensityAnalyzer()
    if not headlines:
        return 0.0
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return sum(scores) / len(scores)
