"""
NLP Sentiment Microservice with FinBERT/Twitter-RoBERTa/BART-MNLI
Real sentiment signals for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)
import torch
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import re
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Structured sentiment output."""
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    source: str


class FinancialSentimentAnalyzer:
    """
    Multi-model sentiment analyzer for financial texts.
    Uses domain-specific models for accurate finance sentiment.
    """
    
    def __init__(
        self, 
        device: str = "auto",
        cache_dir: str = "models/sentiment_cache"
    ):
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        self.models = {}
        self._initialize_models()
        
    def _get_device(self, device: str) -> str:
        """Get optimal device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    def _initialize_models(self):
        """Initialize all sentiment models."""
        try:
            # FinBERT for financial sentiment
            logger.info("Loading FinBERT model...")
            self.models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if self.device == "cuda" else -1,
                cache_dir=self.cache_dir
            )
            
            # Twitter-RoBERTa for social sentiment
            logger.info("Loading Twitter-RoBERTa model...")
            self.models['twitter_roberta'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                cache_dir=self.cache_dir
            )
            
            # BART-MNLI for catalyst classification
            logger.info("Loading BART-MNLI for catalyst detection...")
            self.models['bart_mnli'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1,
                cache_dir=self.cache_dir
            )
            
            logger.info("All sentiment models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            raise
            
    def analyze_financial_text(self, text: str) -> SentimentScore:
        """Analyze text using FinBERT."""
        try:
            result = self.models['finbert'](text)[0]
            
            # Map FinBERT labels
            label_map = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
            label = label_map.get(result['label'].lower(), 'neutral')
            
            # Create structured output
            pos = result['score'] if label == 'positive' else 0.0
            neg = result['score'] if label == 'negative' else 0.0  
            neu = result['score'] if label == 'neutral' else 0.0
            
            # Normalize to sum to 1
            total = pos + neg + neu
            if total > 0:
                pos, neg, neu = pos/total, neg/total, neu/total
            else:
                pos, neg, neu = 0.33, 0.33, 0.34
                
            compound = pos - neg  # Range [-1, 1]
            
            return SentimentScore(
                positive=pos,
                negative=neg,
                neutral=neu,
                compound=compound,
                confidence=result['score'],
                source='finbert'
            )
            
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0, 'finbert')
            
    def analyze_social_text(self, text: str) -> SentimentScore:
        """Analyze social media text using Twitter-RoBERTa."""
        try:
            result = self.models['twitter_roberta'](text)[0]
            
            # Map Twitter-RoBERTa labels  
            label = result['label'].lower()
            score = result['score']
            
            if label == 'label_2':  # Positive
                pos, neg, neu = score, 0.0, 1.0 - score
            elif label == 'label_0':  # Negative
                pos, neg, neu = 0.0, score, 1.0 - score
            else:  # Neutral (label_1)
                pos, neg, neu = 0.0, 0.0, 1.0
                
            compound = pos - neg
            
            return SentimentScore(
                positive=pos,
                negative=neg,
                neutral=neu,
                compound=compound,
                confidence=score,
                source='twitter_roberta'
            )
            
        except Exception as e:
            logger.warning(f"Twitter-RoBERTa analysis failed: {e}")
            return SentimentScore(0.33, 0.33, 0.34, 0.0, 0.0, 'twitter_roberta')
            
    def detect_catalysts(self, text: str) -> Dict[str, float]:
        """Detect market catalysts using BART-MNLI zero-shot classification."""
        catalyst_labels = [
            "earnings announcement",
            "guidance update", 
            "mergers and acquisitions",
            "regulatory news",
            "federal reserve policy",
            "inflation data",
            "geopolitical events",
            "central bank decisions",
            "economic indicators",
            "corporate restructuring"
        ]
        
        try:
            result = self.models['bart_mnli'](text, catalyst_labels)
            
            # Create catalyst scores dict
            catalyst_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                catalyst_scores[label] = score
                
            return catalyst_scores
            
        except Exception as e:
            logger.warning(f"Catalyst detection failed: {e}")
            return {label: 0.0 for label in catalyst_labels}
            
    def batch_analyze(
        self, 
        texts: List[str], 
        analysis_type: str = 'financial'
    ) -> List[SentimentScore]:
        """Batch process multiple texts for efficiency."""
        if analysis_type == 'financial':
            analyzer = self.analyze_financial_text
        elif analysis_type == 'social':
            analyzer = self.analyze_social_text
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        results = []
        for text in texts:
            results.append(analyzer(text))
            
        return results


class NewsDataFetcher:
    """Fetch news data from multiple sources."""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.session = aiohttp.ClientSession()
        
    async def fetch_news(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Fetch recent news articles for symbol."""
        articles = []
        
        # Placeholder for news API integration
        # In production, integrate with:
        # - Alpha Vantage News API
        # - NewsAPI
        # - Financial Modeling Prep
        # - Custom RSS feeds
        
        logger.info(f"Fetching news for {symbol} (last {hours_back}h)")
        
        # Mock news data for demonstration
        mock_articles = [
            {
                "title": f"Bitcoin shows strong momentum amid institutional adoption",
                "summary": f"Major institutions continue to adopt {symbol} as digital gold narrative strengthens",
                "timestamp": datetime.now() - timedelta(hours=2),
                "source": "crypto_news",
                "url": "https://example.com/news1"
            },
            {
                "title": f"Regulatory clarity boosts {symbol} sentiment",
                "summary": f"Clear regulatory framework supports {symbol} growth prospects",
                "timestamp": datetime.now() - timedelta(hours=6),
                "source": "financial_news", 
                "url": "https://example.com/news2"
            }
        ]
        
        return mock_articles
        
    async def close(self):
        """Clean up session."""
        await self.session.close()


class SentimentAggregator:
    """Aggregate sentiment from multiple sources into trading signals."""
    
    def __init__(
        self,
        analyzer: FinancialSentimentAnalyzer,
        news_fetcher: NewsDataFetcher,
        ewma_alpha: float = 0.1
    ):
        self.analyzer = analyzer
        self.news_fetcher = news_fetcher
        self.ewma_alpha = ewma_alpha
        self.sentiment_history = defaultdict(list)
        
    async def get_sentiment_signals(
        self, 
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        """
        Generate aggregated sentiment signals for symbol.
        
        Returns:
            Dict with sentiment metrics and trading signals
        """
        # Fetch news
        articles = await self.news_fetcher.fetch_news(symbol, lookback_hours)
        
        if not articles:
            logger.warning(f"No news articles found for {symbol}")
            return self._default_sentiment()
            
        # Analyze sentiment
        news_texts = [f"{article['title']} {article['summary']}" for article in articles]
        sentiment_scores = self.analyzer.batch_analyze(news_texts, 'financial')
        
        # Detect catalysts
        catalyst_scores = []
        for text in news_texts:
            catalysts = self.analyzer.detect_catalysts(text)
            catalyst_scores.append(max(catalysts.values()) if catalysts else 0.0)
            
        # Aggregate sentiment
        avg_sentiment = np.mean([s.compound for s in sentiment_scores])
        sentiment_std = np.std([s.compound for s in sentiment_scores])
        max_catalyst = max(catalyst_scores) if catalyst_scores else 0.0
        
        # Time decay weights (more recent = higher weight)
        weights = self._calculate_time_weights(articles)
        weighted_sentiment = np.average([s.compound for s in sentiment_scores], weights=weights)
        
        # Update EWMA
        if symbol in self.sentiment_history:
            self.sentiment_history[symbol].append(weighted_sentiment)
            ewma_sentiment = self._calculate_ewma(self.sentiment_history[symbol])
        else:
            self.sentiment_history[symbol] = [weighted_sentiment]
            ewma_sentiment = weighted_sentiment
            
        # Generate signals
        signals = {
            'sentiment_raw': avg_sentiment,
            'sentiment_ewma': ewma_sentiment,
            'sentiment_volatility': sentiment_std,
            'catalyst_strength': max_catalyst,
            'news_volume': len(articles),
            'bullish_signal': self._generate_bullish_signal(ewma_sentiment, max_catalyst),
            'bearish_signal': self._generate_bearish_signal(ewma_sentiment, max_catalyst),
            'sentiment_regime': self._classify_sentiment_regime(ewma_sentiment, sentiment_std)
        }
        
        logger.info(f"Sentiment signals for {symbol}: {signals}")
        return signals
        
    def _calculate_time_weights(self, articles: List[Dict]) -> np.ndarray:
        """Calculate time-decay weights for articles."""
        now = datetime.now()
        hours_ago = [(now - article['timestamp']).total_seconds() / 3600 for article in articles]
        
        # Exponential decay: more recent = higher weight
        weights = np.exp(-0.1 * np.array(hours_ago))
        return weights / weights.sum()
        
    def _calculate_ewma(self, values: List[float]) -> float:
        """Calculate EWMA of sentiment values."""
        if len(values) == 1:
            return values[0]
            
        ewma = values[0]
        for value in values[1:]:
            ewma = self.ewma_alpha * value + (1 - self.ewma_alpha) * ewma
            
        return ewma
        
    def _generate_bullish_signal(self, sentiment: float, catalyst: float) -> float:
        """Generate bullish signal strength [0, 1]."""
        sentiment_boost = max(0, sentiment) * 0.7
        catalyst_boost = catalyst * 0.3
        return min(1.0, sentiment_boost + catalyst_boost)
        
    def _generate_bearish_signal(self, sentiment: float, catalyst: float) -> float:
        """Generate bearish signal strength [0, 1].""" 
        sentiment_drag = max(0, -sentiment) * 0.7
        catalyst_boost = catalyst * 0.3  # Catalysts can be bearish too
        return min(1.0, sentiment_drag + catalyst_boost)
        
    def _classify_sentiment_regime(self, sentiment: float, volatility: float) -> str:
        """Classify current sentiment regime."""
        if sentiment > 0.2 and volatility < 0.3:
            return "bullish_stable"
        elif sentiment > 0.1 and volatility > 0.3:
            return "bullish_volatile"
        elif sentiment < -0.2 and volatility < 0.3:
            return "bearish_stable"
        elif sentiment < -0.1 and volatility > 0.3:
            return "bearish_volatile"
        else:
            return "neutral"
            
    def _default_sentiment(self) -> Dict[str, float]:
        """Return neutral sentiment when no data available."""
        return {
            'sentiment_raw': 0.0,
            'sentiment_ewma': 0.0,
            'sentiment_volatility': 0.0,
            'catalyst_strength': 0.0,
            'news_volume': 0,
            'bullish_signal': 0.0,
            'bearish_signal': 0.0,
            'sentiment_regime': 'neutral'
        }


# ONNX Export functionality for production deployment
def export_sentiment_models_to_onnx(
    analyzer: FinancialSentimentAnalyzer,
    output_dir: str = "models/onnx"
):
    """Export sentiment models to ONNX for fast inference."""
    import torch.onnx
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Exporting sentiment models to ONNX format...")
    
    # Note: This is a template - actual ONNX export requires more complex setup
    # for transformer models. In production, use:
    # - Optimum library for easy ONNX conversion
    # - Dynamic quantization with ONNX Runtime
    # - OpenVINO for Intel CPU acceleration
    
    try:
        # Export FinBERT (example - requires actual model extraction)
        logger.info("FinBERT export to ONNX requires Optimum library")
        
        # Command to run separately:
        # optimum-cli export onnx --model ProsusAI/finbert models/onnx/finbert/
        
        logger.info(f"Run: optimum-cli export onnx --model ProsusAI/finbert {output_path}/finbert/")
        logger.info(f"Run: optimum-cli export onnx --model cardiffnlp/twitter-roberta-base-sentiment-latest {output_path}/twitter_roberta/")
        logger.info(f"Run: optimum-cli export onnx --model facebook/bart-large-mnli {output_path}/bart_mnli/")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        
        
async def demo_sentiment_analysis():
    """Demonstrate sentiment analysis pipeline."""
    # Initialize components
    analyzer = FinancialSentimentAnalyzer()
    news_fetcher = NewsDataFetcher(api_keys={})
    aggregator = SentimentAggregator(analyzer, news_fetcher)
    
    try:
        # Test sentiment analysis
        test_text = "Bitcoin shows strong institutional adoption with major corporations adding BTC to their balance sheets"
        
        financial_sentiment = analyzer.analyze_financial_text(test_text)
        print(f"Financial Sentiment: {financial_sentiment}")
        
        catalysts = analyzer.detect_catalysts(test_text)
        print(f"Detected Catalysts: {catalysts}")
        
        # Test aggregated signals
        signals = await aggregator.get_sentiment_signals("BTCUSD")
        print(f"Aggregated Signals: {signals}")
        
    finally:
        await news_fetcher.close()


if __name__ == "__main__":
    asyncio.run(demo_sentiment_analysis())
