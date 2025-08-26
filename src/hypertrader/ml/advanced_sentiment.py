#!/usr/bin/env python3
"""Advanced ML Sentiment Analysis Strategies for Enhanced Trading Performance."""

import asyncio
import logging
from typing import Dict, List, Optional, Sequence, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class MultiModalSentimentAnalyzer:
    """Advanced sentiment analyzer combining multiple models and data sources."""
    
    def __init__(self):
        self.models = {}
        self.sentiment_cache = {}
        self.confidence_threshold = 0.7
        
        if HF_AVAILABLE:
            self._load_models()
    
    def _load_models(self):
        """Load multiple sentiment analysis models."""
        try:
            # Financial sentiment models
            self.models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models['financial_roberta'] = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Crypto-specific sentiment
            self.models['crypto_bert'] = pipeline(
                "sentiment-analysis",
                model="ElKulako/cryptobert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Market news sentiment
            self.models['news_bert'] = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            logging.warning(f"Failed to load some sentiment models: {e}")
    
    async def analyze_multi_source_sentiment(self, 
                                          financial_news: List[str],
                                          social_media: List[str], 
                                          market_data: Dict[str, float]) -> Dict[str, float]:
        """Analyze sentiment from multiple sources with advanced weighting."""
        
        results = {
            'composite_sentiment': 0.0,
            'confidence': 0.0,
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'market_fear_greed': 0.0,
            'volatility_sentiment': 0.0
        }
        
        if not HF_AVAILABLE or not self.models:
            return results
        
        try:
            # Analyze financial news
            news_sentiment = await self._analyze_news_sentiment(financial_news)
            results['news_sentiment'] = news_sentiment['score']
            
            # Analyze social media
            social_sentiment = await self._analyze_social_sentiment(social_media)
            results['social_sentiment'] = social_sentiment['score']
            
            # Market-based sentiment indicators
            market_sentiment = self._calculate_market_sentiment(market_data)
            results['market_fear_greed'] = market_sentiment['fear_greed']
            results['volatility_sentiment'] = market_sentiment['volatility_score']
            
            # Composite sentiment with dynamic weighting
            weights = self._calculate_dynamic_weights(news_sentiment['confidence'], 
                                                    social_sentiment['confidence'],
                                                    market_sentiment['reliability'])
            
            results['composite_sentiment'] = (
                news_sentiment['score'] * weights['news'] +
                social_sentiment['score'] * weights['social'] +
                market_sentiment['fear_greed'] * weights['market'] +
                market_sentiment['volatility_score'] * weights['volatility']
            )
            
            results['confidence'] = np.mean([
                news_sentiment['confidence'],
                social_sentiment['confidence'], 
                market_sentiment['reliability']
            ])
            
        except Exception as e:
            logging.error(f"Error in multi-source sentiment analysis: {e}")
        
        return results
    
    async def _analyze_news_sentiment(self, news_texts: List[str]) -> Dict[str, float]:
        """Analyze financial news sentiment with ensemble approach."""
        if not news_texts:
            return {'score': 0.0, 'confidence': 0.0}
        
        try:
            # Use multiple models for news analysis
            finbert_results = []
            crypto_results = []
            news_results = []
            
            for text in news_texts[:10]:  # Limit to avoid rate limits
                if len(text) > 512:
                    text = text[:512]
                
                # FinBERT analysis
                if 'finbert' in self.models:
                    fb_result = self.models['finbert'](text)[0]
                    score = self._convert_sentiment_score(fb_result)
                    finbert_results.append(score)
                
                # Crypto-specific analysis
                if 'crypto_bert' in self.models and any(crypto in text.lower() for crypto in ['btc', 'bitcoin', 'crypto', 'ethereum', 'eth']):
                    cb_result = self.models['crypto_bert'](text)[0]
                    score = self._convert_sentiment_score(cb_result)
                    crypto_results.append(score)
                
                # General news sentiment
                if 'news_bert' in self.models:
                    nb_result = self.models['news_bert'](text)[0]
                    score = self._convert_news_bert_score(nb_result)
                    news_results.append(score)
            
            # Ensemble scoring
            all_scores = []
            if finbert_results:
                all_scores.extend(finbert_results)
            if crypto_results:
                all_scores.extend([s * 1.2 for s in crypto_results])  # Weight crypto higher
            if news_results:
                all_scores.extend(news_results)
            
            if all_scores:
                final_score = np.mean(all_scores)
                confidence = 1.0 - np.std(all_scores) / (np.abs(np.mean(all_scores)) + 0.1)
                confidence = max(0.0, min(1.0, confidence))
            else:
                final_score = 0.0
                confidence = 0.0
            
            return {'score': final_score, 'confidence': confidence}
            
        except Exception as e:
            logging.error(f"Error analyzing news sentiment: {e}")
            return {'score': 0.0, 'confidence': 0.0}
    
    async def _analyze_social_sentiment(self, social_texts: List[str]) -> Dict[str, float]:
        """Analyze social media sentiment with spam filtering."""
        if not social_texts:
            return {'score': 0.0, 'confidence': 0.0}
        
        try:
            # Filter out likely spam/bot content
            filtered_texts = self._filter_social_content(social_texts)
            
            if not filtered_texts:
                return {'score': 0.0, 'confidence': 0.0}
            
            roberta_results = []
            
            for text in filtered_texts[:20]:  # Limit processing
                if len(text) > 280:  # Twitter length
                    text = text[:280]
                
                if 'financial_roberta' in self.models:
                    result = self.models['financial_roberta'](text)[0]
                    score = self._convert_sentiment_score(result)
                    roberta_results.append(score)
            
            if roberta_results:
                # Apply recency weighting (newer posts weighted higher)
                weights = np.exp(-np.arange(len(roberta_results)) * 0.1)
                weighted_score = np.average(roberta_results, weights=weights)
                
                confidence = 1.0 - np.std(roberta_results) / (np.abs(weighted_score) + 0.1)
                confidence = max(0.0, min(1.0, confidence))
            else:
                weighted_score = 0.0
                confidence = 0.0
            
            return {'score': weighted_score, 'confidence': confidence}
            
        except Exception as e:
            logging.error(f"Error analyzing social sentiment: {e}")
            return {'score': 0.0, 'confidence': 0.0}
    
    def _calculate_market_sentiment(self, market_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate market-based sentiment indicators."""
        try:
            # Fear & Greed calculation based on multiple factors
            fear_greed_components = []
            
            # Volatility component (higher vol = more fear)
            if 'volatility' in market_data:
                vol_normalized = min(1.0, market_data['volatility'] / 0.8)  # Normalize to 80% vol
                fear_greed_components.append(1.0 - vol_normalized)  # Invert (low vol = greed)
            
            # Price momentum component
            if 'price_change_24h' in market_data:
                momentum = market_data['price_change_24h'] / 100.0  # Convert percentage
                momentum_normalized = np.tanh(momentum * 2)  # Smooth between -1 and 1
                fear_greed_components.append(momentum_normalized)
            
            # Volume analysis
            if 'volume_ratio' in market_data:
                vol_ratio = market_data['volume_ratio']  # Current vol / avg vol
                if vol_ratio > 2.0:  # High volume
                    vol_sentiment = 0.3 if market_data.get('price_change_24h', 0) > 0 else -0.3
                else:
                    vol_sentiment = 0.0
                fear_greed_components.append(vol_sentiment)
            
            # RSI-based sentiment
            if 'rsi' in market_data:
                rsi = market_data['rsi']
                if rsi > 70:
                    rsi_sentiment = -0.5  # Overbought = bearish
                elif rsi < 30:
                    rsi_sentiment = 0.5   # Oversold = bullish
                else:
                    rsi_sentiment = (50 - rsi) / 100.0  # Normalized
                fear_greed_components.append(rsi_sentiment)
            
            fear_greed_score = np.mean(fear_greed_components) if fear_greed_components else 0.0
            
            # Volatility sentiment (separate indicator)
            volatility_score = 0.0
            if 'volatility' in market_data:
                vol = market_data['volatility']
                if vol > 0.6:  # High volatility
                    volatility_score = -0.3
                elif vol < 0.2:  # Low volatility
                    volatility_score = 0.2
                else:
                    volatility_score = (0.4 - vol) / 2.0  # Linear between extremes
            
            reliability = len(fear_greed_components) / 4.0  # How many components we have
            
            return {
                'fear_greed': fear_greed_score,
                'volatility_score': volatility_score,
                'reliability': reliability
            }
            
        except Exception as e:
            logging.error(f"Error calculating market sentiment: {e}")
            return {'fear_greed': 0.0, 'volatility_score': 0.0, 'reliability': 0.0}
    
    def _filter_social_content(self, texts: List[str]) -> List[str]:
        """Filter out spam and low-quality social media content."""
        filtered = []
        spam_indicators = ['🚀', '💎', '🌙', 'pump', 'moon', 'lambo', 'hodl']
        
        for text in texts:
            # Skip very short posts
            if len(text.split()) < 3:
                continue
                
            # Skip posts with too many spam indicators
            spam_count = sum(1 for indicator in spam_indicators if indicator.lower() in text.lower())
            if spam_count > 2:
                continue
                
            # Skip posts that are mostly caps or numbers
            if sum(1 for c in text if c.isupper()) / len(text) > 0.7:
                continue
                
            filtered.append(text)
        
        return filtered
    
    def _convert_sentiment_score(self, result: Dict) -> float:
        """Convert model output to normalized score (-1 to 1)."""
        label = result['label'].upper()
        score = result['score']
        
        if label in ['POSITIVE', 'POS']:
            return score
        elif label in ['NEGATIVE', 'NEG']:
            return -score
        else:  # NEUTRAL
            return 0.0
    
    def _convert_news_bert_score(self, result: Dict) -> float:
        """Convert news BERT 5-star rating to sentiment score."""
        try:
            stars = int(result['label'].split()[0])  # Extract star rating
            # Convert 1-5 stars to -1 to 1 scale
            return (stars - 3) / 2.0
        except:
            return 0.0
    
    def _calculate_dynamic_weights(self, news_conf: float, social_conf: float, market_rel: float) -> Dict[str, float]:
        """Calculate dynamic weights based on confidence levels."""
        total_conf = news_conf + social_conf + market_rel + 0.5  # Add base weight for volatility
        
        return {
            'news': news_conf / total_conf * 0.4,      # News gets up to 40%
            'social': social_conf / total_conf * 0.3,   # Social gets up to 30%
            'market': market_rel / total_conf * 0.2,    # Market gets up to 20%
            'volatility': 0.5 / total_conf * 0.1        # Volatility gets up to 10%
        }

class SentimentMomentumStrategy:
    """Strategy that combines sentiment analysis with momentum indicators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "sentiment_momentum"
        self.analyzer = MultiModalSentimentAnalyzer()
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.3)
        self.momentum_period = self.config.get('momentum_period', 14)
    
    async def generate_signals(self, market_data: pd.DataFrame, 
                             news_data: List[str], 
                             social_data: List[str]) -> Dict[str, float]:
        """Generate trading signals combining sentiment and momentum."""
        
        if len(market_data) < self.momentum_period:
            return {'signal': 0.0, 'confidence': 0.0}
        
        try:
            # Calculate price momentum
            returns = market_data['close'].pct_change(self.momentum_period)
            momentum_score = np.tanh(returns.iloc[-1] * 10)  # Normalize to -1,1
            
            # Prepare market sentiment data
            market_sentiment_data = {
                'volatility': market_data['close'].pct_change().std() * np.sqrt(252),
                'price_change_24h': (market_data['close'].iloc[-1] / market_data['close'].iloc[-24] - 1) * 100 if len(market_data) >= 24 else 0,
                'volume_ratio': market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1] if len(market_data) >= 20 else 1.0,
                'rsi': self._calculate_rsi(market_data['close'])
            }
            
            # Get advanced sentiment analysis
            sentiment_result = await self.analyzer.analyze_multi_source_sentiment(
                news_data, social_data, market_sentiment_data
            )
            
            composite_sentiment = sentiment_result['composite_sentiment']
            sentiment_confidence = sentiment_result['confidence']
            
            # Combine sentiment and momentum
            signal = 0.0
            confidence = 0.0
            
            # Strong bullish: positive sentiment + positive momentum
            if composite_sentiment > self.sentiment_threshold and momentum_score > 0.1:
                signal = min(1.0, (composite_sentiment + momentum_score) / 2)
                confidence = sentiment_confidence * 0.9
            
            # Strong bearish: negative sentiment + negative momentum
            elif composite_sentiment < -self.sentiment_threshold and momentum_score < -0.1:
                signal = max(-1.0, (composite_sentiment + momentum_score) / 2)
                confidence = sentiment_confidence * 0.9
            
            # Divergence signals (weaker but still valid)
            elif abs(composite_sentiment) > self.sentiment_threshold * 1.5:
                signal = composite_sentiment * 0.5  # Reduced strength for divergence
                confidence = sentiment_confidence * 0.6
            
            return {
                'signal': signal,
                'confidence': confidence,
                'sentiment_score': composite_sentiment,
                'momentum_score': momentum_score,
                'news_sentiment': sentiment_result['news_sentiment'],
                'social_sentiment': sentiment_result['social_sentiment'],
                'market_fear_greed': sentiment_result['market_fear_greed']
            }
            
        except Exception as e:
            logging.error(f"Error in sentiment momentum strategy: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for current price."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

# Factory function for easy integration
async def compute_advanced_sentiment_signals(market_data: pd.DataFrame,
                                           news_headlines: List[str],
                                           social_posts: List[str],
                                           config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Compute advanced sentiment-based trading signals."""
    
    strategy = SentimentMomentumStrategy(config)
    return await strategy.generate_signals(market_data, news_headlines, social_posts)
