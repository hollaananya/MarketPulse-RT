# utils_sentiment.py - Simple sentiment analysis for market headlines
import re
from typing import Dict, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Financial keywords for sentiment adjustment
POSITIVE_FINANCIAL = [
    'growth', 'profit', 'gain', 'surge', 'rally', 'boom', 'bullish', 'rise', 'up',
    'increase', 'strong', 'beat', 'exceed', 'outperform', 'revenue', 'earnings',
    'dividend', 'buyback', 'merger', 'acquisition', 'expansion', 'launch'
]

NEGATIVE_FINANCIAL = [
    'loss', 'decline', 'fall', 'drop', 'crash', 'bear', 'recession', 'crisis',
    'bankruptcy', 'layoff', 'cut', 'reduce', 'miss', 'disappoint', 'concern',
    'risk', 'volatility', 'uncertainty', 'investigation', 'lawsuit', 'scandal'
]

def simple_sentiment(text: str) -> float:
    """
    Simple rule-based sentiment for when VADER is not available
    Returns score between -1.0 (negative) and +1.0 (positive)
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Count positive and negative financial terms
    pos_count = sum(1 for word in POSITIVE_FINANCIAL if word in text_lower)
    neg_count = sum(1 for word in NEGATIVE_FINANCIAL if word in text_lower)
    
    # Simple scoring
    if pos_count == neg_count:
        return 0.0
    elif pos_count > neg_count:
        return min(0.8, (pos_count - neg_count) * 0.2)
    else:
        return max(-0.8, (pos_count - neg_count) * 0.2)

def headline_sentiment(text: str) -> float:
    """
    Analyze sentiment of financial headlines
    Returns compound score between -1.0 and +1.0
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    if not text.strip():
        return 0.0
    
    if VADER_AVAILABLE:
        try:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            
            # VADER returns compound score between -1 and 1
            base_score = scores['compound']
            
            # Adjust for financial context
            text_lower = text.lower()
            
            # Boost positive financial terms
            pos_boost = sum(0.1 for word in POSITIVE_FINANCIAL if word in text_lower)
            neg_boost = sum(0.1 for word in NEGATIVE_FINANCIAL if word in text_lower)
            
            adjusted_score = base_score + pos_boost - neg_boost
            
            # Keep within bounds
            return max(-1.0, min(1.0, adjusted_score))
            
        except Exception:
            # Fall back to simple method if VADER fails
            return simple_sentiment(text)
    else:
        return simple_sentiment(text)

def batch_sentiment(texts: list) -> list:
    """Process multiple texts at once"""
    return [headline_sentiment(text) for text in texts]

