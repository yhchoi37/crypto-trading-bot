__version__ = "2.0.0"
__author__ = "Younghwan Choi"

from .trading_system import MultiCoinTradingSystem
from .social_sentiment import TwitterSentimentCollector, RedditSentimentCollector
from .data_manager import MultiCoinDataManager
from .portfolio_manager import MultiCoinPortfolioManager
from .notifications import NotificationManager

__all__ = [
    'MultiCoinTradingSystem',
    'TwitterSentimentCollector', 
    'RedditSentimentCollector',
    'MultiCoinDataManager',
    'MultiCoinPortfolioManager',
    'NotificationManager',
]