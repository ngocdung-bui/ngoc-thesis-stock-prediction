# Hybrid Stock Market Prediction Model - Complete Implementation
# Chapter 4: Implementation and Experiments

import os
import sys
import logging
import warnings
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

# Transformers and NLP
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import nltk
import spacy
from textblob import TextBlob

# Financial Data and Technical Analysis
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)

# Visualization and Monitoring
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
from tqdm import tqdm

# Optimization and Analysis
import optuna
from optuna.trial import TrialState
import shap

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ================================
# CONFIGURATION AND SETUP
# ================================

class Config:
    """Configuration class containing all hyperparameters and settings"""
    
    def __init__(self):
        # Data Configuration
        self.DATA_START_DATE = '2010-01-01'
        self.DATA_END_DATE = '2023-12-31'
        self.TRAIN_RATIO = 0.70
        self.VAL_RATIO = 0.15
        self.TEST_RATIO = 0.15
        
        # Model Architecture
        self.LSTM_CONFIG = {
            'input_dim': 50,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'bidirectional': True,
            'use_attention': True
        }
        
        self.FINBERT_CONFIG = {
            'model_name': 'ProsusAI/finbert',
            'max_length': 512,
            'num_labels': 3,
            'dropout': 0.1,
            'freeze_layers': 6
        }
        
        self.INTEGRATION_CONFIG = {
            'lstm_dim': 64,
            'finbert_dim': 64,
            'hidden_dim': 128,
            'num_heads': 8,
            'dropout': 0.1
        }
        
        # Training Configuration
        self.TRAINING_CONFIG = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'lstm_lr': 1e-4,
            'finbert_lr': 2e-5,
            'integration_lr': 1e-4,
            'prediction_lr': 1e-3,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'price_loss_weight': 0.7,
            'direction_loss_weight': 0.3
        }
        
        # Scheduler Configuration
        self.SCHEDULER_CONFIG = {
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-6
        }
        
        # Data Processing
        self.SEQUENCE_LENGTH = 60
        self.TECHNICAL_FEATURES = [
            'returns', 'log_returns', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'BB_width', 'ATR', 'OBV',
            'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26', 'Volume_ratio'
        ]
        
        # S&P 500 symbols (subset for demonstration)
        self.SP500_SYMBOLS = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS',
            'BAC', 'ADBE', 'CRM', 'NFLX', 'KO', 'PEP', 'TMO',
            'ABT', 'COST', 'ACN', 'MRK', 'LLY', 'AVGO', 'TXN'
        ]
        
        # File paths
        self.DATA_DIR = 'data'
        self.MODEL_DIR = 'models'
        self.RESULTS_DIR = 'results'
        self.LOGS_DIR = 'logs'
        
        # Create directories
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)

# ================================
# LOGGING SETUP
# ================================

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ================================
# DATA COLLECTION MODULE
# ================================

class StockDataCollector:
    """Collect stock market data from various sources"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
    def collect_price_data(self) -> Tuple[Dict, List]:
        """Collect historical price data using yfinance"""
        price_data = {}
        failed_symbols = []
        
        def fetch_symbol_data(symbol):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date, end=self.end_date)
                if len(hist) > 252:  # At least one year of data
                    return symbol, hist
                else:
                    return symbol, None
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {str(e)}")
                return symbol, None
        
        # Parallel data collection
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_symbol_data, symbol): symbol 
                      for symbol in self.symbols}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Collecting price data"):
                symbol, data = future.result()
                if data is not None:
                    price_data[symbol] = data
                    self.logger.info(f"Successfully collected data for {symbol}")
                else:
                    failed_symbols.append(symbol)
        
        self.logger.info(f"Collected data for {len(price_data)} symbols")
        if failed_symbols:
            self.logger.warning(f"Failed to collect data for: {failed_symbols}")
        
        return price_data, failed_symbols
    
    def collect_fundamental_data(self) -> Dict:
        """Collect fundamental data from yfinance"""
        fundamental_data = {}
        
        for symbol in tqdm(self.symbols, desc="Collecting fundamental data"):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                fundamentals = {
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'dividend_yield': info.get('dividendYield', None),
                    'beta': info.get('beta', None),
                    'profit_margin': info.get('profitMargins', None),
                    'revenue_growth': info.get('revenueGrowth', None),
                    'debt_to_equity': info.get('debtToEquity', None),
                    'current_ratio': info.get('currentRatio', None),
                    'sector': info.get('sector', None),
                    'industry': info.get('industry', None)
                }
                
                fundamental_data[symbol] = fundamentals
                self.logger.info(f"Collected fundamental data for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error collecting fundamentals for {symbol}: {str(e)}")
        
        return fundamental_data
    
    def save_data(self, price_data: Dict, fundamental_data: Dict):
        """Save collected data to files"""
        # Save price data
        for symbol, data in price_data.items():
            data.to_csv(f'data/price/{symbol}_price.csv')
        
        # Save fundamental data
        with open('data/fundamental/fundamental_data.json', 'w') as f:
            json.dump(fundamental_data, f, default=str)
        
        self.logger.info("Data saved successfully")

class NewsDataCollector:
    """Collect and process financial news data"""
    
    def __init__(self, news_dataset_path: str, tweet_dataset_path: str):
        self.news_path = news_dataset_path
        self.tweets_path = tweet_dataset_path
        self.logger = logging.getLogger(__name__)
    
    def load_news_data(self) -> pd.DataFrame:
        """Load and preprocess news data"""
        try:
            # Load the dataset (assuming it's available)
            news_df = pd.read_csv(self.news_path)
            
            # Convert date column to datetime
            if 'date' in news_df.columns:
                news_df['date'] = pd.to_datetime(news_df['date'])
            elif 'timestamp' in news_df.columns:
                news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
                news_df['date'] = news_df['timestamp'].dt.date
            
            # Filter for quality
            news_df = news_df[news_df['text'].str.len() > 100]
            news_df = news_df.drop_duplicates(subset=['text'])
            
            self.logger.info(f"Loaded {len(news_df)} news articles")
            return news_df
            
        except FileNotFoundError:
            self.logger.warning("News dataset not found, creating synthetic data")
            return self._create_synthetic_news_data()
    
    def load_tweet_data(self) -> pd.DataFrame:
        """Load and preprocess tweet data"""
        try:
            tweets_df = pd.read_csv(self.tweets_path)
            
            if 'timestamp' in tweets_df.columns:
                tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
            
            # Filter for quality
            if 'verified' in tweets_df.columns:
                tweets_df = tweets_df[tweets_df['verified'] == True]
            
            # Clean tweet text
            tweets_df['cleaned_text'] = tweets_df['text'].apply(self._clean_tweet)
            
            self.logger.info(f"Loaded {len(tweets_df)} tweets")
            return tweets_df
            
        except FileNotFoundError:
            self.logger.warning("Tweet dataset not found, creating synthetic data")
            return self._create_synthetic_tweet_data()
    
    def _clean_tweet(self, text: str) -> str:
        """Clean tweet text"""
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Keep hashtags but remove #
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _create_synthetic_news_data(self) -> pd.DataFrame:
        """Create synthetic news data for demonstration"""
        np.random.seed(42)
        
        # Sample financial news templates
        templates = [
            "{company} reports strong quarterly earnings, beating analyst expectations",
            "{company} announces new product launch, shares rally",
            "{company} faces regulatory challenges, stock under pressure",
            "{company} CEO optimistic about future growth prospects",
            "{company} downgrades by analysts due to market concerns"
        ]
        
        companies = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla']
        
        synthetic_data = []
        date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        for date in date_range[:1000]:  # Limit for demonstration
            for _ in range(np.random.randint(1, 4)):  # 1-3 articles per day
                company = np.random.choice(companies)
                template = np.random.choice(templates)
                text = template.format(company=company)
                
                synthetic_data.append({
                    'date': date,
                    'text': text,
                    'symbol': 'AAPL',  # Simplified
                    'sentiment': np.random.choice(['positive', 'negative', 'neutral'])
                })
        
        return pd.DataFrame(synthetic_data)
    
    def _create_synthetic_tweet_data(self) -> pd.DataFrame:
        """Create synthetic tweet data for demonstration"""
        np.random.seed(42)
        
        tweet_templates = [
            "$AAPL looking strong today! #bullish",
            "$TSLA volatility is crazy #trading",
            "Market uncertainty ahead #bearish",
            "Great earnings from tech stocks #investing"
        ]
        
        synthetic_tweets = []
        date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')
        
        for timestamp in date_range[:5000]:  # Limit for demonstration
            tweet = np.random.choice(tweet_templates)
            synthetic_tweets.append({
                'timestamp': timestamp,
                'text': tweet,
                'cleaned_text': self._clean_tweet(tweet),
                'verified': True,
                'followers_count': np.random.randint(10000, 100000)
            })
        
        return pd.DataFrame(synthetic_tweets)

# ================================
# DATA PREPROCESSING MODULE
# ================================

class StructuredDataPreprocessor:
    """Preprocess structured financial data"""
    
    def __init__(self, scaling_method: str = 'robust'):
        self.scaling_method = scaling_method
        self.scaler = RobustScaler() if scaling_method == 'robust' else StandardScaler()
        self.feature_names = []
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Generate technical indicators
        df = self._generate_technical_indicators(df)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Drop NaN rows
        df = df.dropna()
        
        # Scale features
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        if len(df) > 0:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            self.feature_names = feature_cols
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Forward fill for small gaps
        df = df.fillna(method='ffill', limit=2)
        # Linear interpolation for larger gaps
        df = df.interpolate(method='linear', limit_direction='forward')
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in returns"""
        for col in ['returns', 'log_returns']:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q3)
        return df
    
    def _generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators"""
        
        # Moving averages
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_30'] = ta.trend.sma_indicator(df['Close'], window=30)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        
        # ATR
        df['ATR'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )
        
        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lagged features"""
        
        for lag in lags:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_roll_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_roll_std_{window}'] = df['returns'].rolling(window).std()
        
        return df

class TextDataPreprocessor:
    """Preprocess textual data for sentiment analysis"""
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # Fallback to BERT if FinBERT not available
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("spaCy model not found, using basic preprocessing")
            self.nlp = None
            
        self.financial_terms = self._load_financial_dictionary()
    
    def _load_financial_dictionary(self) -> set:
        """Load financial-specific terms"""
        return {
            'bullish', 'bearish', 'long', 'short', 'buy', 'sell', 'hold',
            'upgrade', 'downgrade', 'outperform', 'underperform', 'neutral',
            'earnings', 'revenue', 'profit', 'loss', 'margin', 'growth',
            'volatility', 'risk', 'return', 'yield', 'dividend', 'split'
        }
    
    def preprocess_text(self, text: str) -> Dict:
        """Comprehensive text preprocessing"""
        
        # Basic cleaning
        cleaned_text = self._clean_text(text)
        
        # Entity recognition
        entities = self._extract_entities(cleaned_text) if self.nlp else []
        
        # Financial term extraction
        financial_terms = self._extract_financial_terms(cleaned_text)
        
        # Tokenization for BERT
        tokens = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'cleaned_text': cleaned_text,
            'entities': entities,
            'financial_terms': financial_terms,
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving financial information"""
        import re
        
        # Convert to lowercase but preserve ticker symbols
        ticker_pattern = r'\$[A-Z]{1,5}'
        tickers = re.findall(ticker_pattern, text)
        
        text = text.lower()
        
        # Restore ticker symbols
        for ticker in tickers:
            text = text.replace(ticker.lower(), ticker)
        
        # Remove URLs and HTML tags
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        # Keep important characters
        text = re.sub(r'[^a-zA-Z0-9\s\$\%\.\,\-\+]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract financial entities"""
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'PERCENT', 'CARDINAL', 'ORG']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial-specific terms"""
        if self.nlp:
            doc = self.nlp(text.lower())
            terms = [token.text for token in doc if token.text in self.financial_terms]
        else:
            words = text.lower().split()
            terms = [word for word in words if word in self.financial_terms]
            
        return list(set(terms))

# ================================
# DATASET CLASSES
# ================================

class StockDataset(Dataset):
    """PyTorch Dataset for stock market data"""
    
    def __init__(self, structured_data: pd.DataFrame, text_data: pd.DataFrame, 
                 seq_length: int = 60, prediction_horizon: int = 1):
        self.structured_data = structured_data
        self.text_data = text_data
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        
        # Align data by date
        self.aligned_data = self._align_data()
        
    def _align_data(self) -> List[Dict]:
        """Align structured and text data by date"""
        aligned = []
        
        # Get date ranges
        struct_dates = self.structured_data.index
        text_dates = self.text_data['date'] if 'date' in self.text_data.columns else []
        
        for i in range(self.seq_length, len(struct_dates) - self.prediction_horizon):
            current_date = struct_dates[i]
            
            # Get structured sequence
            struct_seq = self.structured_data.iloc[i-self.seq_length:i]
            
            # Get text data for the same period
            if len(text_dates) > 0:
                mask = (text_dates >= struct_dates[i-1]) & (text_dates <= current_date)
                text_seq = self.text_data[mask]
            else:
                text_seq = pd.DataFrame()
            
            # Get target (future return)
            target_return = self.structured_data['returns'].iloc[i + self.prediction_horizon]
            
            aligned.append({
                'date': current_date,
                'structured_seq': struct_seq,
                'text_seq': text_seq,
                'target': target_return
            })
        
        return aligned
    
    def __len__(self) -> int:
        return len(self.aligned_data)
    
    def __getitem__(self, idx: int) -> Dict:
        data_point = self.aligned_data[idx]
        
        # Prepare structured data
        struct_features = data_point['structured_seq'].select_dtypes(include=[np.number]).values
        if len(struct_features) < self.seq_length:
            # Pad if necessary
            padding = np.zeros((self.seq_length - len(struct_features), struct_features.shape[1]))
            struct_features = np.vstack([padding, struct_features])
        
        # Prepare text data (use first available text or empty)
        if len(data_point['text_seq']) > 0:
            text_sample = data_point['text_seq'].iloc[0]['text'] if 'text' in data_point['text_seq'].columns else ""
        else:
            text_sample = ""
        
        # Simple sentiment score (replace with actual FinBERT processing in real implementation)
        sentiment_score = self._get_basic_sentiment(text_sample)
        
        return {
            'structured': torch.FloatTensor(struct_features),
            'text_features': torch.FloatTensor([sentiment_score]),
            'target': torch.FloatTensor([data_point['target']]),
            'date': data_point['date']
        }
    
    def _get_basic_sentiment(self, text: str) -> float:
        """Basic sentiment analysis using TextBlob (placeholder)"""
        if not text:
            return 0.0
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

# ================================
# MODEL COMPONENTS
# ================================

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM with attention mechanism"""
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2, 
                 use_attention: bool = True):
        super(EnhancedLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Self-attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=dropout
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_flat = x.reshape(-1, self.input_dim)
        x_proj = self.input_projection(x_flat)
        x_proj = x_proj.reshape(batch_size, seq_len, self.hidden_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        
        # Attention
        attention_weights = None
        if self.use_attention:
            lstm_out_t = lstm_out.transpose(0, 1)
            attended_out, attention_weights = self.attention(
                lstm_out_t, lstm_out_t, lstm_out_t
            )
            lstm_out = attended_out.transpose(0, 1) + lstm_out
        
        # Use last output
        final_hidden = lstm_out[:, -1, :]
        output = self.output_projection(final_hidden)
        
        return output, attention_weights

class FinBERTSentimentModel(nn.Module):
    """FinBERT-based sentiment analysis model"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 num_labels: int = 3, dropout: float = 0.1):
        super(FinBERTSentimentModel, self).__init__()
        
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except:
            # Fallback to BERT base
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        # Custom heads
        self.dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Linear(768, num_labels)
        self.confidence_predictor = nn.Linear(768, 1)
        
        # Output projection for integration
        self.output_projection = nn.Sequential(
            nn.Linear(768 + num_labels + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Predictions
        sentiment_logits = self.sentiment_classifier(pooled_output)
        confidence_score = torch.sigmoid(self.confidence_predictor(pooled_output))
        
        # Combine features
        combined_features = torch.cat([
            pooled_output, sentiment_logits, confidence_score
        ], dim=-1)
        
        integrated_features = self.output_projection(combined_features)
        
        return {
            'integrated_features': integrated_features,
            'sentiment_logits': sentiment_logits,
            'confidence_score': confidence_score,
            'attention_weights': outputs.attentions[-1] if hasattr(outputs, 'attentions') else None
        }

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, lstm_dim: int = 64, finbert_dim: int = 64, 
                 hidden_dim: int = 128, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        
        # Projections
        self.lstm_projection = nn.Linear(lstm_dim, hidden_dim)
        self.finbert_projection = nn.Linear(finbert_dim, hidden_dim)
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Gating mechanisms
        self.gate_lstm = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.gate_finbert = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, lstm_features: torch.Tensor, 
                finbert_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Project features
        lstm_proj = self.lstm_projection(lstm_features)
        finbert_proj = self.finbert_projection(finbert_features)
        
        # Add sequence dimension for attention
        lstm_proj = lstm_proj.unsqueeze(1)
        finbert_proj = finbert_proj.unsqueeze(1)
        
        # Cross-attention
        lstm_attended, attn_l2f = self.cross_attention(
            lstm_proj, finbert_proj, finbert_proj
        )
        
        finbert_attended, attn_f2l = self.cross_attention(
            finbert_proj, lstm_proj, lstm_proj
        )
        
        # Remove sequence dimension
        lstm_attended = lstm_attended.squeeze(1)
        finbert_attended = finbert_attended.squeeze(1)
        
        # Gating
        lstm_gate_input = torch.cat([lstm_proj.squeeze(1), lstm_attended], dim=-1)
        finbert_gate_input = torch.cat([finbert_proj.squeeze(1), finbert_attended], dim=-1)
        
        lstm_gate = self.gate_lstm(lstm_gate_input)
        finbert_gate = self.gate_finbert(finbert_gate_input)
        
        # Apply gates
        lstm_gated = lstm_attended * lstm_gate
        finbert_gated = finbert_attended * finbert_gate
        
        # Combine
        combined = torch.cat([lstm_gated, finbert_gated], dim=-1)
        output = self.output_projection(combined)
        
        return output, {
            'lstm_to_finbert': attn_l2f,
            'finbert_to_lstm': attn_f2l
        }

class HybridStockPredictionModel(nn.Module):
    """Complete hybrid model integrating LSTM and FinBERT"""
    
    def __init__(self, config: Config):
        super(HybridStockPredictionModel, self).__init__()
        
        # Model components
        self.lstm_encoder = EnhancedLSTM(**config.LSTM_CONFIG)
        self.finbert_encoder = FinBERTSentimentModel(**config.FINBERT_CONFIG)
        self.cross_attention = CrossModalAttention(**config.INTEGRATION_CONFIG)
        
        # Prediction heads
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.direction_classifier = nn.Linear(128, 2)
        self.volatility_predictor = nn.Linear(128, 1)
    
    def forward(self, structured_data: torch.Tensor, 
                text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor) -> Dict:
        
        # Encode structured data
        lstm_features, lstm_attention = self.lstm_encoder(structured_data)
        
        # Encode text data
        finbert_outputs = self.finbert_encoder(text_input_ids, text_attention_mask)
        finbert_features = finbert_outputs['integrated_features']
        
        # Cross-modal attention
        attended_features, cross_attention = self.cross_attention(
            lstm_features, finbert_features
        )
        
        # Predictions
        price_prediction = self.prediction_head(attended_features)
        direction_logits = self.direction_classifier(attended_features)
        volatility_pred = torch.relu(self.volatility_predictor(attended_features))
        
        return {
            'price_prediction': price_prediction,
            'direction_logits': direction_logits,
            'volatility_prediction': volatility_pred,
            'lstm_attention': lstm_attention,
            'finbert_attention': finbert_outputs['attention_weights'],
            'cross_attention': cross_attention,
            'sentiment_logits': finbert_outputs['sentiment_logits']
        }

# ================================
# TRAINING MODULE
# ================================

class ModelTrainer:
    """Training module for the hybrid model"""
    
    def __init__(self, model: nn.Module, config: Config, use_wandb: bool = False):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizers and schedulers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.price_criterion = nn.MSELoss()
        self.direction_criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        if use_wandb:
            wandb.init(project="hybrid-stock-prediction", config=config.__dict__)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates"""
        training_config = self.config.TRAINING_CONFIG
        
        optimizer_params = [
            {'params': self.model.lstm_encoder.parameters(), 
             'lr': training_config['lstm_lr']},
            {'params': self.model.finbert_encoder.parameters(), 
             'lr': training_config['finbert_lr']},
            {'params': self.model.cross_attention.parameters(), 
             'lr': training_config['integration_lr']},
            {'params': self.model.prediction_head.parameters(), 
             'lr': training_config['prediction_lr']}
        ]
        
        return optim.AdamW(
            optimizer_params,
            weight_decay=training_config['weight_decay']
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.SCHEDULER_CONFIG['T_0'],
            T_mult=self.config.SCHEDULER_CONFIG['T_mult'],
            eta_min=self.config.SCHEDULER_CONFIG['eta_min']
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            structured_data = batch['structured'].to(self.device)
            text_features = batch['text_features'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Create dummy text inputs (in real implementation, use actual tokenized text)
            batch_size = structured_data.size(0)
            dummy_input_ids = torch.zeros(batch_size, 512, dtype=torch.long).to(self.device)
            dummy_attention_mask = torch.ones(batch_size, 512, dtype=torch.long).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(structured_data, dummy_input_ids, dummy_attention_mask)
                
                # Calculate loss
                price_loss = self.price_criterion(
                    outputs['price_prediction'].squeeze(), 
                    targets.squeeze()
                )
                
                # Direction loss
                direction_labels = (targets > 0).long().squeeze()
                direction_loss = self.direction_criterion(
                    outputs['direction_logits'], direction_labels
                )
                
                # Combined loss
                total_batch_loss = (
                    self.config.TRAINING_CONFIG['price_loss_weight'] * price_loss +
                    self.config.TRAINING_CONFIG['direction_loss_weight'] * direction_loss
                )
            
            # Backward pass
            self.scaler.scale(total_batch_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.TRAINING_CONFIG['gradient_clip']
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += total_batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_batch_loss.item(),
                'price_loss': price_loss.item(),
                'dir_loss': direction_loss.item()
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                structured_data = batch['structured'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Dummy text inputs
                batch_size = structured_data.size(0)
                dummy_input_ids = torch.zeros(batch_size, 512, dtype=torch.long).to(self.device)
                dummy_attention_mask = torch.ones(batch_size, 512, dtype=torch.long).to(self.device)
                
                outputs = self.model(structured_data, dummy_input_ids, dummy_attention_mask)
                
                loss = self.price_criterion(
                    outputs['price_prediction'].squeeze(),
                    targets.squeeze()
                )
                
                total_loss += loss.item()
                
                all_predictions.extend(outputs['price_prediction'].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        directional_accuracy = np.mean(pred_direction == true_direction) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        return total_loss / len(val_loader), metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training loop"""
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        num_epochs = self.config.TRAINING_CONFIG['num_epochs']
        patience = self.config.TRAINING_CONFIG['early_stopping_patience']
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val RMSE: {val_metrics['rmse']:.4f}, "
                       f"Directional Acc: {val_metrics['directional_accuracy']:.2f}%")
            
            # Save metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_metrics'].append(val_metrics)
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return training_history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = f'{self.config.MODEL_DIR}/best_model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

# ================================
# EVALUATION MODULE
# ================================

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        predictions = []
        targets = []
        dates = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                structured_data = batch['structured'].to(self.device)
                batch_targets = batch['target'].to(self.device)
                batch_dates = batch['date']
                
                # Dummy text inputs
                batch_size = structured_data.size(0)
                dummy_input_ids = torch.zeros(batch_size, 512, dtype=torch.long).to(self.device)
                dummy_attention_mask = torch.ones(batch_size, 512, dtype=torch.long).to(self.device)
                
                outputs = self.model(structured_data, dummy_input_ids, dummy_attention_mask)
                
                predictions.extend(outputs['price_prediction'].cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                dates.extend(batch_dates)
        
        # Convert to arrays
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(predictions, targets)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'date': dates,
            'actual': targets,
            'predicted': predictions,
            'error': targets - predictions,
            'abs_error': np.abs(targets - predictions)
        })
        
        return {
            'metrics': metrics,
            'results': results_df,
            'predictions': predictions,
            'targets': targets
        }
    
    def _calculate_comprehensive_metrics(self, predictions: np.ndarray, 
                                       targets: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
        
        # Correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        directional_accuracy = np.mean(pred_direction == true_direction) * 100
        
        # Classification metrics for direction
        direction_precision = precision_score(
            true_direction > 0, pred_direction > 0, average='binary'
        )
        direction_recall = recall_score(
            true_direction > 0, pred_direction > 0, average='binary'
        )
        direction_f1 = f1_score(
            true_direction > 0, pred_direction > 0, average='binary'
        )
        
        # Financial metrics
        sharpe_ratio = self._calculate_sharpe_ratio(predictions, targets)
        max_drawdown = self._calculate_max_drawdown(predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'direction_precision': direction_precision,
            'direction_recall': direction_recall,
            'direction_f1': direction_f1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_sharpe_ratio(self, predictions: np.ndarray, 
                              targets: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio based on predictions"""
        returns = predictions
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) * 100
    
    def plot_results(self, evaluation_results: Dict, save_path: str = None):
        """Plot evaluation results"""
        results_df = evaluation_results['results']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results_df['actual'], results_df['predicted'], alpha=0.6)
        axes[0, 0].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        axes[0, 0].set_title('Actual vs Predicted Returns')
        axes[0, 0].grid(True)
        
        # Time series of errors
        axes[0, 1].plot(results_df['date'], results_df['error'])
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].set_title('Prediction Errors Over Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True)
        
        # Error distribution
        axes[1, 0].hist(results_df['error'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Prediction Errors')
        axes[1, 0].grid(True)
        
        # Cumulative returns
        actual_cumret = (1 + results_df['actual']).cumprod()
        pred_cumret = (1 + results_df['predicted']).cumprod()
        
        axes[1, 1].plot(results_df['date'], actual_cumret, label='Actual', linewidth=2)
        axes[1, 1].plot(results_df['date'], pred_cumret, label='Predicted', linewidth=2)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Returns')
        axes[1, 1].set_title('Cumulative Returns Comparison')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ================================
# HYPERPARAMETER OPTIMIZATION
# ================================

class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameters"""
    
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, 
                 config: Config, n_trials: int = 50):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_config = config
        self.n_trials = n_trials
        
    def objective(self, trial):
        """Objective function for optimization"""
        # Suggest hyperparameters
        config = Config()
        
        # LSTM hyperparameters
        config.LSTM_CONFIG['hidden_dim'] = trial.suggest_int('lstm_hidden_dim', 64, 256)
        config.LSTM_CONFIG['num_layers'] = trial.suggest_int('lstm_num_layers', 2, 4)
        config.LSTM_CONFIG['dropout'] = trial.suggest_float('lstm_dropout', 0.1, 0.5)
        
        # Training hyperparameters
        config.TRAINING_CONFIG['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        config.TRAINING_CONFIG['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        config.TRAINING_CONFIG['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
        
        # Create and train model
        try:
            model = HybridStockPredictionModel(config)
            trainer = ModelTrainer(model, config, use_wandb=False)
            
            # Short training for optimization
            config.TRAINING_CONFIG['num_epochs'] = 10
            training_history = trainer.train(self.train_loader, self.val_loader)
            
            # Return best validation loss
            best_val_loss = min(training_history['val_loss'])
            return best_val_loss
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf')
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        return {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'study': study
        }

# ================================
# MAIN EXECUTION PIPELINE
# ================================

def main():
    """Main execution pipeline"""
    logger.info("Starting Hybrid Stock Market Prediction Model Training")
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Data Collection
    logger.info("Step 1: Data Collection")
    
    collector = StockDataCollector(
        symbols=config.SP500_SYMBOLS[:10],  # Use subset for demo
        start_date=config.DATA_START_DATE,
        end_date=config.DATA_END_DATE
    )
    
    price_data, failed_symbols = collector.collect_price_data()
    fundamental_data = collector.collect_fundamental_data()
    collector.save_data(price_data, fundamental_data)
    
    # Load news data (placeholder)
    news_collector = NewsDataCollector(
        news_dataset_path='data/news/financial_news.csv',
        tweet_dataset_path='data/tweets/stock_tweets.csv'
    )
    
    news_data = news_collector.load_news_data()
    tweet_data = news_collector.load_tweet_data()
    
    # Step 2: Data Preprocessing
    logger.info("Step 2: Data Preprocessing")
    
    # Process first symbol as example
    symbol = list(price_data.keys())[0]
    raw_data = price_data[symbol]
    
    # Structured data preprocessing
    struct_preprocessor = StructuredDataPreprocessor()
    processed_data = struct_preprocessor.preprocess(raw_data)
    
    # Text data preprocessing  
    text_preprocessor = TextDataPreprocessor()
    
    # Step 3: Dataset Creation
    logger.info("Step 3: Dataset Creation")
    
    # Create dataset
    dataset = StockDataset(
        structured_data=processed_data,
        text_data=news_data,
        seq_length=config.SEQUENCE_LENGTH
    )
    
    # Split dataset
    train_size = int(len(dataset) * config.TRAIN_RATIO)
    val_size = int(len(dataset) * config.VAL_RATIO)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.TRAINING_CONFIG['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.TRAINING_CONFIG['batch_size'], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.TRAINING_CONFIG['batch_size'], 
        shuffle=False
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Step 4: Model Training
    logger.info("Step 4: Model Training")
    
    # Create model
    model = HybridStockPredictionModel(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trainer = ModelTrainer(model, config, use_wandb=False)
    training_history = trainer.train(train_loader, val_loader)
    
    # Step 5: Model Evaluation
    logger.info("Step 5: Model Evaluation")
    
    # Load best model
    checkpoint_files = [f for f in os.listdir(config.MODEL_DIR) if f.startswith('best_model')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(config.MODEL_DIR, x)))
        checkpoint_path = os.path.join(config.MODEL_DIR, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model, config)
    evaluation_results = evaluator.evaluate_model(test_loader)
    
    # Print results
    metrics = evaluation_results['metrics']
    logger.info("=== FINAL RESULTS ===")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"MAPE: {metrics['mape']:.2f}%")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    logger.info(f"Correlation: {metrics['correlation']:.4f}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Plot results
    evaluator.plot_results(evaluation_results, save_path='results/evaluation_plots.png')
    
    # Save results
    with open('results/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    evaluation_results['results'].to_csv('results/predictions.csv', index=False)
    
    logger.info("Training and evaluation completed successfully!")
    logger.info("Results saved to 'results/' directory")

# ================================
# ABLATION STUDY MODULE
# ================================

class AblationStudy:
    """Conduct ablation studies to analyze component contributions"""
    
    def __init__(self, config: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.results = {}
        
    def run_ablation_study(self):
        """Run complete ablation study"""
        logger.info("Starting Ablation Study")
        
        # 1. Full model (baseline)
        self.results['full_model'] = self._train_and_evaluate_model('full')
        
        # 2. LSTM only (no sentiment)
        self.results['lstm_only'] = self._train_and_evaluate_model('lstm_only')
        
        # 3. FinBERT only (no technical indicators)
        self.results['finbert_only'] = self._train_and_evaluate_model('finbert_only')
        
        # 4. No cross-modal attention
        self.results['no_cross_attention'] = self._train_and_evaluate_model('no_cross_attention')
        
        # 5. Simple concatenation instead of attention
        self.results['simple_concat'] = self._train_and_evaluate_model('simple_concat')
        
        # Save ablation results
        self._save_ablation_results()
        
        # Generate comparison plots
        self._plot_ablation_results()
        
        return self.results
    
    def _train_and_evaluate_model(self, model_type: str) -> Dict:
        """Train and evaluate a specific model configuration"""
        logger.info(f"Training {model_type} model")
        
        # Create model based on type
        if model_type == 'full':
            model = HybridStockPredictionModel(self.config)
        elif model_type == 'lstm_only':
            model = self._create_lstm_only_model()
        elif model_type == 'finbert_only':
            model = self._create_finbert_only_model()
        elif model_type == 'no_cross_attention':
            model = self._create_no_attention_model()
        elif model_type == 'simple_concat':
            model = self._create_simple_concat_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        trainer = ModelTrainer(model, self.config, use_wandb=False)
        training_history = trainer.train(self.train_loader, self.val_loader)
        
        # Evaluate model
        evaluator = ModelEvaluator(model, self.config)
        evaluation_results = evaluator.evaluate_model(self.test_loader)
        
        return {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'model_type': model_type
        }
    
    def _create_lstm_only_model(self) -> nn.Module:
        """Create LSTM-only model"""
        class LSTMOnlyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.lstm_encoder = EnhancedLSTM(**config.LSTM_CONFIG)
                self.prediction_head = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1)
                )
                self.direction_classifier = nn.Linear(64, 2)
                
            def forward(self, structured_data, text_input_ids, text_attention_mask):
                lstm_features, _ = self.lstm_encoder(structured_data)
                return {
                    'price_prediction': self.prediction_head(lstm_features),
                    'direction_logits': self.direction_classifier(lstm_features)
                }
        
        return LSTMOnlyModel(self.config)
    
    def _create_finbert_only_model(self) -> nn.Module:
        """Create FinBERT-only model"""
        class FinBERTOnlyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.finbert_encoder = FinBERTSentimentModel(**config.FINBERT_CONFIG)
                self.prediction_head = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1)
                )
                self.direction_classifier = nn.Linear(64, 2)
                
            def forward(self, structured_data, text_input_ids, text_attention_mask):
                finbert_outputs = self.finbert_encoder(text_input_ids, text_attention_mask)
                features = finbert_outputs['integrated_features']
                return {
                    'price_prediction': self.prediction_head(features),
                    'direction_logits': self.direction_classifier(features)
                }
        
        return FinBERTOnlyModel(self.config)
    
    def _create_no_attention_model(self) -> nn.Module:
        """Create model without cross-modal attention"""
        class NoAttentionModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.lstm_encoder = EnhancedLSTM(**config.LSTM_CONFIG)
                self.finbert_encoder = FinBERTSentimentModel(**config.FINBERT_CONFIG)
                self.prediction_head = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 1)
                )
                self.direction_classifier = nn.Linear(128, 2)
                
            def forward(self, structured_data, text_input_ids, text_attention_mask):
                lstm_features, _ = self.lstm_encoder(structured_data)
                finbert_outputs = self.finbert_encoder(text_input_ids, text_attention_mask)
                finbert_features = finbert_outputs['integrated_features']
                
                # Simple concatenation
                combined_features = torch.cat([lstm_features, finbert_features], dim=-1)
                
                return {
                    'price_prediction': self.prediction_head(combined_features),
                    'direction_logits': self.direction_classifier(combined_features)
                }
        
        return NoAttentionModel(self.config)
    
    def _create_simple_concat_model(self) -> nn.Module:
        """Create model with simple concatenation"""
        return self._create_no_attention_model()  # Same as no attention for this implementation
    
    def _save_ablation_results(self):
        """Save ablation study results"""
        summary_results = {}
        
        for model_type, result in self.results.items():
            metrics = result['evaluation_results']['metrics']
            summary_results[model_type] = {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'directional_accuracy': metrics['directional_accuracy'],
                'correlation': metrics['correlation'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown']
            }
        
        with open('results/ablation_study_results.json', 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(summary_results).T
        comparison_df.to_csv('results/ablation_comparison.csv')
        
        logger.info("Ablation study results saved")
    
    def _plot_ablation_results(self):
        """Plot ablation study results"""
        # Extract metrics for plotting
        model_names = list(self.results.keys())
        rmse_values = [self.results[name]['evaluation_results']['metrics']['rmse'] for name in model_names]
        directional_acc = [self.results[name]['evaluation_results']['metrics']['directional_accuracy'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        bars1 = ax1.bar(model_names, rmse_values, color=['blue', 'red', 'green', 'orange', 'purple'])
        ax1.set_title('RMSE Comparison Across Model Variants')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Directional Accuracy comparison
        bars2 = ax2.bar(model_names, directional_acc, color=['blue', 'red', 'green', 'orange', 'purple'])
        ax2.set_title('Directional Accuracy Comparison')
        ax2.set_ylabel('Directional Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, directional_acc):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/ablation_study_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

