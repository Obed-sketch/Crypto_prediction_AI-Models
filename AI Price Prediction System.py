# STATUS (updated Nov 2026): this file began as an architecture BLUEPRINT.
# The five functions the blueprint called but never defined --
# scrape_reddit_sentiment(), historical_max_drawdown(),
# current_volatility(), get_live_sentiment() and load_latest_data() --
# plus the global `models` registry, are now IMPLEMENTED below (marked
# "NEW (Nov 2026)"). The blueprint's structure, model shapes, blend
# weights (0.6 LSTM / 0.4 Prophet) and risk weights (0.4/0.3/0.3) are
# unchanged. The script now runs end-to-end: it trains on public
# DOGE-USD data from Yahoo Finance and answers one /predict request
# in-process (see the __main__ block -- no public server is started).
# One honest limitation: Reddit's public JSON endpoints generally
# refuse unauthenticated scraping, so the sentiment feed falls back to
# a neutral 0.0 series with a printed notice; live Reddit sentiment
# would need OAuth credentials (e.g. via praw).
#
# AI Price Prediction System Architecture
# graph TD
#    A[Data Collection] --> B[Feature Engineering]
#    B --> C[Model Training]
#    C --> D[Prediction Engine]
#    D --> E[Risk Analysis]
#    E --> F[Deployment]
#    F --> G[User Interface]

# Data Collection & Preprocessing
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Columns the LSTM consumes: 5 price/volume series + merged sentiment.
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']
SEQ_LEN = 30           # LSTM look-back window, from the blueprint's input_shape
DEFAULT_TICKER = 'DOGE-USD'
HISTORY_START = '2018-01-01'


def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# NEW (Nov 2026): the five functions the blueprint called but never wrote,
# plus small caches so the no-argument risk helpers share one download.
# ---------------------------------------------------------------------------

_price_cache = {}
_sentiment_cache = {}


def _price_history(ticker=DEFAULT_TICKER):
    """Download (once) and cache the raw OHLCV history for a ticker."""
    if ticker not in _price_cache:
        data = yf.download(ticker, start=HISTORY_START, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        _price_cache[ticker] = data
    return _price_cache[ticker]


def scrape_reddit_sentiment(subreddit='dogecoin'):
    """Daily mean Reddit sentiment for a subreddit, as a DataFrame with a
    'Sentiment' column indexed by date (daily, 2018-01-01 to today).

    Live path: Reddit's public search JSON, scored with VADER. Reddit
    usually refuses unauthenticated scraping (HTTP 403/429); when that
    happens this prints a notice and returns a NEUTRAL (0.0) series so
    the pipeline still runs -- live sentiment needs OAuth credentials.
    """
    if subreddit in _sentiment_cache:
        return _sentiment_cache[subreddit]
    daily = pd.date_range(HISTORY_START, pd.Timestamp.now('UTC').normalize().tz_localize(None), freq='D')
    scores = None
    try:
        resp = requests.get(
            f'https://www.reddit.com/r/{subreddit}/search.json',
            params={'q': subreddit, 'restrict_sr': 1, 'sort': 'new', 'limit': 100},
            headers={'User-Agent': 'crypto-prediction-demo/0.1 (educational)'},
            timeout=20,
        )
        if resp.status_code == 200:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            posts = resp.json()['data']['children']
            by_day = {}
            for p in posts:
                day = pd.to_datetime(p['data']['created_utc'], unit='s').normalize()
                by_day.setdefault(day, []).append(
                    sia.polarity_scores(p['data']['title'])['compound'])
            if by_day:
                scores = pd.Series({d: float(np.mean(v)) for d, v in by_day.items()})
        else:
            print(f"[sentiment] Reddit public JSON returned HTTP {resp.status_code}; "
                  "live sentiment needs OAuth -- using neutral 0.0 sentiment.")
    except Exception as exc:  # network blocked, bad JSON, etc.
        print(f"[sentiment] Reddit fetch failed ({exc}); "
              "using neutral 0.0 sentiment.")
    if scores is None:
        sentiment = pd.DataFrame({'Sentiment': 0.0}, index=daily)
    else:
        sentiment = scores.reindex(daily).ffill().fillna(0.0).to_frame('Sentiment')
    _sentiment_cache[subreddit] = sentiment
    return sentiment


def historical_max_drawdown(ticker=DEFAULT_TICKER):
    """Worst peak-to-trough decline of the price history, as a positive
    fraction (0.55 means a 55% drawdown happened at some point)."""
    close = _price_history(ticker)['Close']
    running_max = close.cummax()
    return float((1 - close / running_max).max())


def current_volatility(ticker=DEFAULT_TICKER):
    """Standard deviation of the last 30 days of daily returns."""
    close = _price_history(ticker)['Close']
    return float(close.pct_change().tail(30).std())


def get_live_sentiment(subreddit='dogecoin'):
    """Most recent sentiment level: mean of the last 7 daily values."""
    sentiment = scrape_reddit_sentiment(subreddit)['Sentiment']
    return float(sentiment.tail(7).mean())


def load_latest_data(ticker=DEFAULT_TICKER):
    """The processed feature frame for a ticker (prices + indicators +
    merged sentiment), i.e. fetch_crypto_data() for that ticker."""
    return fetch_crypto_data(ticker=ticker)


# ---------------------------------------------------------------------------
# Blueprint code, with the two internal inconsistencies repaired (marked).
# ---------------------------------------------------------------------------

def fetch_crypto_data(ticker='DOGE-USD', start='2018-01-01'):
    data = _price_history(ticker).copy()

    # Add technical indicators
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['Volatility'] = data['Close'].pct_change().rolling(7).std()

    # Merge with sentiment data
    subreddit = 'dogecoin' if ticker == 'DOGE-USD' else 'cryptocurrency'
    reddit_sentiment = scrape_reddit_sentiment(subreddit)
    data = pd.merge(data, reddit_sentiment, left_index=True, right_index=True)

    return data.dropna()


# 2. Hybrid Prediction Model (LSTM + Prophet)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler


def _make_sequences(scaled, close_idx, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, close_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def create_hybrid_model(train_data):
    # LSTM Component
    lstm_model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(30, 6)),
        LSTM(50),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')
    # REPAIRED (Nov 2026): the blueprint called lstm_model.fit(train_data)
    # with no targets, which cannot train. Build supervised sequences
    # from the frame instead: each sample is 30 days of the 6 FEATURES,
    # the target is the next day's (scaled) Close.
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_data[FEATURES].astype(float))
    close_idx = FEATURES.index('Close')
    X_train, y_train = _make_sequences(scaled, close_idx)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Prophet Component
    prophet_data = train_data[['Close']].reset_index()
    prophet_data.columns = ['ds', 'y']
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds']).dt.tz_localize(None)
    prophet_model = Prophet(weekly_seasonality=True)
    prophet_model.fit(prophet_data)

    return {'lstm': lstm_model, 'prophet': prophet_model,
            'scaler': scaler, 'close_idx': close_idx}


def hybrid_predict(models, input_data):
    # REPAIRED (Nov 2026): the blueprint passed the whole frame to the
    # LSTM; it needs the last 30 rows of FEATURES, scaled, as one sample.
    window = input_data[FEATURES].astype(float).tail(SEQ_LEN)
    scaled = models['scaler'].transform(window)
    lstm_input = scaled.reshape(1, SEQ_LEN, len(FEATURES)).astype(np.float32)
    lstm_scaled = float(models['lstm'].predict(lstm_input, verbose=0)[0][0])
    ci = models['close_idx']
    lstm_pred = lstm_scaled * (models['scaler'].data_max_[ci] - models['scaler'].data_min_[ci]) \
        + models['scaler'].data_min_[ci]
    future = models['prophet'].make_future_dataframe(periods=1)
    prophet_pred = models['prophet'].predict(future)['yhat'].iloc[-1]
    return (lstm_pred * 0.6) + (prophet_pred * 0.4)


# 3. Risk Analysis Engine
def calculate_risk(prediction, current_price):
    risk_metrics = {
        'max_drawdown': historical_max_drawdown(),
        'volatility_index': current_volatility(),
        'sentiment_score': get_live_sentiment()
    }

    risk_score = (risk_metrics['max_drawdown'] * 0.4 +
                  risk_metrics['volatility_index'] * 0.3 +
                  risk_metrics['sentiment_score'] * 0.3)

    return {
        'predicted_price': prediction,
        'risk_category': 'High' if risk_score > 0.7 else
                         'Medium' if risk_score > 0.4 else 'Low',
        'recommended_action': 'Sell' if prediction < current_price else
                              'Hold' if (prediction - current_price)/current_price < 0.05
                              else 'Buy'
    }


# 4. Deployment with FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# NEW (Nov 2026): the global model registry the endpoint indexes.
# Models are trained lazily on first use (training takes minutes).
models = {}


def get_models(ticker):
    if ticker not in models:
        print(f"[models] training hybrid model for {ticker} (first request)...")
        models[ticker] = create_hybrid_model(load_latest_data(ticker))
    return models[ticker]


class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float


@app.post("/predict")
async def predict(request: PredictionRequest):
    data = load_latest_data(request.ticker)
    prediction = hybrid_predict(get_models(request.ticker), data)
    risk_assessment = calculate_risk(prediction, float(data['Close'].iloc[-1]))

    return {
        "prediction": f"${prediction:.4f}",
        "risk": risk_assessment['risk_category'],
        "recommendation": risk_assessment['recommended_action'],
        "time_horizon": "7-day forecast"
    }


# Deployment Architecture
# graph LR
#    A[Cloud Storage] --> B[Preprocessing]
#    B --> C[Prediction Models]
#    C --> D[API Layer]
#    D --> E[Mobile App]
#    D --> F[Web Dashboard]
#    G[Live Crypto Feeds] --> B
#    H[Social Media APIs] --> B

if __name__ == "__main__":
    # End-to-end smoke run (Nov 2026): warm the registry for DOGE-USD,
    # then answer one /predict request IN-PROCESS via FastAPI's
    # TestClient -- no public server is started. For real deployment:
    #     uvicorn "AI Price Prediction System:app" --host 0.0.0.0 --port 8000
    # (first request per ticker triggers training and takes minutes).
    from fastapi.testclient import TestClient

    get_models(DEFAULT_TICKER)
    client = TestClient(app)
    resp = client.post("/predict", json={"ticker": DEFAULT_TICKER,
                                         "investment_amount": 1000.0})
    print("POST /predict ->", resp.status_code)
    print(resp.json())
