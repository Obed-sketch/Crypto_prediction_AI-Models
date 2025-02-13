# AI Price Prediction System Architecture
# graph TD
#    A[Data Collection] --> B[Feature Engineering]
#    B --> C[Model Training]
#    C --> D[Prediction Engine]
#    D --> E[Risk Analysis]
 #   E --> F[Deployment]
#    F --> G[User Interface]

# Data Collection & Preprocessing
import yfinance as yf
import pandas as pd

def fetch_crypto_data(ticker='DOGE-USD', start='2018-01-01'):
    data = yf.download(ticker, start=start)
    
    # Add technical indicators
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['Volatility'] = data['Close'].pct_change().rolling(7).std()
    
    # Merge with sentiment data
    reddit_sentiment = scrape_reddit_sentiment('dogecoin')
    data = pd.merge(data, reddit_sentiment, left_index=True, right_index=True)
    
    return data.dropna()

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# 2. Hybrid Prediction Model (LSTM + Prophet)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet

def create_hybrid_model(train_data):
    # LSTM Component
    lstm_model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(30, 6)),
        LSTM(50),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(train_data, epochs=50, batch_size=32)
    
    # Prophet Component
    prophet_data = train_data[['Close']].reset_index()
    prophet_data.columns = ['ds', 'y']
    prophet_model = Prophet(weekly_seasonality=True)
    prophet_model.fit(prophet_data)
    
    return {'lstm': lstm_model, 'prophet': prophet_model}

def hybrid_predict(models, input_data):
    lstm_pred = models['lstm'].predict(input_data)
    future = models['prophet'].make_future_dataframe(periods=1)
    prophet_pred = models['prophet'].predict(future)['yhat'].iloc[-1]
    return (lstm_pred[0][0] * 0.6) + (prophet_pred * 0.4)

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

class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float

@app.post("/predict")
async def predict(request: PredictionRequest):
    data = load_latest_data(request.ticker)
    prediction = hybrid_predict(models[request.ticker], data)
    risk_assessment = calculate_risk(prediction, data['Close'].iloc[-1])
    
    return {
        "prediction": f"${prediction:.4f}",
        "risk": risk_assessment['risk_category'],
        "recommendation": risk_assessment['recommended_action'],
        "time_horizon": "7-day forecast"
    }

#Deployment Architecture
#graph LR
   # A[Cloud Storage] --> B[Preprocessing]
   # B --> C[Prediction Models]
   # C --> D[API Layer]
   # D --> E[Mobile App]
   # D --> F[Web Dashboard]
  #  G[Live Crypto Feeds] --> B
  #  H[Social Media APIs] --> B
