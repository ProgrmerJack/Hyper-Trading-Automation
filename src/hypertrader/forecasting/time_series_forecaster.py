import pandas as pd
import numpy as np
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Optional, Dict, Any
import logging

class TimeSeriesForecaster:
    def __init__(self, prophet_params: Optional[Dict[str, Any]] = None, lstm_params: Optional[Dict[str, Any]] = None):
        self.prophet_model = Prophet(**prophet_params) if prophet_params else Prophet()
        self.lstm_model = self._build_lstm_model(lstm_params) if lstm_params else None
        self.hybrid = bool(prophet_params and lstm_params)
        
    def _build_lstm_model(self, params: dict) -> Sequential:
        model = Sequential()
        model.add(LSTM(units=params.get('units', 50), 
                      input_shape=(params.get('time_steps', 10), 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def fit(self, df: pd.DataFrame):
        # Prophet fitting - handle various column name formats
        prophet_df = df.copy()
        
        # Rename columns for Prophet compatibility
        if 'timestamp' in prophet_df.columns and 'price' in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'price': 'y'})
        elif 'date' in prophet_df.columns and 'close' in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={'date': 'ds', 'close': 'y'})
        elif 'time' in prophet_df.columns and 'value' in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={'time': 'ds', 'value': 'y'})
        elif 'datetime' in prophet_df.columns and 'price' in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={'datetime': 'ds', 'price': 'y'})
        elif hasattr(prophet_df.index, 'name') and prophet_df.index.name == 'timestamp':
            prophet_df = prophet_df.reset_index().rename(columns={'timestamp': 'ds', 'price': 'y'})
        else:
            # Try to infer the required columns
            if len(prophet_df.columns) >= 2:
                prophet_df = prophet_df.rename(columns={
                    prophet_df.columns[0]: 'ds',
                    prophet_df.columns[1]: 'y'
                })
        
        # Special case: if 'ds' exists but 'y' doesn't, try to map price columns to 'y'
        if 'ds' in prophet_df.columns and 'y' not in prophet_df.columns:
            if 'close' in prophet_df.columns:
                prophet_df['y'] = prophet_df['close']
            elif 'price' in prophet_df.columns:
                prophet_df['y'] = prophet_df['price']
            elif 'value' in prophet_df.columns:
                prophet_df['y'] = prophet_df['value']
            
        # Validate input data after renaming
        if 'ds' not in prophet_df.columns or 'y' not in prophet_df.columns:
            raise ValueError(
                "DataFrame must contain 'ds' and 'y' columns for Prophet. "
                f"Available columns: {prophet_df.columns.tolist()}"
            )
            
        # Drop NA specifically in the required columns
        prophet_df = prophet_df[['ds', 'y']].dropna()
        
        # Validate we have at least 2 non-NaN rows
        if len(prophet_df) < 2:
            raise ValueError("Insufficient data for forecasting (need at least 2 non-NaN rows after cleaning)")
            
        # Ensure datetime type
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Fit the model
        self.prophet_model.fit(prophet_df[['ds', 'y']])
        
        # LSTM fitting if hybrid mode
        if self.hybrid and self.lstm_model is not None:
            prophet_forecast = self.prophet_model.predict(prophet_df)
            
            # Try to get price data for residuals
            if 'close' in df.columns:
                price_values = df['close'].values
            elif 'y' in prophet_df.columns:
                price_values = prophet_df['y'].values
            else:
                price_values = prophet_df.iloc[:, 1].values
                
            residuals = price_values - prophet_forecast['yhat'].values[:len(price_values)]
            
            # Prepare LSTM data
            X_list = []
            y_list = []
            for i in range(len(residuals) - 10):
                X_list.append(residuals[i:i+10])
                y_list.append(residuals[i+10])
            
            X = np.array(X_list).reshape((len(X_list), 10, 1))
            y = np.array(y_list)
            
            # Train LSTM
            self.lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    def predict(self, periods: int) -> pd.DataFrame:
        # Prophet forecast
        future = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future)
        
        # Hybrid forecast with LSTM residuals
        if self.hybrid:
            residuals = self.prophet_model.history['y'] - self.prophet_model.history['yhat']
            
            # Predict residuals
            last_residuals = residuals[-10:].values.reshape((1, 10, 1))
            residual_forecast = []
            for _ in range(periods):
                next_res = self.lstm_model.predict(last_residuals)[0,0]
                residual_forecast.append(next_res)
                last_residuals = np.roll(last_residuals, -1)
                last_residuals[0, -1, 0] = next_res
            
            # Combine forecasts
            forecast['yhat'] = forecast['yhat'] + np.array(residual_forecast)
        
        return forecast[['ds', 'yhat']].tail(periods)
