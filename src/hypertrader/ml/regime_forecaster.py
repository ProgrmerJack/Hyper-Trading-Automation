"""Time‑series regime and volatility forecasting using HuggingFace models + Classical Methods.

This module encapsulates multiple forecasting approaches:
1. Transformer architectures (TimesFM, Time Series Transformer) 
2. Classical volatility models (HAR-RV, GARCH)
3. Hybrid ensemble combining both approaches

The primary classes expose APIs for forecasting volatility and directional regimes
to gate trading decisions based on market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Any, Tuple
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

# HuggingFace imports are optional.  If they fail we fall back to a
# dummy predictor that always returns zero.
try:
    from transformers import AutoModelForTimeSeriesForecasting  # type: ignore
    from optimum.onnxruntime import ORTModelForTimeSeriesForecasting  # type: ignore
except Exception:
    AutoModelForTimeSeriesForecasting = None  # type: ignore
    ORTModelForTimeSeriesForecasting = None  # type: ignore

# Classical volatility model imports
try:
    from arch import arch_model  # GARCH models
except ImportError:
    arch_model = None

logger = logging.getLogger(__name__)


class HARRVModel:
    """
    HAR-RV (Heterogeneous Autoregressive model for Realized Volatility)
    Classical, reliable volatility forecasting model.
    """
    
    def __init__(self, lags: Tuple[int, int, int] = (1, 5, 22)):
        self.daily_lag, self.weekly_lag, self.monthly_lag = lags
        self.coeffs = None
        self.intercept = None
        self.fitted = False
        
    def fit(self, realized_vol: pd.Series) -> None:
        """
        Fit HAR-RV model to realized volatility time series.
        
        Args:
            realized_vol: Daily realized volatility series
        """
        if len(realized_vol) < max(self.daily_lag, self.weekly_lag, self.monthly_lag) + 10:
            logger.warning("Insufficient data for HAR-RV model")
            return
            
        # Prepare HAR regressors
        rv_data = realized_vol.dropna()
        
        # Daily component (RV_t-1)
        rv_daily = rv_data.shift(self.daily_lag)
        
        # Weekly component (average of RV_t-1 to RV_t-5)  
        rv_weekly = rv_data.rolling(self.weekly_lag).mean().shift(1)
        
        # Monthly component (average of RV_t-1 to RV_t-22)
        rv_monthly = rv_data.rolling(self.monthly_lag).mean().shift(1)
        
        # Align data
        start_idx = max(self.daily_lag, self.weekly_lag, self.monthly_lag)
        y = rv_data.iloc[start_idx:].values
        X_daily = rv_daily.iloc[start_idx:].values
        X_weekly = rv_weekly.iloc[start_idx:].values  
        X_monthly = rv_monthly.iloc[start_idx:].values
        
        # Remove NaN values
        mask = ~(np.isnan(y) | np.isnan(X_daily) | np.isnan(X_weekly) | np.isnan(X_monthly))
        y = y[mask]
        X_daily = X_daily[mask]
        X_weekly = X_weekly[mask]
        X_monthly = X_monthly[mask]
        
        if len(y) < 10:
            logger.warning("Insufficient valid data for HAR-RV fitting")
            return
            
        # Design matrix
        X = np.column_stack([np.ones(len(y)), X_daily, X_weekly, X_monthly])
        
        try:
            # OLS estimation
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            self.intercept = coeffs[0]
            self.coeffs = coeffs[1:]
            self.fitted = True
            
            logger.info(f"HAR-RV fitted: α={self.intercept:.4f}, β_d={self.coeffs[0]:.4f}, "
                       f"β_w={self.coeffs[1]:.4f}, β_m={self.coeffs[2]:.4f}")
                       
        except np.linalg.LinAlgError:
            logger.error("HAR-RV fitting failed - singular matrix")
            
    def predict(self, realized_vol: pd.Series, horizon: int = 1) -> float:
        """
        Forecast next period realized volatility.
        
        Args:
            realized_vol: Recent realized volatility data
            horizon: Forecast horizon (only 1-step supported)
            
        Returns:
            Predicted realized volatility
        """
        if not self.fitted or self.coeffs is None:
            return realized_vol.iloc[-1] if len(realized_vol) > 0 else 0.02
            
        rv_data = realized_vol.dropna()
        
        if len(rv_data) < max(self.daily_lag, self.weekly_lag, self.monthly_lag):
            return rv_data.iloc[-1] if len(rv_data) > 0 else 0.02
            
        # Get HAR components
        rv_daily = rv_data.iloc[-self.daily_lag]
        rv_weekly = rv_data.iloc[-self.weekly_lag:].mean()
        rv_monthly = rv_data.iloc[-self.monthly_lag:].mean()
        
        # HAR-RV forecast
        forecast = self.intercept + (
            self.coeffs[0] * rv_daily + 
            self.coeffs[1] * rv_weekly + 
            self.coeffs[2] * rv_monthly
        )
        
        return max(0.001, forecast)  # Ensure positive volatility


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.
    Alternative to HAR-RV with different assumptions.
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns: pd.Series) -> None:
        """
        Fit GARCH model to return series.
        
        Args:
            returns: Return time series
        """
        if arch_model is None:
            logger.warning("arch package not available - GARCH model disabled")
            return
            
        clean_returns = returns.dropna() * 100  # Scale for numerical stability
        
        if len(clean_returns) < 50:
            logger.warning("Insufficient data for GARCH model")
            return
            
        try:
            # GARCH(1,1) specification
            self.model = arch_model(clean_returns, vol='Garch', p=1, q=1, dist='normal')
            self.fitted_model = self.model.fit(disp='off')
            logger.info("GARCH(1,1) model fitted successfully")
            
        except Exception as e:
            logger.error(f"GARCH fitting failed: {e}")
            self.fitted_model = None
            
    def predict(self, horizon: int = 1) -> float:
        """
        Forecast volatility using fitted GARCH model.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Predicted volatility
        """
        if self.fitted_model is None:
            return 0.02  # Default volatility
            
        try:
            forecast = self.fitted_model.forecast(horizon=horizon)
            return np.sqrt(forecast.variance.values[-1, -1]) / 100  # Unscale
        except Exception as e:
            logger.warning(f"GARCH prediction failed: {e}")
            return 0.02


class EnhancedRegimeForecaster:
    """
    Enhanced regime forecaster combining classical models (HAR-RV/GARCH) 
    with modern transformers (TimesFM) for robust volatility and directional forecasting.
    """
    
    def __init__(
        self,
        use_har_rv: bool = True,
        use_garch: bool = True,
        use_timesfm: bool = True,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        self.use_har_rv = use_har_rv
        self.use_garch = use_garch 
        self.use_timesfm = use_timesfm
        
        # Initialize models
        self.har_rv = HARRVModel() if use_har_rv else None
        self.garch = GARCHModel() if use_garch else None
        self.timesfm = RegimeForecaster(
            model_name="google/timesfm-1.0-200m", 
            use_onnx=True
        ) if use_timesfm else None
        
        # Ensemble weights (default equal weighting)
        self.weights = ensemble_weights or {
            'har_rv': 0.4,
            'garch': 0.3, 
            'timesfm': 0.3
        }
        
        self.fitted = False
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit all regime forecasting models.
        
        Args:
            data: OHLCV DataFrame with price data
        """
        if data.empty:
            logger.warning("No data provided for regime forecaster fitting")
            return
            
        # Compute returns and realized volatility
        returns = data['close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std()  # 20-period rolling volatility
        
        models_fitted = 0
        
        # Fit HAR-RV model
        if self.har_rv is not None:
            try:
                self.har_rv.fit(realized_vol)
                if self.har_rv.fitted:
                    models_fitted += 1
                    logger.info("HAR-RV model fitted successfully")
            except Exception as e:
                logger.error(f"HAR-RV fitting failed: {e}")
                
        # Fit GARCH model
        if self.garch is not None:
            try:
                self.garch.fit(returns)
                if self.garch.fitted_model is not None:
                    models_fitted += 1 
                    logger.info("GARCH model fitted successfully")
            except Exception as e:
                logger.error(f"GARCH fitting failed: {e}")
                
        # TimesFM doesn't require explicit fitting
        if self.timesfm is not None:
            models_fitted += 1
            logger.info("TimesFM model ready")
            
        self.fitted = models_fitted > 0
        logger.info(f"Regime forecaster fitted with {models_fitted} models")
        
    def predict_volatility(self, data: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
        """
        Predict volatility using ensemble of models.
        
        Args:
            data: Recent price data
            horizon: Forecast horizon
            
        Returns:
            Dict with individual model predictions and ensemble forecast
        """
        if not self.fitted:
            logger.warning("Models not fitted - returning default volatility")
            return {'ensemble': 0.02, 'regime': 'neutral'}
            
        predictions = {}
        
        # HAR-RV prediction
        if self.har_rv is not None and self.har_rv.fitted:
            returns = data['close'].pct_change().dropna()
            realized_vol = returns.rolling(20).std()
            predictions['har_rv'] = self.har_rv.predict(realized_vol, horizon)
            
        # GARCH prediction  
        if self.garch is not None and self.garch.fitted_model is not None:
            predictions['garch'] = self.garch.predict(horizon)
            
        # TimesFM prediction
        if self.timesfm is not None:
            returns = data['close'].pct_change().dropna()
            predictions['timesfm'] = abs(self.timesfm.forecast(returns.values)) * 0.02
            
        # Ensemble prediction
        if predictions:
            ensemble_vol = 0.0
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                weight = self.weights.get(model_name, 0.0)
                ensemble_vol += weight * pred
                total_weight += weight
                
            if total_weight > 0:
                ensemble_vol /= total_weight
            else:
                ensemble_vol = 0.02
                
            predictions['ensemble'] = ensemble_vol
            
        else:
            predictions['ensemble'] = 0.02
            
        return predictions
        
    def predict_direction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict directional regime (bullish/bearish/neutral).
        
        Args:
            data: Recent price data
            
        Returns:
            Dict with directional predictions and confidence
        """
        if data.empty:
            return {'direction': 'neutral', 'confidence': 0.0}
            
        # Use TimesFM for directional prediction
        if self.timesfm is not None:
            returns = data['close'].pct_change().dropna()
            direction_signal = self.timesfm.forecast(returns.values)
            
            regime = self.timesfm.classify_regime(direction_signal)
            confidence = abs(direction_signal)
            
        else:
            # Fallback: simple momentum
            returns = data['close'].pct_change()
            recent_return = returns.iloc[-5:].mean()
            
            if recent_return > 0.001:
                regime = "uptrend"
                confidence = min(1.0, abs(recent_return) * 10)
            elif recent_return < -0.001:
                regime = "downtrend" 
                confidence = min(1.0, abs(recent_return) * 10)
            else:
                regime = "neutral"
                confidence = 0.1
                
        return {
            'direction': regime,
            'confidence': confidence,
            'signal_strength': confidence
        }
        
    def generate_regime_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate comprehensive regime score for meta-controller.
        
        Args:
            data: Recent market data
            
        Returns:
            Dict with regime metrics for gating decisions
        """
        vol_pred = self.predict_volatility(data)
        dir_pred = self.predict_direction(data)
        
        # Volatility regime (low/medium/high)
        vol_level = vol_pred.get('ensemble', 0.02)
        if vol_level < 0.015:
            vol_regime = 'low_vol'
            vol_score = 0.8  # Favorable for trading
        elif vol_level < 0.03:
            vol_regime = 'medium_vol'
            vol_score = 0.6
        else:
            vol_regime = 'high_vol' 
            vol_score = 0.2  # Unfavorable for trading
            
        # Directional regime
        direction = dir_pred.get('direction', 'neutral')
        dir_confidence = dir_pred.get('confidence', 0.0)
        
        # Combined regime score
        regime_score = 0.6 * vol_score + 0.4 * dir_confidence
        
        return {
            'volatility_forecast': vol_level,
            'volatility_regime': vol_regime,
            'volatility_score': vol_score,
            'direction_forecast': direction,
            'direction_confidence': dir_confidence,
            'regime_score': regime_score,
            'trading_favorable': regime_score > 0.5
        }


@dataclass
class RegimeForecaster:
    """Original wrapper around HuggingFace time‑series forecasting model.

    Parameters
    ----------
    model_name:
        Name of the HuggingFace model checkpoint to load.  Defaults to
        a generic time‑series transformer.  For state‑of‑the‑art
        performance consider ``intel/timesfm-pytorch`` when available.
    horizon:
        Number of future steps to predict.  The module currently
        aggregates the predicted sequence into a single scalar via the
        mean of the last ``horizon`` predictions.
    use_onnx:
        If ``True`` and the Optimum library is available the model is
        loaded in ONNX format for faster inference.  Fallback to
        vanilla PyTorch if ONNX cannot be used.
    device:
        Optional device specification (e.g. "cpu" or "cuda").  This
        parameter is forwarded to the underlying model when supported.

    Attributes
    ----------
    model:
        The underlying forecasting model instance or ``None`` if loading
        fails.
    use_onnx:
        Flag indicating whether ONNX acceleration is active.
    """

    model_name: str = "huggingface/time-series-transformer"
    horizon: int = 1
    use_onnx: bool = False
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the configured model.

        If ONNX is requested and supported the ONNX runtime model is
        loaded first.  Otherwise the standard PyTorch implementation is
        used.  Any exceptions during loading result in ``self.model``
        remaining ``None``, in which case forecasting returns zero.
        """
        # Try ONNX first if requested
        if self.use_onnx and ORTModelForTimeSeriesForecasting is not None:
            try:
                self.model = ORTModelForTimeSeriesForecasting.from_pretrained(
                    self.model_name
                )
                return
            except Exception:
                # fall back to PyTorch
                pass
        if AutoModelForTimeSeriesForecasting is not None:
            try:
                self.model = AutoModelForTimeSeriesForecasting.from_pretrained(
                    self.model_name
                )
                return
            except Exception:
                self.model = None
        else:
            self.model = None

    def forecast(self, series: Union[pd.Series, Sequence[float], np.ndarray]) -> float:
        """Predict the next period value given a univariate time series.

        If no model is loaded or the input series is empty, the method
        returns ``0.0``.  Otherwise the model's generate method is used
        to produce a forecast.  The returned scalar is the mean of the
        last ``horizon`` predicted values.  Users can interpret
        positive values as bullish/uptrend signals and negative values
        as bearish/downtrend signals, or feed the raw value into a
        meta‑scoring function.
        """
        if self.model is None:
            return 0.0
        # Convert input to numpy array shape (1, T)
        if isinstance(series, pd.Series):
            arr = series.dropna().to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(series, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        arr = arr.reshape(1, -1)
        # Build required inputs.  These keys follow the interface used by
        # HuggingFace time‑series models.  We wrap calls in try/except
        # to ensure robustness if the API changes or the model is
        # unavailable.
        try:
            inputs = {
                "past_values": arr,
                "past_observed_mask": np.isfinite(arr).astype(np.float32),
            }
            # Some models may require additional optional keys, but the
            # generate method is designed to handle missing keys gracefully.
            outputs = self.model.generate(
                past_values=inputs["past_values"],
                past_observed_mask=inputs["past_observed_mask"],
                use_cache=True,
            )
            # ``outputs.sequences`` has shape (1, output_length).  We take
            # the mean of the last ``horizon`` predictions to obtain a
            # single scalar.
            seq = getattr(outputs, "sequences", None)
            if seq is None:
                return 0.0
            # Convert to numpy in case it's a torch tensor
            arr_pred = np.asarray(seq)
            last_vals = arr_pred[0, -self.horizon :]
            return float(np.mean(last_vals))
        except Exception:
            return 0.0

    def classify_regime(self, value: float, threshold: float = 0.0) -> str:
        """Classify the forecast value into a qualitative regime.

        Parameters
        ----------
        value:
            Output of :meth:`forecast` representing expected next change.
        threshold:
            Decision boundary between uptrend and downtrend.  Defaults to
            zero.  Users may calibrate this threshold based on the model
            distribution.

        Returns
        -------
        str
            ``"uptrend"`` if ``value >= threshold``, otherwise
            ``"downtrend"``.
        """
        return "uptrend" if value >= threshold else "downtrend"
