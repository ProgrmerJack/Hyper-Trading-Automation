"""Time‑series regime and volatility forecasting using HuggingFace models.

This module encapsulates a time‑series forecasting model based on
transformer architectures.  It supports loading pre‑trained
HuggingFace models such as the "Time Series Transformer" or the
foundation model "TimesFM".  Optional ONNX acceleration is provided
via the Optimum library when available to reduce inference latency.

The primary class :class:`RegimeForecaster` exposes a simple API for
forecasting the next period's value from a univariate history and
classifying the resulting forecast into qualitative regimes (e.g.
``uptrend`` versus ``downtrend``).  The predicted value itself can be
fed into a meta‑scoring function for finer‑grained weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

# HuggingFace imports are optional.  If they fail we fall back to a
# dummy predictor that always returns zero.
try:
    # ``AutoModelForTimeSeriesForecasting`` is part of the
    # transformers library.  ``ORTModelForTimeSeriesForecasting`` is
    # provided by Optimum for ONNX runtime acceleration.
    from transformers import AutoModelForTimeSeriesForecasting  # type: ignore
    from optimum.onnxruntime import ORTModelForTimeSeriesForecasting  # type: ignore
except Exception:
    AutoModelForTimeSeriesForecasting = None  # type: ignore
    ORTModelForTimeSeriesForecasting = None  # type: ignore


@dataclass
class RegimeForecaster:
    """Wrapper around a time‑series forecasting model.

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
