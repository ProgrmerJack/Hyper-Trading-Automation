import pytest
import pandas as pd
from hypertrader.forecasting.time_series_forecaster import TimeSeriesForecaster

class TestTimeSeriesForecaster:
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = [100 + i for i in range(100)]
        return pd.DataFrame({'close': prices}, index=dates)
        
    def test_forecast_creation(self, sample_data):
        """Test forecasting model creation"""
        forecaster = TimeSeriesForecaster()
        forecaster.fit(sample_data)
        forecast = forecaster.predict(7)
        assert len(forecast) == 7
        assert 'forecast' in forecast.columns
