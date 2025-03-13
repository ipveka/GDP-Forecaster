"""
VariableForecaster module for forecasting individual economic variables.
"""

import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('variable_forecaster')

class VariableForecaster:
    """Class to forecast individual variables using appropriate time series methods."""
    
    def __init__(self):
        """Initialize the VariableForecaster."""
        self.models = {}
        self.scalers = {}
    
    def forecast_population(self, historical_data: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Forecast population using growth rate method.
        
        Args:
            historical_data: Historical population data
            horizon: Number of years to forecast
            
        Returns:
            Series with forecasted population
        """
        # Calculate average annual growth rate over the last 5 years
        historical_data = historical_data.sort_index()
        growth_rates = historical_data.pct_change().dropna()
        
        # Use the average of last 5 years or all available if less
        n_years = min(5, len(growth_rates))
        avg_growth_rate = growth_rates.iloc[-n_years:].mean()
        
        # If no valid growth rate (e.g., only one data point), use global average of 1.1%
        if pd.isna(avg_growth_rate) or avg_growth_rate == 0:
            avg_growth_rate = 0.011  # Global average
        
        # Generate future dates
        last_date = historical_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(years=1),
            periods=horizon,
            freq='YS'  # Year start
        )
        
        # Forecast
        last_value = historical_data.iloc[-1]
        forecast_values = [last_value]
        
        for _ in range(horizon):
            next_value = forecast_values[-1] * (1 + avg_growth_rate)
            forecast_values.append(next_value)
        
        forecast = pd.Series(
            forecast_values[1:],
            index=forecast_index
        )
        
        return forecast
    
    def forecast_arima(self, historical_data: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Forecast a variable using ARIMA model.
        
        Args:
            historical_data: Historical data
            horizon: Number of years to forecast
            
        Returns:
            Series with forecasted values
        """
        # Fill missing values with linear interpolation
        data = historical_data.copy()
        data = data.interpolate(method='linear')
        
        # If still have missing values at ends, use backfill/forward fill
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        # Fit ARIMA(2,1,2) model - a reasonable default for many economic series
        try:
            model = ARIMA(data, order=(2, 1, 2))
            model_fit = model.fit()
            
            # Generate future dates
            last_date = data.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=horizon,
                freq='YS'
            )
            
            # Forecast
            forecast = model_fit.forecast(steps=horizon)
            forecast = pd.Series(forecast, index=forecast_index)
            
            return forecast
            
        except Exception as e:
            logger.warning(f"ARIMA forecast failed: {str(e)}. Using exponential smoothing instead.")
            return self.forecast_exponential_smoothing(historical_data, horizon)
    
    def forecast_exponential_smoothing(self, historical_data: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Forecast a variable using exponential smoothing.
        
        Args:
            historical_data: Historical data
            horizon: Number of years to forecast
            
        Returns:
            Series with forecasted values
        """
        # Fill missing values
        data = historical_data.copy()
        data = data.interpolate(method='linear')
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        # Check if we have at least 3 data points
        if len(data) < 3:
            # Not enough data, use simple percentage increase
            last_value = data.iloc[-1]
            # Assume 2% annual change
            forecast_index = pd.date_range(
                start=data.index[-1] + pd.DateOffset(years=1),
                periods=horizon,
                freq='YS'  # Year start
            )
            forecast_values = [last_value * (1.02 ** i) for i in range(1, horizon + 1)]
            return pd.Series(forecast_values, index=forecast_index)
        
        # For economic data, often trend=add and seasonal=None is appropriate for annual data
        try:
            model = ExponentialSmoothing(
                data, 
                trend='add',  # Additive trend
                seasonal=None,  # No seasonality for annual data
                initialization_method='estimated'
            )
            model_fit = model.fit()
            
            # Generate future dates
            last_date = data.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=horizon,
                freq='YS'  # Year start
            )
            
            # Forecast
            forecast = model_fit.forecast(horizon)
            forecast = pd.Series(forecast, index=forecast_index)
            
            return forecast
            
        except Exception as e:
            logger.warning(f"Exponential smoothing forecast failed: {str(e)}. Using simple growth method.")
            # Fallback to simple growth method
            avg_change = data.pct_change().mean()
            if pd.isna(avg_change):
                avg_change = 0.02  # Default 2% change
                
            last_value = data.iloc[-1]
            forecast_index = pd.date_range(
                start=data.index[-1] + pd.DateOffset(years=1),
                periods=horizon,
                freq='YS'  # Year start
            )
            forecast_values = [last_value * (1 + avg_change) ** i for i in range(1, horizon + 1)]
            return pd.Series(forecast_values, index=forecast_index)
    
    def select_forecast_method(self, variable_name: str, historical_data: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Select appropriate forecasting method based on variable type.
        
        Args:
            variable_name: Name of the variable to forecast
            historical_data: Historical data
            horizon: Number of years to forecast
            
        Returns:
            Series with forecasted values
        """
        # Choose method based on variable characteristics
        if "POP" in variable_name:
            # Population
            return self.forecast_population(historical_data, horizon)
        elif any(term in variable_name for term in ["GINI", "HDI"]):
            # Slower changing structural indicators
            return self.forecast_exponential_smoothing(historical_data, horizon)
        else:
            # Economic indicators with potential cyclical patterns
            return self.forecast_arima(historical_data, horizon)