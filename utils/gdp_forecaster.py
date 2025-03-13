import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import logging
import os
import warnings
from typing import Dict, Tuple, List, Union, Optional
from tabulate import tabulate
from scipy import stats
import time
import copy

from .data_collector import DataCollector
from .variable_forecaster import VariableForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gdp_forecaster')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")
warnings.filterwarnings("ignore", message="No frequency information was provided")

class GDPForecaster:
    """Main class for GDP forecasting using ElasticNet regression."""
    
    def __init__(self):
        """Initialize the GDP forecaster."""
        self.data_collector = DataCollector()
        self.variable_forecaster = VariableForecaster()
        self.model = None
        self.pipeline = None
        self.scaler = None
        self.feature_names = []
        self.country_code = None
        self.historical_data = None
        self.forecasted_features = None
        self.gdp_forecast = None
        self.official_forecast = None
        self.test_predictions = None
        self.test_actuals = None
        self.feature_selection_threshold = 0.001  # Min coefficient threshold
        self.backtest_results = None
        
    def load_data(self, country_code: str) -> pd.DataFrame:
        """
        Load historical data for a country.
        
        Args:
            country_code: ISO 3-letter country code
            
        Returns:
            DataFrame with historical data
        """
        self.country_code = country_code
        self.historical_data = self.data_collector.fetch_world_bank_data(country_code)
        return self.historical_data
    
    def prepare_features(self, target_col: str = 'NY.GDP.MKTP.CD', data: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training with improved preprocessing.
        
        Args:
            target_col: The column to use as the target (GDP)
            data: Optional specific dataset to use (for backtesting)
            
        Returns:
            Tuple of (X features, y target)
        """
        if data is None:
            if self.historical_data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            data = self.historical_data
        
        # Drop columns with too many missing values
        min_years = 5  # Require at least 5 years of data
        cols_to_keep = [col for col in data.columns 
                         if data[col].count() >= min_years]
        
        if target_col not in cols_to_keep:
            raise ValueError(f"Target column {target_col} does not have enough data")
        
        # Keep target and feature columns
        df = data[cols_to_keep].copy()
        
        # Improved handling of missing values with interpolation first
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaNs, use median imputation
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Drop rows where target is still NA after filling
        df = df.dropna(subset=[target_col])
        
        # Handle outliers by capping extreme values
        for col in df.columns:
            if col != target_col:  # Don't modify the target
                q1 = df[col].quantile(0.05)
                q3 = df[col].quantile(0.95)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Save all features instead of removing correlated variables
        features = [col for col in df.columns if col != target_col]
        self.feature_names = features
        
        # Separate features and target
        X = df[features]
        y = df[target_col]
        
        return X, y
    
    def _simple_forecast(self, historical_data: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Simple trend-based forecast for problematic series.
        
        Args:
            historical_data: Historical time series data
            horizon: Forecast horizon in years
            
        Returns:
            Forecasted values
        """
        # Clean data
        data = historical_data.dropna()
        
        if len(data) <= 1:
            # Not enough data, use global average growth of 2%
            last_value = data.iloc[-1] if len(data) > 0 else 0
            growth_rate = 0.02
        else:
            # Calculate average growth over available data
            growth_rates = data.pct_change().dropna()
            
            # Remove extreme outliers
            valid_rates = growth_rates[np.abs(stats.zscore(growth_rates, nan_policy='omit')) < 3]
            
            if len(valid_rates) > 0:
                growth_rate = valid_rates.mean()
            else:
                growth_rate = 0.02  # Default to 2% growth
        
        # Generate future dates with yearly frequency
        last_date = historical_data.index[-1]
        forecast_index = pd.date_range(
            start=pd.Timestamp(year=last_date.year+1, month=1, day=1),
            periods=horizon,
            freq='YS'  # Year start frequency
        )
        
        # Generate forecast
        last_value = historical_data.iloc[-1]
        forecast_values = [last_value * (1 + growth_rate) ** i for i in range(1, horizon + 1)]
        
        return pd.Series(forecast_values, index=forecast_index)

    def train_model(self, test_years: int = 3, custom_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Train the model with increased weight on recent observations.
        
        Args:
            test_years: Number of years to hold out for testing
            custom_data: Optional data to use for training (for backtesting)
            
        Returns:
            Dictionary with performance metrics
        """
        # Prepare data
        X, y = self.prepare_features(data=custom_data)
        print(f"Features used for training: {list(X.columns)}")
        
        if len(X) <= test_years + 3:
            logger.warning(f"Limited data available ({len(X)} years). Consider reducing test_years.")
            test_years = min(1, len(X) - 3)  # Ensure at least 3 years for training
            if test_years <= 0:
                logger.warning(f"Not enough data for testing. Using simplified training approach.")
                test_years = 0  # Use all data for training if there's not enough
                
        X, y = X.sort_index(), y.sort_index()
        
        # Split data
        if test_years > 0:
            train_X, train_y = X.iloc[:-test_years], y.iloc[:-test_years]
            test_X, test_y = X.iloc[-test_years:], y.iloc[-test_years:]
            self.test_actuals = pd.Series(test_y.values, index=test_y.index)
        else:
            train_X, train_y = X, y
            test_X, test_y = None, None
        
        # Apply sample weights to emphasize recent data - increased importance
        # Use exponential weighting instead of linear to put more emphasis on recent data
        sample_weights = np.exp(np.linspace(0, 1, len(train_y)))
        sample_weights = sample_weights / np.mean(sample_weights)
        
        # Define pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(random_state=42, max_iter=10000))
        ])
        
        # Skip cross-validation for very small datasets
        if len(train_X) < 5:
            logger.info(f"Small dataset detected ({len(train_X)} samples). Using simplified model without CV.")
            # Use basic ElasticNet with default parameters
            self.pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000))
            ])
            
            self.pipeline.fit(train_X, train_y, model__sample_weight=sample_weights)
            self.model = self.pipeline.named_steps['model']
            self.scaler = self.pipeline.named_steps['scaler']
            
            if test_X is not None and test_y is not None:
                return self._evaluate_model(test_X, test_y)
            else:
                # If no test set, return basic metrics
                return {'MSE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 1.0}
        
        # Define hyperparameter grid for larger datasets
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'model__l1_ratio': [0.2, 0.5, 0.8],
            'model__tol': [1e-4],
            'model__selection': ['cyclic']
        }
        
        # For very small datasets, reduce grid further
        if len(train_X) < 15:
            param_grid = {
                'model__alpha': [0.1, 1.0, 10.0],
                'model__l1_ratio': [0.5],
                'model__tol': [1e-4],
                'model__selection': ['cyclic']
            }
        
        # Cross-validation - adjust splits based on data size
        n_splits = min(3, max(2, len(train_X) // 3))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_search.fit(train_X, train_y, model__sample_weight=sample_weights)
            
            self.pipeline = grid_search.best_estimator_
            self.model = self.pipeline.named_steps['model']
            self.scaler = self.pipeline.named_steps['scaler']
            
            best_params = grid_search.best_params_
            print(f"Best hyperparameters found: {best_params}")
            logger.info(f"Best parameters: {best_params}")
            
            # Evaluate model
            if test_X is not None and test_y is not None:
                return self._evaluate_model(test_X, test_y)
            else:
                # If no test set, return basic metrics
                return {'MSE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 1.0}
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            logger.info("Falling back to simpler model...")
            
            # Fallback to Ridge model
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=100, random_state=42))
            ])
            self.pipeline.fit(train_X, train_y, model__sample_weight=sample_weights)
            self.model = self.pipeline.named_steps['model']
            self.scaler = self.pipeline.named_steps['scaler']
            
            if test_X is not None and test_y is not None:
                return self._evaluate_model(test_X, test_y)
            else:
                # If no test set, return basic metrics
                return {'MSE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 1.0}

    def _evaluate_model(self, test_X: pd.DataFrame, test_y: pd.Series) -> Dict[str, float]:
        """Helper function to evaluate model performance."""
        test_predictions = self.pipeline.predict(test_X)
        self.test_predictions = pd.Series(test_predictions, index=test_y.index)
        
        mse = mean_squared_error(test_y, test_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test_y, test_predictions) * 100
        r2 = 1 - (np.sum((test_y - test_predictions) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2))
        
        return {'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    def forecast_features(self, horizon: int = 5, cutoff_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Forecast all features for the specified horizon.
        
        Args:
            horizon: Number of years to forecast
            cutoff_date: Optional cutoff date for backtesting
            
        Returns:
            DataFrame with forecasted features
        """
        if self.historical_data is None or not self.feature_names:
            raise ValueError("Model not trained. Call load_data() and train_model() first.")
        
        # Filter data if cutoff date is provided
        historical_data = self.historical_data
        if cutoff_date is not None:
            historical_data = historical_data.loc[:cutoff_date]
        
        # Forecast each feature with robust error handling
        all_forecasts = {}
        failed_features = []
        
        for feature in self.feature_names:
            start_time = time.time()
            historical_series = historical_data[feature].copy()
            
            try:
                # Set explicit yearly frequency to avoid warnings
                historical_series.index = pd.DatetimeIndex(historical_series.index)
                historical_series = historical_series.asfreq('YS')
                
                # Check if series has enough data for forecasting
                if len(historical_series.dropna()) < 5:
                    logger.warning(f"Limited data for {feature}. Using simple trend forecast.")
                    forecast = self._simple_forecast(historical_series, horizon)
                else:
                    # First try regular forecasting
                    try:
                        forecast = self.variable_forecaster.select_forecast_method(
                            feature, historical_series, horizon)
                    except Exception as e:
                        # If it fails, fall back to simple trend forecast
                        logger.warning(f"Forecasting failed for {feature}: {str(e)}. Using simple trend.")
                        forecast = self._simple_forecast(historical_series, horizon)
                
                all_forecasts[feature] = forecast
                logger.debug(f"Forecast for {feature} completed in {time.time() - start_time:.2f} seconds")
            
            except Exception as e:
                logger.error(f"Failed to forecast {feature}: {str(e)}")
                failed_features.append(feature)
        
        # If any features failed, remove them from the model
        if failed_features:
            self.feature_names = [f for f in self.feature_names if f not in failed_features]
            logger.warning(f"Removed {len(failed_features)} features that could not be forecasted")
        
        # Combine into a single dataframe
        if not all_forecasts:
            raise ValueError("Failed to forecast any features. Check your data.")
            
        self.forecasted_features = pd.DataFrame(all_forecasts)
        
        # Ensure consistent frequency in the index
        self.forecasted_features.index = pd.DatetimeIndex(self.forecasted_features.index)
        self.forecasted_features = self.forecasted_features.asfreq('YS')
        
        return self.forecasted_features
    
    def forecast_gdp(self, horizon: int = 5, cutoff_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Forecast GDP using the trained model with improved error handling.
        
        Args:
            horizon: Number of years to forecast
            cutoff_date: Optional cutoff date for backtesting
            
        Returns:
            DataFrame with GDP forecast
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.forecasted_features is None or cutoff_date is not None:
            self.forecast_features(horizon, cutoff_date)
        
        # Handle case where some features couldn't be forecasted
        missing_features = set(self.feature_names) - set(self.forecasted_features.columns)
        if missing_features:
            logger.warning(f"Some features are missing from forecasts: {missing_features}")
            # Update feature names to match what's available
            self.feature_names = [f for f in self.feature_names if f in self.forecasted_features.columns]
        
        # Check if we have enough features
        if len(self.feature_names) == 0:
            raise ValueError("No features available for forecasting. Check feature forecasts.")
        
        # Reorder columns to match the expected feature order
        forecast_features = self.forecasted_features[self.feature_names].copy()
        
        # Make predictions using the pipeline (handles scaling internally)
        gdp_forecast = self.pipeline.predict(forecast_features)
        
        # Create DataFrame with results
        self.gdp_forecast = pd.DataFrame({
            'GDP_Forecast': gdp_forecast
        }, index=forecast_features.index)
        
        # Add growth rate
        last_historical_gdp = None
        if cutoff_date is not None:
            last_historical_gdp = self.historical_data.loc[:cutoff_date, 'NY.GDP.MKTP.CD'].iloc[-1]
        else:
            last_historical_gdp = self.historical_data['NY.GDP.MKTP.CD'].iloc[-1]
            
        self.gdp_forecast['Previous_GDP'] = [
            last_historical_gdp if i == 0 else self.gdp_forecast['GDP_Forecast'].iloc[i-1]
            for i in range(len(self.gdp_forecast))
        ]
        self.gdp_forecast['Growth_Rate'] = (
            (self.gdp_forecast['GDP_Forecast'] / self.gdp_forecast['Previous_GDP']) - 1
        ) * 100
        
        return self.gdp_forecast
    
    def run_rolling_backtests(self, n_years: int = 3) -> pd.DataFrame:
        """
        Run rolling backtests for the last N available years in the data.
        
        Args:
            n_years: Number of years to backtest
            
        Returns:
            DataFrame with backtest results
        """
        if self.historical_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Get years available in data
        years = sorted(list(set([date.year for date in self.historical_data.index])))
        
        # Ensure we have enough years to backtest
        if len(years) < n_years + 3:  # Need at least 3 years to train
            logger.warning(f"Not enough years for {n_years} backtests. Using {len(years)-3} backtests.")
            n_years = max(1, len(years) - 3)
            
        # Setup backtest results storage
        years_to_test = years[-n_years:]
        actuals = {}
        predictions = {}
        metrics = {}
        
        # For each year to test
        for i, test_year in enumerate(years_to_test):
            logger.info(f"Running backtest for year {test_year}")
            
            # Find cutoff date (last date in the year before test_year)
            cutoff_dates = self.historical_data.index[self.historical_data.index.year < test_year]
            if len(cutoff_dates) == 0:
                logger.warning(f"No data before {test_year}. Skipping.")
                continue
                
            cutoff_date = cutoff_dates[-1]
            
            # Get actual value for the test year
            actual_dates = self.historical_data.index[self.historical_data.index.year == test_year]
            if len(actual_dates) == 0:
                logger.warning(f"No actual data for {test_year}. Skipping.")
                continue
                
            actual_date = actual_dates[0]
            actual_value = self.historical_data.loc[actual_date, 'NY.GDP.MKTP.CD']
            
            # Get historical data up to cutoff date
            historical_subset = self.historical_data.loc[:cutoff_date].copy()
            
            # Create a temporary forecaster for this test
            temp_forecaster = GDPForecaster()
            temp_forecaster.country_code = self.country_code
            temp_forecaster.historical_data = historical_subset
            
            # Skip cross-validation for very small datasets
            try:
                # Prepare features directly for small datasets
                X, y = temp_forecaster.prepare_features()
                min_years_for_cv = 10  # Minimum years needed for cross-validation
                
                if len(X) < min_years_for_cv:
                    logger.warning(f"Limited data for {test_year} ({len(X)} years). Using simplified model without CV.")
                    # Use direct training without cross-validation
                    sample_weights = np.exp(np.linspace(0, 1, len(X)))
                    sample_weights = sample_weights / np.mean(sample_weights)
                    
                    # Simple pipeline
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('model', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000))
                    ])
                    
                    # Fit directly without CV
                    pipeline.fit(X, y, model__sample_weight=sample_weights)
                    temp_forecaster.pipeline = pipeline
                    temp_forecaster.model = pipeline.named_steps['model']
                    temp_forecaster.feature_names = list(X.columns)
                else:
                    # Use regular training with CV for larger datasets
                    temp_forecaster.train_model(test_years=0)  # Use all data for training
                
                # Forecast 1 year ahead
                forecast_result = temp_forecaster.forecast_gdp(horizon=1)
                prediction = forecast_result['GDP_Forecast'].iloc[0]
                
                # Store results
                actuals[test_year] = actual_value
                predictions[test_year] = prediction
                
                # Calculate metrics
                error = actual_value - prediction
                pct_error = (error / actual_value) * 100
                metrics[test_year] = {
                    'Actual': actual_value,
                    'Predicted': prediction,
                    'Error': error,
                    'Percent_Error': pct_error
                }
                
                logger.info(f"  Actual: {actual_value:.2f}, Predicted: {prediction:.2f}, Error: {error:.2f} ({pct_error:.2f}%)")
                
            except Exception as e:
                logger.error(f"Backtest failed for {test_year}: {str(e)}")
                continue
                
        # Combine results into a DataFrame
        if not metrics:
            logger.warning("No successful backtests. Check your data.")
            return pd.DataFrame()
            
        backtest_df = pd.DataFrame(metrics).T
        backtest_df.index.name = 'Year'
        
        # Calculate aggregate metrics
        mape = np.mean(np.abs(backtest_df['Percent_Error']))
        rmse = np.sqrt(np.mean(backtest_df['Error'] ** 2))
        
        logger.info(f"Backtest Results - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
        
        # Store results
        self.backtest_results = backtest_df
        
        return backtest_df
    
    def get_official_forecasts(self) -> pd.DataFrame:
        """
        Get official GDP forecasts from sources like IMF.
        
        Returns:
            DataFrame with official forecasts
        """
        imf_forecasts = self.data_collector.fetch_imf_forecasts(self.country_code)
        
        # Align with our forecast period
        if self.gdp_forecast is not None:
            common_index = self.gdp_forecast.index.intersection(imf_forecasts.index)
            self.official_forecast = imf_forecasts.loc[common_index]
        else:
            self.official_forecast = imf_forecasts
        
        return self.official_forecast
    
    def plot_forecast(self, show_history_years: int = 10, include_backtests: bool = True) -> plt.Figure:
        """
        Plot the GDP forecast along with historical data and official forecasts.
        
        Args:
            show_history_years: Number of historical years to show (default: 10)
            include_backtests: Whether to include backtest results
            
        Returns:
            Matplotlib figure
        """
        if self.gdp_forecast is None:
            raise ValueError("No forecast available. Call forecast_gdp() first.")
        
        # Get historical GDP
        historical_gdp = self.historical_data['NY.GDP.MKTP.CD'].copy()
        
        # Limit history to specified years for cleaner plot
        if show_history_years > 0 and len(historical_gdp) > show_history_years:
            historical_gdp = historical_gdp.iloc[-show_history_years:]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot historical GDP
        ax1.plot(historical_gdp.index, historical_gdp.values, 'o-', 
                color='blue', label='Historical GDP')
        
        # Plot test predictions if available
        if self.test_predictions is not None:
            ax1.plot(self.test_predictions.index, self.test_predictions.values, 's-', 
                    color='purple', label='Backtested Predictions')
            ax1.plot(self.test_actuals.index, self.test_actuals.values, 'o', 
                    color='black', alpha=0.6, label='Actual Test Values')
        
        # Plot rolling backtest results if available and requested
        if include_backtests and self.backtest_results is not None:
            backtest_years = self.backtest_results.index.astype(int)
            backtest_dates = [pd.Timestamp(f"{year}-01-01") for year in backtest_years]
            
            ax1.plot(backtest_dates, self.backtest_results['Predicted'].values, 'D', 
                    color='orange', markersize=8, label='Rolling Backtest Predictions')
            ax1.plot(backtest_dates, self.backtest_results['Actual'].values, 'D', 
                    color='green', markersize=8, label='Rolling Backtest Actuals')
        
        # Plot forecasted GDP
        ax1.plot(self.gdp_forecast.index, self.gdp_forecast['GDP_Forecast'].values, 'o-', 
                color='red', label='Forecasted GDP')
        
        # Add official forecasts if available
        if self.official_forecast is not None:
            # Check if IMF forecast contains GDP value or just growth rate
            if 'IMF_GDP_Value' in self.official_forecast.columns:
                ax1.plot(self.official_forecast.index, 
                        self.official_forecast['IMF_GDP_Value'].values, 's-',
                        color='green', label='IMF Forecast')
            elif 'IMF_GDP_Growth' in self.official_forecast.columns:
                # We need to calculate the GDP values from growth rates
                if len(historical_gdp) > 0:
                    last_gdp = historical_gdp.iloc[-1]
                    imf_gdp_values = [last_gdp]
                    
                    for growth in self.official_forecast['IMF_GDP_Growth']:
                        next_gdp = imf_gdp_values[-1] * (1 + growth / 100)
                        imf_gdp_values.append(next_gdp)
                    
                    imf_gdp_values = imf_gdp_values[1:]  # Remove the initial last_gdp
                    ax1.plot(self.official_forecast.index, imf_gdp_values, 's-',
                            color='green', label='IMF Forecast')
        
        # Format the first subplot
        ax1.set_title('GDP Forecast', fontsize=15)
        ax1.set_ylabel('GDP (current US$)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis to show only years
        locator = mdates.YearLocator()
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show billions and start at 0
        def billions(x, pos):
            return f'${x / 1e9:.1f}B'
        
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(billions))
        ax1.set_ylim(bottom=0)  # Set y-axis to start at 0
        
        # Calculate historical growth for the growth subplot
        if len(historical_gdp) > 1:
            historical_growth = historical_gdp.pct_change() * 100
            ax2.plot(historical_growth.index, historical_growth.values, 'o-', 
                    color='blue', label='Historical Growth')
        
        # Add forecasted growth
        ax2.plot(self.gdp_forecast.index, self.gdp_forecast['Growth_Rate'].values, 'o-', 
                color='red', label='Forecasted Growth')
        
        # Add official growth forecasts
        if self.official_forecast is not None and 'IMF_GDP_Growth' in self.official_forecast.columns:
            ax2.plot(self.official_forecast.index, 
                    self.official_forecast['IMF_GDP_Growth'].values, 's-',
                    color='green', label='IMF Growth Forecast')
        
        # Format the second subplot
        ax2.set_title('GDP Growth Rate', fontsize=15)
        ax2.set_ylabel('Annual Growth Rate (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis to show only years
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show percentage
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add horizontal line at 0% for growth rate
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table of forecast results and covariables.
        
        Returns:
            DataFrame with forecast results and comparison with official sources
        """
        if self.gdp_forecast is None:
            raise ValueError("No forecast available. Call forecast_gdp() first.")
        
        # Create comparison table
        comparison = self.gdp_forecast[['GDP_Forecast', 'Growth_Rate']].copy()
        
        # Add forecasted features
        if self.forecasted_features is not None:
            for col in self.forecasted_features.columns:
                # Use more readable column names in the table
                readable_name = col.replace('.', '_')
                comparison[readable_name] = self.forecasted_features[col]
        
        # Add official forecasts if available
        if self.official_forecast is not None:
            if 'IMF_GDP_Growth' in self.official_forecast.columns:
                comparison['IMF_Growth_Rate'] = self.official_forecast['IMF_GDP_Growth']
                comparison['Growth_Difference'] = comparison['Growth_Rate'] - comparison['IMF_Growth_Rate']
        
        # Format GDP values to billions
        comparison['GDP_Forecast'] = comparison['GDP_Forecast'] / 1e9
        comparison.rename(columns={'GDP_Forecast': 'GDP_Forecast_Billions'}, inplace=True)
        
        return comparison
    
    def print_forecast_summary(self):
        """Print a summary of the forecast results."""
        if self.gdp_forecast is None:
            raise ValueError("No forecast available. Call forecast_gdp() first.")
        
        comparison = self.generate_comparison_table()
        
        print(f"\nGDP Forecast Summary for {self.country_code}")
        print("=" * 80)
        
        # Format table for display
        table_data = []
        for idx, row in comparison.iterrows():
            year = idx.year
            gdp = f"${row['GDP_Forecast_Billions']:.2f}B"
            growth = f"{row['Growth_Rate']:.2f}%"
            
            if 'IMF_Growth_Rate' in row:
                imf_growth = f"{row['IMF_Growth_Rate']:.2f}%"
                diff = f"{row['Growth_Difference']:.2f}%"
                table_data.append([year, gdp, growth, imf_growth, diff])
            else:
                table_data.append([year, gdp, growth, 'N/A', 'N/A'])
        
        headers = ['Year', 'GDP Forecast', 'Growth Rate', 'IMF Growth', 'Difference']
        print(tabulate(table_data, headers=headers, tablefmt='pretty'))
        
        # Print key covariables
        print("\nKey Economic Indicators (Forecasted):")
        print("-" * 80)
        
        for idx, row in comparison.iterrows():
            year = idx.year
            indicators = []
            
            # Select a few important indicators to display
            for col in comparison.columns:
                if col in ['GDP_Forecast_Billions', 'Growth_Rate', 'IMF_Growth_Rate', 'Growth_Difference']:
                    continue
                
                # Format based on indicator type
                if 'POP' in col:
                    indicators.append(f"{col}: {row[col]/1e6:.2f}M")
                elif 'GINI' in col:
                    indicators.append(f"{col}: {row[col]:.1f}")
                elif any(x in col for x in ['ZS', 'ZG']):
                    indicators.append(f"{col}: {row[col]:.2f}%")
                else:
                    indicators.append(f"{col}: {row[col]:.2f}")
            
            print(f"{year}: {', '.join(indicators)}")
        
        # Print model performance if available
        if self.test_predictions is not None:
            print("\nBacktest Performance:")
            print("-" * 80)
            
            mape = mean_absolute_percentage_error(self.test_actuals, self.test_predictions) * 100
            print(f"Mean Absolute Percentage Error: {mape:.2f}%")
            
            mse = mean_squared_error(self.test_actuals, self.test_predictions)
            rmse = np.sqrt(mse)
            print(f"Root Mean Squared Error: ${rmse/1e9:.2f}B")
            
            # Calculate R-squared
            ss_total = np.sum((self.test_actuals - np.mean(self.test_actuals)) ** 2)
            ss_residual = np.sum((self.test_actuals - self.test_predictions) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            print(f"R-squared: {r2:.4f}")
        
        # Print rolling backtest results if available
        if self.backtest_results is not None:
            print("\nRolling Backtest Results:")
            print("-" * 80)
            
            backtest_table = []
            for year, row in self.backtest_results.iterrows():
                backtest_table.append([
                    year,
                    f"${row['Actual']/1e9:.2f}B",
                    f"${row['Predicted']/1e9:.2f}B",
                    f"${row['Error']/1e9:.2f}B",
                    f"{row['Percent_Error']:.2f}%"
                ])
            
            backtest_headers = ['Year', 'Actual GDP', 'Predicted GDP', 'Error', 'Percent Error']
            print(tabulate(backtest_table, headers=backtest_headers, tablefmt='pretty'))
            
            # Summary metrics
            mape = np.mean(np.abs(self.backtest_results['Percent_Error']))
            rmse = np.sqrt(np.mean(self.backtest_results['Error'] ** 2))
            print(f"Rolling Backtest MAPE: {mape:.2f}%")
            print(f"Rolling Backtest RMSE: ${rmse/1e9:.2f}B")
        
        print("=" * 80)
    
    def export_results(self, output_dir: str = './output'):
        """
        Export forecast results to CSV files and save plots.
        
        Args:
            output_dir: Directory to save output files
        """
        if self.gdp_forecast is None:
            raise ValueError("No forecast available. Call forecast_gdp() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export forecast data
        forecast_file = os.path.join(output_dir, f"{self.country_code}_gdp_forecast.csv")
        self.gdp_forecast.to_csv(forecast_file)
        logger.info(f"GDP forecast exported to {forecast_file}")
        
        # Export feature forecasts
        if self.forecasted_features is not None:
            features_file = os.path.join(output_dir, f"{self.country_code}_features_forecast.csv")
            self.forecasted_features.to_csv(features_file)
            logger.info(f"Feature forecasts exported to {features_file}")
        
        # Export comparison table
        comparison = self.generate_comparison_table()
        comparison_file = os.path.join(output_dir, f"{self.country_code}_forecast_comparison.csv")
        comparison.to_csv(comparison_file)
        logger.info(f"Forecast comparison exported to {comparison_file}")
        
        # Export historical data
        if self.historical_data is not None:
            historical_file = os.path.join(output_dir, f"{self.country_code}_historical_data.csv")
            self.historical_data.to_csv(historical_file)
            logger.info(f"Historical data exported to {historical_file}")
        
        # Save forecast plot
        plot_file = os.path.join(output_dir, f"{self.country_code}_gdp_forecast_plot.png")
        fig = self.plot_forecast()
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Forecast plot saved to {plot_file}")
        
        # Export backtest results if available
        if self.backtest_results is not None:
            backtest_file = os.path.join(output_dir, f"{self.country_code}_rolling_backtest_results.csv")
            self.backtest_results.to_csv(backtest_file)
            logger.info(f"Rolling backtest results exported to {backtest_file}")
        
        return output_dir
    
    def get_model_coefficients(self) -> pd.DataFrame:
        """
        Get the coefficients of the trained model.
        
        Returns:
            DataFrame with feature names and their coefficients
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })
        
        # Add normalized importance (absolute value)
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df['Normalized_Importance'] = coef_df['Abs_Coefficient'] / coef_df['Abs_Coefficient'].sum()
        
        # Sort by importance
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        return coef_df
    
    def plot_feature_importance(self) -> plt.Figure:
        """
        Plot feature importance based on model coefficients.
        
        Returns:
            Matplotlib figure showing feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get coefficients
        coef_df = self.get_model_coefficients()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot feature importance
        y_pos = range(len(coef_df))
        ax.barh(y_pos, coef_df['Coefficient'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_df['Feature'])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Importance in GDP Prediction Model')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig