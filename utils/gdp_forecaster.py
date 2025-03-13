"""
GDPForecaster module for forecasting GDP using ElasticNet regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import logging
import os
import warnings
from typing import Dict, Tuple, List
from tabulate import tabulate

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


class GDPForecaster:
    """Main class for GDP forecasting using ElasticNet regression."""
    
    def __init__(self):
        """Initialize the GDP forecaster."""
        self.data_collector = DataCollector()
        self.variable_forecaster = VariableForecaster()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.country_code = None
        self.historical_data = None
        self.forecasted_features = None
        self.gdp_forecast = None
        self.official_forecast = None
        self.test_predictions = None
        self.test_actuals = None
    
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
    
    def prepare_features(self, target_col: str = 'NY.GDP.MKTP.CD') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training.
        
        Args:
            target_col: The column to use as the target (GDP)
            
        Returns:
            Tuple of (X features, y target)
        """
        if self.historical_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Drop columns with too many missing values
        min_years = 5
        cols_to_keep = [col for col in self.historical_data.columns if self.historical_data[col].count() >= min_years]
        
        if target_col not in cols_to_keep:
            raise ValueError(f"Target column {target_col} does not have enough data")
        
        # Keep target and feature columns
        df = self.historical_data[cols_to_keep].copy()
        
        # Forward fill and backward fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows where target is still NA after filling
        df = df.dropna(subset=[target_col])
        
        # Select features (exclude the target)
        features = [col for col in df.columns if col != target_col]
        self.feature_names = features
        
        # Separate features and target
        X = df[features]
        y = df[target_col]
        
        return X, y
    
    def train_model(self, test_years: int = 3) -> Dict[str, float]:
        """
        Train the ElasticNet model with grid search for hyperparameters.
        
        Args:
            test_years: Number of years to hold out for testing
            
        Returns:
            Dictionary with performance metrics
        """
        X, y = self.prepare_features()
        
        # Sort by date
        X = X.sort_index()
        y = y.sort_index()
        
        # Split into train and test sets based on years
        train_X = X.iloc[:-test_years]
        train_y = y.iloc[:-test_years]
        test_X = X.iloc[-test_years:]
        test_y = y.iloc[-test_years:]
        
        # Store test data for later evaluation
        self.test_actuals = pd.Series(test_y.values, index=test_y.index)
        
        # Scale features
        self.scaler = StandardScaler()
        train_X_scaled = self.scaler.fit_transform(train_X)
        test_X_scaled = self.scaler.transform(test_X)
        
        # Define parameter grid for grid search
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000],
            'tol': [1e-4]
        }
        
        # Initialize the model
        base_model = ElasticNet(random_state=42)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search with time series cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit the grid search
        grid_search.fit(train_X_scaled, train_y)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Test the model
        test_predictions = self.model.predict(test_X_scaled)
        self.test_predictions = pd.Series(test_predictions, index=test_y.index)
        
        # Calculate metrics
        mse = mean_squared_error(test_y, test_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test_y, test_predictions) * 100
        
        # Calculate R-squared
        ss_total = np.sum((test_y - np.mean(test_y)) ** 2)
        ss_residual = np.sum((test_y - test_predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        return metrics
    
    def forecast_features(self, horizon: int = 5) -> pd.DataFrame:
        """
        Forecast all features for the specified horizon.
        
        Args:
            horizon: Number of years to forecast
            
        Returns:
            DataFrame with forecasted features
        """
        if self.historical_data is None or not self.feature_names:
            raise ValueError("Model not trained. Call load_data() and train_model() first.")
        
        # Forecast each feature
        all_forecasts = {}
        for feature in self.feature_names:
            historical_series = self.historical_data[feature].copy()
            forecast = self.variable_forecaster.select_forecast_method(
                feature, historical_series, horizon)
            all_forecasts[feature] = forecast
        
        # Combine into a single dataframe
        self.forecasted_features = pd.DataFrame(all_forecasts)
        return self.forecasted_features
    
    def forecast_gdp(self, horizon: int = 5) -> pd.DataFrame:
        """
        Forecast GDP using the trained model and forecasted features.
        
        Args:
            horizon: Number of years to forecast
            
        Returns:
            DataFrame with GDP forecast
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.forecasted_features is None:
            self.forecast_features(horizon)
        
        # Scale the forecasted features
        features_scaled = self.scaler.transform(self.forecasted_features)
        
        # Make predictions
        gdp_forecast = self.model.predict(features_scaled)
        
        # Create DataFrame with results
        self.gdp_forecast = pd.DataFrame({
            'GDP_Forecast': gdp_forecast
        }, index=self.forecasted_features.index)
        
        # Add growth rate
        last_historical_gdp = self.historical_data['NY.GDP.MKTP.CD'].iloc[-1]
        self.gdp_forecast['Previous_GDP'] = [
            last_historical_gdp if i == 0 else self.gdp_forecast['GDP_Forecast'].iloc[i-1]
            for i in range(len(self.gdp_forecast))
        ]
        self.gdp_forecast['Growth_Rate'] = (
            (self.gdp_forecast['GDP_Forecast'] / self.gdp_forecast['Previous_GDP']) - 1
        ) * 100
        
        return self.gdp_forecast
    
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
    
    def plot_forecast(self, show_history_years: int = 5) -> plt.Figure:
        """
        Plot the GDP forecast along with historical data and official forecasts.
        
        Args:
            show_history_years: Number of historical years to show
            
        Returns:
            Matplotlib figure
        """
        if self.gdp_forecast is None:
            raise ValueError("No forecast available. Call forecast_gdp() first.")
        
        # Get historical GDP
        historical_gdp = self.historical_data['NY.GDP.MKTP.CD'].copy()
        
        # Limit history to specified years for cleaner plot
        if show_history_years > 0:
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
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show billions
        def billions(x, pos):
            return f'${x / 1e9:.1f}B'
        
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(billions))
        
        # Plot growth rates in the second subplot
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
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show percentage
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
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
        
        # Export backtesting results if available
        if self.test_predictions is not None:
            backtest_df = pd.DataFrame({
                'Actual': self.test_actuals,
                'Predicted': self.test_predictions,
                'Error': self.test_actuals - self.test_predictions,
                'Percent_Error': ((self.test_actuals - self.test_predictions) / self.test_actuals) * 100
            })
            
            backtest_file = os.path.join(output_dir, f"{self.country_code}_backtest_results.csv")
            backtest_df.to_csv(backtest_file)
            logger.info(f"Backtesting results exported to {backtest_file}")
            
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
    
    def evaluate_model_stability(self, n_permutations: int = 10) -> Dict[str, float]:
        """
        Evaluate model stability by training with different train/test splits.
        
        Args:
            n_permutations: Number of different train/test splits to evaluate
            
        Returns:
            Dictionary with stability metrics
        """
        if self.historical_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        X, y = self.prepare_features()
        
        # Sort by date
        X = X.sort_index()
        y = y.sort_index()
        
        # Store metrics
        mape_scores = []
        r2_scores = []
        
        # Run permutations
        for i in range(n_permutations):
            # Create a different test set each time
            test_size = max(1, int(len(X) * 0.2))  # Use at least 20% for testing
            test_start = np.random.randint(len(X) - test_size - 5, len(X) - test_size)
            test_end = test_start + test_size
            
            # Split data
            train_X = pd.concat([X.iloc[:test_start], X.iloc[test_end:]])
            train_y = pd.concat([y.iloc[:test_start], y.iloc[test_end:]])
            test_X = X.iloc[test_start:test_end]
            test_y = y.iloc[test_start:test_end]
            
            # Scale features
            scaler = StandardScaler()
            train_X_scaled = scaler.fit_transform(train_X)
            test_X_scaled = scaler.transform(test_X)
            
            # Train model
            model = ElasticNet(
                alpha=0.1, 
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42 + i  # Different seed each time
            )
            model.fit(train_X_scaled, train_y)
            
            # Evaluate
            test_predictions = model.predict(test_X_scaled)
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(test_y, test_predictions) * 100
            mape_scores.append(mape)
            
            # Calculate R-squared
            ss_total = np.sum((test_y - np.mean(test_y)) ** 2)
            ss_residual = np.sum((test_y - test_predictions) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            r2_scores.append(r2)
        
        # Calculate stability metrics
        results = {
            'mean_mape': np.mean(mape_scores),
            'std_mape': np.std(mape_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'coef_variation_mape': np.std(mape_scores) / np.mean(mape_scores) * 100,
            'stability_score': 100 - (np.std(mape_scores) / np.mean(mape_scores) * 100)
        }
        
        return results
    
    def perform_sensitivity_analysis(self, feature_name: str, variation_pct: float = 10.0, horizon: int = 5) -> plt.Figure:
        """
        Perform sensitivity analysis by varying a specific feature.
        
        Args:
            feature_name: Name of the feature to vary
            variation_pct: Percentage to vary the feature by
            horizon: Forecast horizon
            
        Returns:
            Matplotlib figure showing sensitivity analysis
        """
        if self.model is None or self.forecasted_features is None:
            raise ValueError("Model not trained or features not forecasted. Call train_model() and forecast_gdp() first.")
            
        if feature_name not in self.forecasted_features.columns:
            raise ValueError(f"Feature {feature_name} not found in forecasted features.")
        
        # Create base, upper, and lower scenarios
        base_features = self.forecasted_features.copy()
        upper_features = self.forecasted_features.copy()
        lower_features = self.forecasted_features.copy()
        
        # Apply variations
        upper_features[feature_name] *= (1 + variation_pct / 100)
        lower_features[feature_name] *= (1 - variation_pct / 100)
        
        # Scale features
        base_scaled = self.scaler.transform(base_features)
        upper_scaled = self.scaler.transform(upper_features)
        lower_scaled = self.scaler.transform(lower_features)
        
        # Make predictions
        base_forecast = self.model.predict(base_scaled)
        upper_forecast = self.model.predict(upper_scaled)
        lower_forecast = self.model.predict(lower_scaled)
        
        # Create DataFrames for results
        base_df = pd.DataFrame({
            'GDP_Forecast': base_forecast,
            'Scenario': 'Base'
        }, index=base_features.index)
        
        upper_df = pd.DataFrame({
            'GDP_Forecast': upper_forecast,
            'Scenario': f'{feature_name} +{variation_pct}%'
        }, index=base_features.index)
        
        lower_df = pd.DataFrame({
            'GDP_Forecast': lower_forecast,
            'Scenario': f'{feature_name} -{variation_pct}%'
        }, index=base_features.index)
        
        # Combine results
        results = pd.concat([base_df, upper_df, lower_df])
        
        # Calculate growth rates
        last_historical_gdp = self.historical_data['NY.GDP.MKTP.CD'].iloc[-1]
        scenarios = results['Scenario'].unique()
        
        for scenario in scenarios:
            scenario_data = results[results['Scenario'] == scenario].copy()
            previous_values = [last_historical_gdp]
            previous_values.extend(scenario_data['GDP_Forecast'].values[:-1])
            
            scenario_data['Previous_GDP'] = previous_values
            scenario_data['Growth_Rate'] = (
                (scenario_data['GDP_Forecast'] / scenario_data['Previous_GDP']) - 1
            ) * 100
            
            # Update the results
            results.loc[results['Scenario'] == scenario, 'Growth_Rate'] = scenario_data['Growth_Rate'].values
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot GDP forecasts
        for scenario in scenarios:
            scenario_data = results[results['Scenario'] == scenario]
            ax1.plot(scenario_data.index, scenario_data['GDP_Forecast'] / 1e9, 'o-', 
                    label=scenario)
        
        # Format the first subplot
        ax1.set_title(f'GDP Forecast Sensitivity to {feature_name}', fontsize=15)
        ax1.set_ylabel('GDP (Billions of USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Plot growth rates
        for scenario in scenarios:
            scenario_data = results[results['Scenario'] == scenario]
            ax2.plot(scenario_data.index, scenario_data['Growth_Rate'], 'o-', 
                    label=scenario)
        
        # Format the second subplot
        ax2.set_title('GDP Growth Rate Sensitivity', fontsize=15)
        ax2.set_ylabel('Annual Growth Rate (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show percentage
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        plt.tight_layout()
        return fig
    
    @classmethod
    def compare_countries(cls, country_codes: List[str], horizon: int = 5, metric: str = 'growth') -> plt.Figure:
        """
        Compare GDP forecasts across multiple countries.
        
        Args:
            country_codes: List of ISO 3-letter country codes to compare
            horizon: Forecast horizon
            metric: Metric to compare ('growth' or 'gdp')
            
        Returns:
            Matplotlib figure showing country comparison
        """
        if not country_codes:
            raise ValueError("No country codes provided.")
        
        # Forecast data for each country
        country_data = {}
        
        for code in country_codes:
            try:
                forecaster = cls()
                forecaster.load_data(code)
                forecaster.train_model()
                gdp_forecast = forecaster.forecast_gdp(horizon=horizon)
                
                if metric == 'growth':
                    country_data[code] = gdp_forecast['Growth_Rate']
                else:  # metric == 'gdp'
                    # Normalize GDP to initial year for comparison
                    initial_gdp = gdp_forecast['GDP_Forecast'].iloc[0]
                    country_data[code] = gdp_forecast['GDP_Forecast'] / initial_gdp * 100
            except Exception as e:
                logger.warning(f"Failed to forecast for {code}: {str(e)}")
        
        if not country_data:
            raise ValueError("Failed to generate forecasts for any country.")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot data for each country
        for code, data in country_data.items():
            ax.plot(data.index, data.values, 'o-', linewidth=2, label=code)
        
        # Format the plot
        if metric == 'growth':
            ax.set_title('GDP Growth Rate Comparison', fontsize=15)
            ax.set_ylabel('Annual Growth Rate (%)', fontsize=12)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        else:  # metric == 'gdp'
            ax.set_title('Normalized GDP Comparison (Base Year = 100)', fontsize=15)
            ax.set_ylabel('Normalized GDP', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        return fig
        
    def run_regional_analysis(self, region_countries: Dict[str, List[str]], horizon: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Run analysis for multiple regions and countries.
        
        Args:
            region_countries: Dictionary mapping region names to lists of country codes
            horizon: Forecast horizon
            
        Returns:
            Dictionary with regional analysis results
        """
        region_results = {}
        
        for region_name, country_codes in region_countries.items():
            logger.info(f"Analyzing region: {region_name}")
            
            # Store country level data
            country_forecasts = {}
            country_features = {}
            
            for country_code in country_codes:
                try:
                    # Create a new forecaster for each country
                    forecaster = GDPForecaster()
                    forecaster.load_data(country_code)
                    forecaster.train_model()
                    gdp_forecast = forecaster.forecast_gdp(horizon=horizon)
                    
                    # Store results
                    country_forecasts[country_code] = gdp_forecast
                    country_features[country_code] = forecaster.forecasted_features
                    
                    logger.info(f"  Successfully forecasted GDP for {country_code}")
                except Exception as e:
                    logger.warning(f"  Failed to forecast for {country_code}: {str(e)}")
            
            if country_forecasts:
                # Calculate regional average growth rates
                growth_rates = []
                
                for country, forecast in country_forecasts.items():
                    growth_rates.append(forecast['Growth_Rate'])
                
                region_growth = pd.concat(growth_rates, axis=1)
                region_growth.columns = list(country_forecasts.keys())
                
                # Calculate average
                region_growth['Average'] = region_growth.mean(axis=1)
                
                # Store in results
                region_results[region_name] = region_growth
        
        return region_results
    
    def plot_confidence_intervals(self, confidence_level: float = 0.90, horizon: int = 5) -> plt.Figure:
        """
        Plot GDP forecast with confidence intervals based on model stability.
        
        Args:
            confidence_level: Confidence level (0.0-1.0)
            horizon: Forecast horizon
            
        Returns:
            Matplotlib figure with confidence intervals
        """
        if self.gdp_forecast is None:
            raise ValueError("No forecast available. Call forecast_gdp() first.")
        
        # Evaluate model stability to estimate variance
        stability = self.evaluate_model_stability(n_permutations=10)
        
        # Calculate z-score based on confidence level
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Estimate forecast standard deviation based on historical MAPE
        forecast_std = self.gdp_forecast['GDP_Forecast'] * (stability['mean_mape'] / 100)
        
        # Calculate confidence intervals
        lower_bound = self.gdp_forecast['GDP_Forecast'] - z_score * forecast_std
        upper_bound = self.gdp_forecast['GDP_Forecast'] + z_score * forecast_std
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get historical GDP
        historical_gdp = self.historical_data['NY.GDP.MKTP.CD'].copy()
        historical_gdp = historical_gdp.iloc[-5:]  # Last 5 years
        
        # Plot historical GDP
        ax.plot(historical_gdp.index, historical_gdp.values / 1e9, 'o-', 
                color='blue', label='Historical GDP')
        
        # Plot forecasted GDP with confidence intervals
        ax.plot(self.gdp_forecast.index, self.gdp_forecast['GDP_Forecast'] / 1e9, 'o-', 
                color='red', label='Forecasted GDP')
        
        # Plot confidence intervals
        ax.fill_between(
            self.gdp_forecast.index, 
            lower_bound / 1e9, 
            upper_bound / 1e9, 
            color='red', 
            alpha=0.2, 
            label=f'{int(confidence_level*100)}% Confidence Interval'
        )
        
        # Format the plot
        ax.set_title(f'GDP Forecast with {int(confidence_level*100)}% Confidence Intervals', fontsize=15)
        ax.set_ylabel('GDP (Billions of USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show billions
        def billions(x, pos):
            return f'${x:.1f}B'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(billions))
        
        plt.tight_layout()
        return fig