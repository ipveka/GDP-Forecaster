"""
Runner script for GDP forecasting using the improved GDPForecaster.
"""

import sys
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gdp_forecaster import GDPForecaster
from utils.utils import generate_report

# Configuration - modify these variables as needed
COUNTRY_CODE = "USA"
FORECAST_HORIZON = 5
BACKTEST_YEARS = 3
RUN_ROLLING_BACKTESTS = True

# Setup output directory
PROJECT_ROOT = Path(__file__).parents[1]  # Go up one level from src
RESULTS_DIR = PROJECT_ROOT / "results" / COUNTRY_CODE
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = RESULTS_DIR / f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('gdp_forecast')

# Run Forecast
def run_forecast():
    """Run the GDP forecast with rolling backtests and save results."""
    logger.info(f"Starting GDP forecast for {COUNTRY_CODE}")
    logger.info(f"Results will be saved to {RESULTS_DIR}")
    
    # Initialize forecaster
    forecaster = GDPForecaster()
    
    # Step 1: Load historical data
    logger.info(f"Loading historical data for {COUNTRY_CODE}")
    historical_data = forecaster.load_data(COUNTRY_CODE)
    logger.info(f"Loaded {len(historical_data)} years of historical data")
    
    # Step 2: Run rolling backtests first to evaluate model performance
    backtest_metrics = None
    if RUN_ROLLING_BACKTESTS:
        logger.info(f"Running rolling backtests for the last {BACKTEST_YEARS} years")
        backtest_results = forecaster.run_rolling_backtests(n_years=BACKTEST_YEARS)
        
        if not backtest_results.empty:
            # Calculate aggregate metrics
            mape = np.mean(np.abs(backtest_results['Percent_Error']))
            rmse = np.sqrt(np.mean(backtest_results['Error'] ** 2))
            
            backtest_metrics = {
                'MAPE': mape,
                'RMSE': rmse,
                'Detailed': backtest_results
            }
            
            logger.info(f"Rolling backtest MAPE: {mape:.2f}%, RMSE: ${rmse/1e9:.2f}B")
    
    # Step 3: Train the final model with all available data for forecasting
    logger.info("Training final model with all available data for forecasting")
    metrics = forecaster.train_model(test_years=0)  # Use all data for the forecasting model
    logger.info(f"Model training complete. Metrics: {metrics}")
    
    # Step 4: Analyze feature importance
    feature_importance = forecaster.get_model_coefficients()
    top_features = feature_importance.head(5)
    logger.info("Top 5 most important features:")
    for i, (feature, importance) in enumerate(zip(
            top_features['Feature'], top_features['Normalized_Importance'])):
        logger.info(f"  {i+1}. {feature}: {importance:.2%}")
    
    # Save feature importance
    fig_importance = forecaster.plot_feature_importance()
    importance_path = RESULTS_DIR / f"{COUNTRY_CODE}_feature_importance.png"
    fig_importance.savefig(importance_path, dpi=300, bbox_inches='tight')
    feature_importance.to_csv(RESULTS_DIR / f"{COUNTRY_CODE}_feature_importance.csv")
    plt.close(fig_importance)
    logger.info(f"Saved feature importance to {importance_path}")
    
    # Step 5: Generate forecast for covariates first
    logger.info(f"Forecasting economic indicators for the next {FORECAST_HORIZON} years...")
    forecasted_features = forecaster.forecast_features(horizon=FORECAST_HORIZON)
    
    # Plot key forecasted covariates
    try:
        # Select top 5 important features
        top_features = feature_importance.head(5).reset_index(drop=True)
        key_indicators = list(top_features['Feature'])
        
        # Create a figure with subplots for each indicator
        fig, axes = plt.subplots(len(key_indicators), 1, figsize=(12, 4*len(key_indicators)))
        
        # If only one indicator, axes is not a list
        if len(key_indicators) == 1:
            axes = [axes]
        
        # Plot each indicator
        for i, feature in enumerate(key_indicators):
            # Get historical data (last 15 years only)
            historical_series = forecaster.historical_data[feature].copy()
            
            # Get last 15 years of data
            current_year = datetime.now().year
            start_year = current_year - 15
            start_date = pd.Timestamp(f"{start_year}-01-01")
            
            # Filter data to the last 15 years
            if len(historical_series) > 15:
                historical_series = historical_series[historical_series.index >= start_date]
            
            # Get forecasted data
            forecasted_series = forecasted_features[feature]
            
            # Plot historical data
            axes[i].plot(historical_series.index, historical_series.values, 'o-', 
                        color='blue', label='Historical')
            
            # Plot forecasted data
            axes[i].plot(forecasted_series.index, forecasted_series.values, 'o-', 
                        color='red', label='Forecasted')
            
            # Format plot
            readable_name = feature.replace('.', ' ').replace('_', ' ')
            axes[i].set_title(f'{readable_name}', fontsize=14)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Format x-axis
            locator = mdates.YearLocator()
            axes[i].xaxis.set_major_locator(locator)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Format y-axis appropriately based on the indicator
            if 'PERCENT' in feature.upper() or 'RATE' in feature.upper() or 'ZS' in feature.upper() or 'ZG' in feature.upper():
                axes[i].yaxis.set_major_formatter(mtick.PercentFormatter())
                axes[i].set_ylabel('Percentage (%)')
            elif 'POP' in feature.upper():
                def millions(x, pos):
                    return f'{x/1e6:.1f}M'
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(millions))
                axes[i].set_ylabel('Population (Millions)')
            else:
                axes[i].set_ylabel('Value')
        
        plt.tight_layout()
        indicators_path = RESULTS_DIR / f"{COUNTRY_CODE}_key_indicators_forecast.png"
        fig.savefig(indicators_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved key indicators plot to {indicators_path}")
    except Exception as e:
        logger.error(f"Error creating indicators plot: {str(e)}")
    
    # Step 6: Generate GDP forecast
    logger.info(f"Forecasting GDP for the next {FORECAST_HORIZON} years")
    gdp_forecast = forecaster.forecast_gdp(horizon=FORECAST_HORIZON)
    
    # Step 7: Get official forecasts
    logger.info("Getting official IMF forecasts for comparison")
    try:
        official_forecast = forecaster.get_official_forecasts()
    except Exception as e:
        logger.warning(f"Could not get official forecasts: {str(e)}")
        official_forecast = None
    
    # Step 8: Create and save GDP and growth plots using plot_forecast method
    try:
        # Use the plot_forecast method with show_history_years=10 and include_backtests=True
        fig_forecast = forecaster.plot_forecast(show_history_years=10, include_backtests=True)
        
        # Save the figure
        forecast_path = RESULTS_DIR / f"{COUNTRY_CODE}_gdp_forecast.png"
        fig_forecast.savefig(forecast_path, dpi=300, bbox_inches='tight')
        plt.close(fig_forecast)
        logger.info(f"Saved GDP and growth plots to {forecast_path}")
        
        # Create a separate GDP growth plot for the report
        # Extract just the growth rate subplot for a dedicated growth visualization
        fig_growth, ax = plt.subplots(figsize=(12, 6))
        
        # Get historical GDP data (last 10 years)
        historical_gdp = forecaster.historical_data['NY.GDP.MKTP.CD'].copy()
        if len(historical_gdp) > 10:
            historical_gdp = historical_gdp.iloc[-10:]
        
        # Calculate historical growth
        if len(historical_gdp) > 1:
            historical_growth = historical_gdp.pct_change() * 100
            ax.plot(historical_growth.index, historical_growth.values, 'o-',
                    color='blue', label='Historical Growth')
        
        # Add forecasted growth
        ax.plot(forecaster.gdp_forecast.index, forecaster.gdp_forecast['Growth_Rate'].values, 'o-',
                color='red', label='Forecasted Growth')
        
        # Add IMF growth if available
        if forecaster.official_forecast is not None and 'IMF_GDP_Growth' in forecaster.official_forecast.columns:
            ax.plot(forecaster.official_forecast.index,
                    forecaster.official_forecast['IMF_GDP_Growth'].values, 's-',
                    color='green', label='IMF Growth Forecast')
        
        # Format plot
        ax.set_title('GDP Annual Growth Rate', fontsize=16)
        ax.set_ylabel('Annual Growth Rate (%)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Format x-axis to show only years
        locator = mdates.YearLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Format y-axis to show percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        growth_path = RESULTS_DIR / f"{COUNTRY_CODE}_gdp_growth.png"
        fig_growth.savefig(growth_path, dpi=300, bbox_inches='tight')
        plt.close(fig_growth)
        logger.info(f"Saved GDP growth plot to {growth_path}")
        
    except Exception as e:
        logger.error(f"Error creating forecast plots: {str(e)}")
    
    # Step 9: Export all forecast data
    logger.info(f"Exporting all forecast data to {RESULTS_DIR}")
    forecaster.export_results(str(RESULTS_DIR))
    
    # Step 10: Generate HTML report
    # Include backtest metrics if available
    if backtest_metrics:
        metrics.update({'Backtest_MAPE': backtest_metrics['MAPE'], 
                       'Backtest_RMSE': backtest_metrics['RMSE']})
    
    report_path = generate_report(
        country_code=COUNTRY_CODE,
        results_dir=RESULTS_DIR,
        metrics=metrics,
        gdp_forecast=gdp_forecast,
        feature_importance=feature_importance,
        forecast_horizon=FORECAST_HORIZON,
        backtest_years=BACKTEST_YEARS,
        backtest_results=forecaster.backtest_results if RUN_ROLLING_BACKTESTS else None
    )
    logger.info(f"Generated comprehensive report: {report_path}")
    
    logger.info("Forecast completed successfully")
    return forecaster

# Main
if __name__ == "__main__":
    try:
        forecaster = run_forecast()
        print(f"\nForecast completed successfully! Results saved to: {RESULTS_DIR}")
    except Exception as e:
        logger.error(f"Error during forecasting: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print(f"Check the log file for more details: {log_file}")
        sys.exit(1)