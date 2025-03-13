#!/usr/bin/env python
"""
Runner script for GDP forecasting.
"""

import sys
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gdp_forecaster import GDPForecaster
from utils.utils import generate_report

# Configuration - modify these variables as needed
COUNTRY_CODE = "USA"          # Change to analyze different countries
FORECAST_HORIZON = 5          # Number of years to forecast
BACKTEST_YEARS = 3            # Number of years for backtesting
COMPARISON_COUNTRIES = ["DEU", "FRA", "GBR"]  # Leave empty list [] for no comparison

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


def run_forecast():
    """Run the GDP forecast and save results."""
    logger.info(f"Starting GDP forecast for {COUNTRY_CODE}")
    logger.info(f"Results will be saved to {RESULTS_DIR}")
    
    # Initialize forecaster
    forecaster = GDPForecaster()
    
    # Step 1: Load historical data
    logger.info(f"Loading historical data for {COUNTRY_CODE}")
    historical_data = forecaster.load_data(COUNTRY_CODE)
    logger.info(f"Loaded {len(historical_data)} years of historical data")
    
    # Step 2: Train the model
    logger.info(f"Training model with {BACKTEST_YEARS} years of backtesting")
    metrics = forecaster.train_model(test_years=BACKTEST_YEARS)
    logger.info(f"Model training complete. Backtest metrics: {metrics}")
    
    # Step 3: Analyze feature importance
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
    
    # Step 4: Generate forecast
    logger.info(f"Forecasting GDP for the next {FORECAST_HORIZON} years")
    gdp_forecast = forecaster.forecast_gdp(horizon=FORECAST_HORIZON)
    
    # Step 5: Get official forecasts
    logger.info("Getting official IMF forecasts for comparison")
    official_forecast = forecaster.get_official_forecasts()
    
    # Step 6: Create and save main forecast plot
    fig_forecast = forecaster.plot_forecast()
    forecast_path = RESULTS_DIR / f"{COUNTRY_CODE}_gdp_forecast.png"
    fig_forecast.savefig(forecast_path, dpi=300, bbox_inches='tight')
    plt.close(fig_forecast)
    logger.info(f"Saved forecast plot to {forecast_path}")
    
    # Step 7: Create and save confidence intervals
    fig_confidence = forecaster.plot_confidence_intervals(confidence_level=0.90)
    confidence_path = RESULTS_DIR / f"{COUNTRY_CODE}_confidence_intervals.png"
    fig_confidence.savefig(confidence_path, dpi=300, bbox_inches='tight')
    plt.close(fig_confidence)
    logger.info(f"Saved confidence intervals plot to {confidence_path}")
    
    # Step 8: Country comparison (if requested)
    comparison_plots = []
    if COMPARISON_COUNTRIES:
        all_countries = [COUNTRY_CODE] + COMPARISON_COUNTRIES
        logger.info(f"Comparing GDP across countries: {', '.join(all_countries)}")
        
        try:
            # Growth rate comparison
            fig_growth = GDPForecaster.compare_countries(all_countries, metric='growth')
            growth_path = RESULTS_DIR / "country_growth_comparison.png"
            fig_growth.savefig(growth_path, dpi=300, bbox_inches='tight')
            plt.close(fig_growth)
            comparison_plots.append(("Growth Rate Comparison", growth_path.name))
            
            # Normalized GDP comparison
            fig_gdp = GDPForecaster.compare_countries(all_countries, metric='gdp')
            gdp_path = RESULTS_DIR / "country_gdp_comparison.png"
            fig_gdp.savefig(gdp_path, dpi=300, bbox_inches='tight')
            plt.close(fig_gdp)
            comparison_plots.append(("Normalized GDP Comparison", gdp_path.name))
            
            logger.info(f"Saved country comparison plots to {RESULTS_DIR}")
        except Exception as e:
            logger.error(f"Error in country comparison: {str(e)}")
    
    # Step 9: Export all forecast data
    logger.info(f"Exporting all forecast data to {RESULTS_DIR}")
    forecaster.export_results(str(RESULTS_DIR))
    
    # Step 10: Generate HTML report
    report_path = generate_report(
        country_code=COUNTRY_CODE,
        results_dir=RESULTS_DIR,
        metrics=metrics,
        gdp_forecast=gdp_forecast,
        feature_importance=feature_importance,
        forecast_horizon=FORECAST_HORIZON,
        backtest_years=BACKTEST_YEARS,
        comparison_plots=comparison_plots
    )
    logger.info(f"Generated comprehensive report: {report_path}")
    
    logger.info("Forecast completed successfully")
    return forecaster


if __name__ == "__main__":
    try:
        forecaster = run_forecast()
        print(f"\nForecast completed successfully! Results saved to: {RESULTS_DIR}")
    except Exception as e:
        logger.error(f"Error during forecasting: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print(f"Check the log file for more details: {log_file}")
        sys.exit(1)