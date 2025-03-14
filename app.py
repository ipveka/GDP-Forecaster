"""
GDP Forecaster Streamlit App

This app provides an interactive interface for forecasting GDP using
the GDPForecaster module. Users can select countries, forecast horizons,
and other parameters to generate forecasts and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
import logging
from pathlib import Path

# Configure warnings and logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Ensure the package is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if not any(p.endswith(os.path.abspath(".")) for p in sys.path):
    sys.path.insert(0, os.path.abspath("."))

# Import GDPForecaster and utility functions
from utils.gdp_forecaster import GDPForecaster
from utils.app_utils import (
    format_number,
    display_summary_tab,
    display_backtesting_tab,
    display_economic_indicators_tab,
    display_feature_importance_tab,
    display_forecast_visualization_tab
)

# Set page configuration
st.set_page_config(
    page_title="GDP Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("GDP Forecaster")
st.markdown(
    """
    This application forecasts GDP for selected countries using economic indicators and machine learning.
    The model identifies relationships between various economic metrics and GDP, then projects future values
    based on trends and patterns in the data.
    
    **How to use this app:**
    1. Select a country and forecast parameters in the sidebar
    2. Click "Run Forecast" to generate predictions
    3. Explore the results in different tabs:
       - Summary: Overview of forecast and model performance
       - Backtesting: Model accuracy evaluation
       - Economic Indicators: Projections for underlying variables
       - Feature Importance: Key drivers of GDP in the model
       - Forecast Visualization: Comprehensive GDP projections
    
    **Understanding forecasts:** Economic forecasts help with planning but always involve uncertainty.
    The further into the future, the wider the range of possible outcomes. Use backtesting metrics
    to gauge reliability, and consider forecasts as informed projections rather than precise predictions.
    """
)

# Create sidebar for inputs
st.sidebar.title("Forecast Parameters")

# Country selection
countries = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "JPN": "Japan",
    "CAN": "Canada",
    "AUS": "Australia",
    "CHN": "China",
    "IND": "India",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "ESP": "Spain",
    "KOR": "South Korea",
    "RUS": "Russia",
    "ZAF": "South Africa",
}

country_code = st.sidebar.selectbox(
    "Select Country", 
    options=list(countries.keys()),
    format_func=lambda x: f"{x} - {countries[x]}"
)

# Forecast horizon
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Years)", 
    min_value=1, 
    max_value=10, 
    value=5,
    help="Number of years to forecast into the future"
)

# Backtest years
backtest_years = st.sidebar.slider(
    "Backtest Years", 
    min_value=1, 
    max_value=5, 
    value=3,
    help="Number of years to use for backtesting model performance"
)

# Run rolling backtests option
run_rolling_backtests = st.sidebar.checkbox(
    "Run Rolling Backtests", 
    value=True,
    help="Perform additional backtests for model validation"
)

# Advanced options collapsible section
advanced_params = {}
with st.sidebar.expander("Advanced Options"):
    show_history_years = st.slider(
        "Historical Years to Show", 
        min_value=5, 
        max_value=20, 
        value=10,
        help="Number of historical years to display in plots"
    )
    
    # Optional advanced model parameters
    st.subheader("Model Parameters")
    advanced_params['alpha'] = st.slider(
        "Alpha (Regularization Strength)", 
        0.01, 10.0, 1.0, 0.1,
        help="Controls regularization strength. Higher values = simpler model with fewer features."
    )
    advanced_params['l1_ratio'] = st.slider(
        "L1 Ratio (0=Ridge, 1=Lasso)", 
        0.0, 1.0, 0.5, 0.1,
        help="Balance between L1 and L2 regularization. 0 = Ridge, 1 = Lasso, between = ElasticNet."
    )

# Create a run button
run_forecast = st.sidebar.button("Run Forecast", type="primary")

# Add methodology explanation to sidebar
with st.sidebar.expander("Methodology"):
    st.markdown("""
    This forecaster uses a multi-step approach:
    
    1. **Data Collection**: Historical economic data from World Bank and other sources
    2. **Feature Selection**: Identifies important economic indicators
    3. **Model Training**: Uses ElasticNet regression (combines Ridge and Lasso)
    4. **Backtesting**: Tests model on historical data to validate accuracy
    5. **Variable Forecasting**: Projects individual economic indicators
    6. **GDP Prediction**: Uses forecasted indicators to predict GDP
    
    The model emphasizes recent data and uses regularization to reduce overfitting.
    """)

# Initialize session state to store results
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
    
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None

# Main forecasting workflow
if run_forecast:
    with st.spinner(f"Forecasting GDP for {country_code}..."):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Step 1: Initialize forecaster
        forecaster = GDPForecaster()
        progress_bar.progress(10)
        
        # Step 2: Load historical data
        st.info(f"Loading historical data for {country_code}...")
        historical_data = forecaster.load_data(country_code)
        progress_bar.progress(20)
        
        # Step 3: Run rolling backtests first to evaluate model performance
        if run_rolling_backtests:
            st.info(f"Running rolling backtests for the last {backtest_years} years to evaluate model performance...")
            backtest_results = forecaster.run_rolling_backtests(n_years=backtest_years)
            
            if not backtest_results.empty:
                # Calculate aggregate metrics
                mape = np.mean(np.abs(backtest_results['Percent_Error']))
                rmse = np.sqrt(np.mean(backtest_results['Error'] ** 2))
                
                metrics = {
                    'Backtest_MAPE': mape,
                    'Backtest_RMSE': rmse
                }
                
                st.success(f"Backtest Complete - Average MAPE: {mape:.2f}%, RMSE: ${rmse/1e9:.2f}B")
            else:
                metrics = {}
        else:
            metrics = {}
            backtest_results = None
            
        progress_bar.progress(40)
        
        # Step 4: Train the final model for forecasting
        st.info(f"Training the final model for future forecasting...")
        training_metrics = forecaster.train_model(test_years=0)
        metrics.update(training_metrics)
        progress_bar.progress(60)
        
        # Step 5: Generate forecasts for features
        st.info("Forecasting economic indicators...")
        forecasted_features = forecaster.forecast_features(horizon=forecast_horizon)
        progress_bar.progress(80)
        
        # Step 6: Generate GDP forecast
        st.info(f"Forecasting GDP for the next {forecast_horizon} years...")
        gdp_forecast = forecaster.forecast_gdp(horizon=forecast_horizon)
        
        # Try to get official forecasts
        try:
            official_forecast = forecaster.get_official_forecasts()
        except Exception as e:
            st.warning(f"Could not get official forecasts: {str(e)}")
            official_forecast = None
        
        # Store results in session state
        st.session_state.forecast_results = {
            'metrics': metrics,
            'gdp_forecast': gdp_forecast,
            'forecasted_features': forecasted_features,
            'backtest_results': forecaster.backtest_results if run_rolling_backtests else None,
            'feature_importance': forecaster.get_model_coefficients(),
            'test_predictions': forecaster.test_predictions,
            'test_actuals': forecaster.test_actuals
        }
        st.session_state.forecaster = forecaster
        
        progress_bar.progress(100)
        st.success("Forecast completed!")
        time.sleep(0.5)  # Slight pause to show success message
        progress_bar.empty()  # Remove progress bar

# Display results if available
if st.session_state.forecast_results is not None:
    results = st.session_state.forecast_results
    forecaster = st.session_state.forecaster
    
    # Create tabs for different sections
    tabs = st.tabs([
        "Summary", 
        "Backtesting", 
        "Economic Indicators",
        "Feature Importance",
        "Forecast Visualization"
    ])
    
    # 1. SUMMARY TAB
    with tabs[0]:
        display_summary_tab(forecaster, country_code, countries, forecast_horizon, 
                            backtest_years, run_rolling_backtests, results, advanced_params)
    
    # 2. BACKTESTING TAB
    with tabs[1]:
        display_backtesting_tab(forecaster, results)
    
    # 3. ECONOMIC INDICATORS TAB
    with tabs[2]:
        display_economic_indicators_tab(forecaster, results)
    
    # 4. FEATURE IMPORTANCE TAB
    with tabs[3]:
        display_feature_importance_tab(forecaster, results)
    
    # 5. FORECAST VISUALIZATION TAB
    with tabs[4]:
        display_forecast_visualization_tab(forecaster, results, show_history_years)
    
else:
    # Display initial app instructions
    st.info("Select parameters in the sidebar and click 'Run Forecast' to generate GDP forecasts.")
    
    # Show example/demo image
    st.subheader("Example Forecast Visualization")
    example_image_path = os.path.join(script_dir, "example_forecast.png")
    if os.path.exists(example_image_path):
        st.image(example_image_path, caption="Example GDP forecast visualization")
    else:
        st.markdown("""
        *Example visualization would be shown here*
        
        The visualization will include:
        - Historical GDP values and growth rates
        - Forecasted GDP values and growth rates
        - Comparison with official forecasts (when available)
        - Backtest performance indicators
        """)
    
    # Brief methodology explanation
    st.subheader("How It Works")
    
    # Create three columns for explanation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Data & Features")
        st.markdown("""
        - Historical economic data from World Bank
        - 15+ economic indicators as features
        - Automatic feature importance ranking
        - Time series preprocessing and cleaning
        """)
    
    with col2:
        st.markdown("### Modeling Approach")
        st.markdown("""
        - ElasticNet regression combines Lasso and Ridge
        - Variable-specific forecasting methods
        - Time-based cross-validation
        - Emphasis on recent economic trends
        """)
    
    with col3:
        st.markdown("### Validation & Output")
        st.markdown("""
        - Rolling backtests for reliability assessment
        - Performance metrics (MAPE, RMSE)
        - Comparison with official IMF forecasts
        - Visualizations and downloadable results
        """)

# Footer
st.markdown("---")
st.markdown("GDP Forecaster App • Developed with Streamlit")

# Add analytics disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("""
**Disclaimer**: Economic forecasts involve inherent uncertainty.
Results should be used as one input among many for decision-making.
""")

# Run the app with: streamlit run app.py