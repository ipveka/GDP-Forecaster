"""
GDP Forecaster Streamlit App

This app provides an interactive interface for forecasting GDP using
the GDPForecaster module. Users can select countries, forecast horizons,
and other parameters to generate forecasts and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import warnings
import logging

# Configure warnings and logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Ensure the package is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if not any(p.endswith(os.path.abspath(".")) for p in sys.path):
    sys.path.insert(0, os.path.abspath("."))

# Import GDPForecaster (assuming it's in the utils module)
from utils.gdp_forecaster import GDPForecaster

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
    Select parameters on the left sidebar and run the forecast to view detailed results.
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
    # Add more countries as needed
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
    # These could be connected to model parameters in a more advanced implementation
    alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.1)
    l1_ratio = st.slider("L1 Ratio (0=Ridge, 1=Lasso)", 0.0, 1.0, 0.5, 0.1)

# Create a run button
run_forecast = st.sidebar.button("Run Forecast", type="primary")

# Initialize session state to store results
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
    
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None

# Function to format numbers for display
def format_number(number, is_percent=False):
    if is_percent:
        return f"{number:.2f}%"
    elif abs(number) >= 1e9:
        return f"${number/1e9:.2f}B"
    elif abs(number) >= 1e6:
        return f"${number/1e6:.2f}M"
    else:
        return f"${number:.2f}"

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
        
        # Step 3: Train the model
        st.info(f"Training model with {backtest_years} years for testing...")
        metrics = forecaster.train_model(test_years=backtest_years)
        progress_bar.progress(40)
        
        # Step 4: Run rolling backtests if requested
        if run_rolling_backtests:
            st.info(f"Running rolling backtests for the last {backtest_years} years...")
            backtest_results = forecaster.run_rolling_backtests(n_years=backtest_years)
            
            if not backtest_results.empty:
                # Calculate aggregate metrics
                mape = np.mean(np.abs(backtest_results['Percent_Error']))
                rmse = np.sqrt(np.mean(backtest_results['Error'] ** 2))
                
                metrics.update({
                    'Backtest_MAPE': mape,
                    'Backtest_RMSE': rmse
                })
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
        "GDP Forecast", 
        "Backtests", 
        "Economic Indicators",
        "Feature Importance",
        "Visualizations"
    ])
    
    # 1. SUMMARY TAB
    with tabs[0]:
        st.header("Forecast Summary")
        
        # Create two columns for summary cards
        col1, col2 = st.columns(2)
        
        # Card 1: Forecast Parameters
        with col1:
            st.subheader("Forecast Parameters")
            st.markdown(f"**Forecast Horizon:** {forecast_horizon} years")
            st.markdown(f"**Backtest Years:** {backtest_years}")
            st.markdown(f"**Country:** {country_code} - {countries.get(country_code, country_code)}")
        
        # Card 2: Backtest Performance
        with col2:
            st.subheader("Backtest Performance")
            if 'Backtest_MAPE' in results['metrics']:
                st.markdown(f"**Backtest MAPE:** {results['metrics']['Backtest_MAPE']:.2f}%")
                st.markdown(f"**Backtest RMSE:** ${results['metrics']['Backtest_RMSE']/1e9:.2f}B")
            else:
                st.markdown("Backtest metrics not available. Run with rolling backtests enabled.")
        
        # General forecast overview
        st.subheader("Forecast Overview")
        gdp_forecast = results['gdp_forecast']
        avg_growth = gdp_forecast['Growth_Rate'].mean()
        total_growth = ((gdp_forecast['GDP_Forecast'].iloc[-1] / gdp_forecast['GDP_Forecast'].iloc[0]) - 1) * 100
        first_year_gdp = gdp_forecast['GDP_Forecast'].iloc[0] / 1e9
        last_year_gdp = gdp_forecast['GDP_Forecast'].iloc[-1] / 1e9
        
        st.markdown(f"""
        - Average Annual Growth Rate: **{avg_growth:.2f}%**
        - Total Growth Over Forecast Period: **{total_growth:.2f}%**
        - GDP in {gdp_forecast.index[0].year}: **${first_year_gdp:.2f} billion**
        - GDP in {gdp_forecast.index[-1].year}: **${last_year_gdp:.2f} billion**
        """)
    
    # 2. GDP FORECAST TAB
    with tabs[1]:
        st.header("GDP Forecast Table")
        
        # Display forecast data
        forecast_data = []
        for idx, row in gdp_forecast.iterrows():
            year = idx.year
            gdp_billions = row['GDP_Forecast'] / 1e9
            growth_rate = row['Growth_Rate']
            
            # Format for display
            forecast_data.append({
                "Year": year,
                "GDP Forecast": f"${gdp_billions:.2f}B",
                "Growth Rate": f"{growth_rate:.2f}%",
            })
        
        # Convert to DataFrame for display
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True)
        
        # Add download button for CSV
        csv = gdp_forecast.to_csv().encode('utf-8')
        st.download_button(
            label="Download GDP Forecast as CSV",
            data=csv,
            file_name=f"{country_code}_gdp_forecast.csv",
            mime="text/csv",
        )
    
    # 3. BACKTESTS TAB
    with tabs[2]:
        st.header("Rolling Backtest Results")
        
        if results['backtest_results'] is not None and not results['backtest_results'].empty:
            backtest_results = results['backtest_results']
            
            # Calculate metrics
            mape = np.mean(np.abs(backtest_results['Percent_Error']))
            rmse = np.sqrt(np.mean(backtest_results['Error'] ** 2))
            
            # Display summary metrics
            st.subheader("Backtest Performance Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
            col2.metric("Root Mean Squared Error (RMSE)", f"${rmse/1e9:.2f}B")
            
            # Display detailed backtest results
            st.subheader("Detailed Backtest Results")
            backtest_data = []
            for year, row in backtest_results.iterrows():
                backtest_data.append({
                    "Year": year,
                    "Actual GDP": f"${row['Actual']/1e9:.2f}B",
                    "Predicted GDP": f"${row['Predicted']/1e9:.2f}B",
                    "Error": f"${row['Error']/1e9:.2f}B",
                    "Percent Error": f"{row['Percent_Error']:.2f}%"
                })
            
            backtest_df = pd.DataFrame(backtest_data)
            st.dataframe(backtest_df, use_container_width=True)
            
            # Create bar chart comparing actual vs predicted
            st.subheader("Actual vs Predicted GDP")
            fig, ax = plt.subplots(figsize=(10, 6))
            years = backtest_results.index.astype(int)
            
            # Plot comparison
            x = np.arange(len(years))
            width = 0.35
            ax.bar(x - width/2, backtest_results['Actual'] / 1e9, width, label='Actual GDP')
            ax.bar(x + width/2, backtest_results['Predicted'] / 1e9, width, label='Predicted GDP')
            
            # Format chart
            ax.set_xticks(x)
            ax.set_xticklabels(years)
            ax.set_ylabel('GDP (Billions USD)')
            ax.set_title('Backtesting Results: Actual vs Predicted GDP')
            ax.legend()
            
            # Add values as text on bars
            for i, v in enumerate(backtest_results['Actual'] / 1e9):
                ax.text(i - width/2, v + 0.1, f'${v:.1f}B', ha='center', fontsize=9)
            for i, v in enumerate(backtest_results['Predicted'] / 1e9):
                ax.text(i + width/2, v + 0.1, f'${v:.1f}B', ha='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create error chart
            st.subheader("Backtest Error Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(years, backtest_results['Percent_Error'], 
                  color=['red' if x > 0 else 'green' for x in backtest_results['Percent_Error']])
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('Percent Error (%)')
            ax.set_title('Backtest Percent Error by Year')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            
            # Add values as text on bars
            for i, v in enumerate(backtest_results['Percent_Error']):
                ax.text(years[i], v + (1 if v > 0 else -1), f'{v:.2f}%', ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("No backtest results available. Enable rolling backtests to see this section.")
    
    # 4. ECONOMIC INDICATORS TAB
    with tabs[3]:
        st.header("Key Economic Indicators")
        
        # Display forecasted features
        if results['forecasted_features'] is not None:
            features_df = results['forecasted_features']
            
            # Get feature importance to identify top indicators
            feature_importance = results['feature_importance']
            top_features = feature_importance.head(5)['Feature'].tolist()
            
            st.subheader("Top 5 Economic Indicators")
            st.write("These are the most influential indicators for the GDP forecast:")
            
            # Create a tab for each top indicator
            indicator_tabs = st.tabs([f.replace('.', ' ').replace('_', ' ').title() for f in top_features])
            
            for i, feature in enumerate(top_features):
                with indicator_tabs[i]:
                    # Get historical and forecasted data for this feature
                    historical_series = forecaster.historical_data[feature].copy()
                    # Limit to last 15 years
                    current_year = datetime.now().year
                    start_year = current_year - 15
                    start_date = pd.Timestamp(f"{start_year}-01-01")
                    if len(historical_series) > 15:
                        historical_series = historical_series[historical_series.index >= start_date]
                    
                    forecasted_series = features_df[feature]
                    
                    # Create line chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot historical and forecasted data
                    ax.plot(historical_series.index, historical_series.values, 'o-', 
                            color='blue', label='Historical Data')
                    ax.plot(forecasted_series.index, forecasted_series.values, 'o-', 
                            color='red', label='Forecasted Data')
                    
                    # Format chart
                    readable_name = feature.replace('.', ' ').replace('_', ' ')
                    ax.set_title(f'{readable_name} - Historical and Forecasted Values')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Format x-axis
                    locator = mdates.YearLocator()
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    
                    # Format y-axis based on indicator type
                    if 'PERCENT' in feature.upper() or 'RATE' in feature.upper() or 'ZS' in feature.upper() or 'ZG' in feature.upper():
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                        ax.set_ylabel('Percentage (%)')
                    elif 'POP' in feature.upper():
                        def millions(x, pos):
                            return f'{x/1e6:.1f}M'
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(millions))
                        ax.set_ylabel('Population (Millions)')
                    else:
                        ax.set_ylabel('Value')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show data table
                    combined_data = pd.concat([
                        historical_series.rename('Historical'),
                        forecasted_series.rename('Forecasted')
                    ], axis=1)
                    
                    # Format the data based on type
                    if 'PERCENT' in feature.upper() or 'RATE' in feature.upper() or 'ZS' in feature.upper() or 'ZG' in feature.upper():
                        combined_data = combined_data.applymap(lambda x: f"{x:.2f}%" if not pd.isna(x) else "")
                    elif 'POP' in feature.upper():
                        combined_data = combined_data.applymap(lambda x: f"{x/1e6:.2f}M" if not pd.isna(x) else "")
                    
                    st.dataframe(combined_data)
            
            # All indicators table
            st.subheader("All Forecasted Indicators")
            
            # Create a formatted version of the dataframe for display
            display_df = features_df.copy()
            
            # Format each column appropriately
            for col in display_df.columns:
                if 'PERCENT' in col.upper() or 'RATE' in col.upper() or 'ZS' in col.upper() or 'ZG' in col.upper():
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "")
                elif 'POP' in col.upper():
                    display_df[col] = display_df[col].apply(lambda x: f"{x/1e6:.2f}M" if not pd.isna(x) else "")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Add download button for CSV
            csv = features_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download All Economic Indicators as CSV",
                data=csv,
                file_name=f"{country_code}_economic_indicators.csv",
                mime="text/csv",
            )
            
        else:
            st.info("No forecasted features available.")
    
    # 5. FEATURE IMPORTANCE TAB
    with tabs[4]:
        st.header("Feature Importance Analysis")
        
        # Get feature importance data
        feature_importance = results['feature_importance']
        
        # Scale coefficient values based on magnitude
        coef_scale = 1.0
        max_coef = feature_importance['Coefficient'].abs().max()
        
        if max_coef >= 1e9:
            coef_scale = 1e9
            coef_unit = "billions"
        elif max_coef >= 1e6:
            coef_scale = 1e6
            coef_unit = "millions"
        elif max_coef >= 1e3:
            coef_scale = 1e3
            coef_unit = "thousands"
        else:
            coef_unit = ""
        
        # Display feature importance table
        st.subheader("Feature Importance Table")
        
        # Create a formatted version of the dataframe for display
        display_data = []
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            scaled_coef = row['Coefficient'] / coef_scale
            display_data.append({
                "Rank": i+1,
                "Feature": row['Feature'],
                "Importance": f"{row['Normalized_Importance']:.2%}",
                "Coefficient": f"{scaled_coef:.4f} {coef_unit}"
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True)
        
        # Create horizontal bar chart of feature importance
        st.subheader("Feature Importance Visualization")
        
        # Get top 10 features
        top_n = 10
        top_features = feature_importance.head(top_n).iloc[::-1]  # Reverse for bottom-to-top display
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        bars = ax.barh(top_features['Feature'], top_features['Normalized_Importance'], color='skyblue')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{top_features['Normalized_Importance'].iloc[i]:.1%}", 
                    va='center')
        
        ax.set_title('Top Features by Importance', fontsize=15)
        ax.set_xlabel('Normalized Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add download button for CSV
        csv = feature_importance.to_csv().encode('utf-8')
        st.download_button(
            label="Download Feature Importance as CSV",
            data=csv,
            file_name=f"{country_code}_feature_importance.csv",
            mime="text/csv",
        )
    
    # 6. VISUALIZATIONS TAB
    with tabs[5]:
        st.header("GDP Forecast Visualizations")
        
        # Plot the main forecast chart
        try:
            st.subheader("GDP Forecast (Values and Growth Rates)")
            fig = forecaster.plot_forecast(show_history_years=show_history_years, include_backtests=True)
            st.pyplot(fig)
            
            # Add download button for PNG
            plt.savefig("temp_forecast_plot.png", dpi=300, bbox_inches='tight')
            with open("temp_forecast_plot.png", "rb") as file:
                st.download_button(
                    label="Download Plot as PNG",
                    data=file,
                    file_name=f"{country_code}_gdp_forecast.png",
                    mime="image/png",
                )
            os.remove("temp_forecast_plot.png")  # Clean up
            
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            
else:
    # Display initial app instructions
    st.info("Select parameters in the sidebar and click 'Run Forecast' to generate GDP forecasts.")
    
    # Show example/demo image
    st.subheader("Example Forecast Visualization")
    example_image_path = os.path.join(script_dir, "example_forecast.png")
    if os.path.exists(example_image_path):
        st.image(example_image_path, caption="Example GDP forecast visualization")
    else:
        st.markdown("*Example visualization would be shown here*")
    
    # Brief methodology explanation
    st.subheader("Methodology")
    st.markdown("""
    This app uses machine learning to forecast GDP based on economic indicators:
    
    1. **Data Collection**: Historical economic data is retrieved for the selected country
    2. **Feature Selection**: Key economic indicators are identified
    3. **Model Training**: An ElasticNet regression model is trained on historical data
    4. **Backtesting**: The model is validated on past data to assess accuracy
    5. **Forecasting**: Future values of economic indicators are predicted
    6. **GDP Prediction**: These forecasted indicators are used to predict future GDP
    
    Select a country and parameters to get started!
    """)

# Footer
st.markdown("---")
st.markdown("GDP Forecaster App • Developed with Streamlit")