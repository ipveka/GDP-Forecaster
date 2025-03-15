"""
Utility functions for the GDP Forecaster Streamlit App

This module contains helper functions used by the main app.py
to organize code and improve maintainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import os
from datetime import datetime
import time

def format_number(number, is_percent=False):
    """
    Format numbers for display with appropriate units
    
    Args:
        number: The number to format
        is_percent: Whether the number represents a percentage
        
    Returns:
        Formatted string representation of the number
    """
    if pd.isna(number):
        return "N/A"
    if is_percent:
        return f"{number:.2f}%"
    elif abs(number) >= 1e9:
        return f"${number/1e9:.2f}B"
    elif abs(number) >= 1e6:
        return f"${number/1e6:.2f}M"
    else:
        return f"${number:.2f}"

def display_summary_tab(forecaster, country_code, countries, forecast_horizon, backtest_years, 
                        run_rolling_backtests, results, advanced_params):
    """
    Display the Summary tab with enhanced information
    
    Args:
        forecaster: The GDPForecaster instance
        country_code: ISO country code
        countries: Dictionary mapping country codes to names
        forecast_horizon: Number of years to forecast
        backtest_years: Number of years used for backtesting
        run_rolling_backtests: Whether rolling backtests were run
        results: Dictionary containing forecast results
        advanced_params: Dictionary of advanced parameters
    """
    st.header("Forecast Summary")
    
    # Add explanation of summary tab
    st.markdown("""
    This summary provides an overview of your GDP forecast and model performance. 
    Examine key parameters, model configuration, and high-level results before exploring 
    detailed information in other tabs.
    """)
    
    # Create three columns for summary cards
    col1, col2, col3 = st.columns(3)
    
    # Card 1: Forecast Parameters
    with col1:
        st.subheader("Forecast Parameters")
        st.markdown(f"**Country:** {country_code} - {countries.get(country_code, country_code)}")
        st.markdown(f"**Forecast Horizon:** {forecast_horizon} years")
        st.markdown(f"**Backtest Years:** {backtest_years}")
        st.markdown(f"**Rolling Backtests:** {'Enabled' if run_rolling_backtests else 'Disabled'}")
        
        st.markdown("""
        *These are the core parameters that define the scope of your forecast.*
        """)
    
    # Card 2: Model Hyperparameters
    with col2:
        st.subheader("Model Hyperparameters")
        
        # Get actual hyperparameters used (if available)
        if forecaster.model is not None:
            try:
                used_alpha = forecaster.model.alpha
                used_l1_ratio = getattr(forecaster.model, 'l1_ratio', "N/A")
                st.markdown(f"**Alpha:** {used_alpha:.4f}")
                if used_l1_ratio != "N/A":
                    st.markdown(f"**L1 Ratio:** {used_l1_ratio:.4f}")
                st.markdown(f"**Model Type:** {type(forecaster.model).__name__}")
            except Exception as e:
                st.markdown(f"**Alpha:** {advanced_params.get('alpha', 'N/A')}")
                st.markdown(f"**L1 Ratio:** {advanced_params.get('l1_ratio', 'N/A')}")
        else:
            st.markdown(f"**Alpha:** {advanced_params.get('alpha', 'N/A')}")
            st.markdown(f"**L1 Ratio:** {advanced_params.get('l1_ratio', 'N/A')}")
            
        st.markdown("""
        *Hyperparameters control model behavior. Alpha controls regularization strength 
        (higher = more regularization). L1 ratio balances between L1 (Lasso) and L2 (Ridge) 
        regularization (0 = Ridge, 1 = Lasso).*
        """)
    
    # Card 3: Backtest Performance
    with col3:
        st.subheader("Backtest Performance")
        if 'metrics' in results and 'Backtest_MAPE' in results['metrics']:
            st.markdown(f"**Backtest MAPE:** {results['metrics']['Backtest_MAPE']:.2f}%")
            st.markdown(f"**Backtest RMSE:** {format_number(results['metrics']['Backtest_RMSE'])}")
        elif 'MAPE' in results.get('metrics', {}):
            st.markdown(f"**Model MAPE:** {results['metrics']['MAPE']:.2f}%")
            st.markdown(f"**Model RMSE:** {format_number(results['metrics']['RMSE'])}")
            st.markdown(f"**Model R²:** {results['metrics'].get('R2', 0):.4f}")
        else:
            st.markdown("Performance metrics not available.")
            
        st.markdown("""
        *MAPE (Mean Absolute Percentage Error) shows the average % error in predictions.
        RMSE (Root Mean Squared Error) measures prediction error in absolute terms.
        Lower values indicate better performance for both metrics.*
        """)
    
    # General forecast overview
    st.subheader("Forecast Overview")
    gdp_forecast = results['gdp_forecast']
    avg_growth = gdp_forecast['Growth_Rate'].mean()
    total_growth = ((gdp_forecast['GDP_Forecast'].iloc[-1] / gdp_forecast['GDP_Forecast'].iloc[0]) - 1) * 100
    first_year_gdp = gdp_forecast['GDP_Forecast'].iloc[0] / 1e9
    last_year_gdp = gdp_forecast['GDP_Forecast'].iloc[-1] / 1e9
    
    # Display statistics in a more organized way with columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Annual Growth Rate", f"{avg_growth:.2f}%")
        st.metric("GDP in " + str(gdp_forecast.index[0].year), f"${first_year_gdp:.2f}B")
    
    with col2:
        st.metric("Total Growth Over Forecast Period", f"{total_growth:.2f}%")
        st.metric("GDP in " + str(gdp_forecast.index[-1].year), f"${last_year_gdp:.2f}B")
        
    st.markdown("""
    **How to interpret these results:**
    - **Average Annual Growth Rate**: This represents the expected year-over-year growth throughout the forecast period. Compare this to historical growth rates to judge if the forecast is conservative or aggressive.
    - **Total Growth Over Forecast Period**: This shows the cumulative growth from the first to the last year of the forecast.
    - **GDP Values**: These show the predicted economic size at the beginning and end of the forecast period.
    
    A high-quality GDP forecast typically shows growth rates consistent with historical patterns, taking into account current economic conditions and structural factors of the economy.
    """)

def display_backtesting_tab(forecaster, results):
    """
    Display the Backtesting tab with enhanced visualizations
    
    Args:
        forecaster: The GDPForecaster instance
        results: Dictionary containing forecast results
    """
    st.header("Model Backtesting")
    
    # Add explanation of backtesting
    st.markdown("""
    Backtesting evaluates model accuracy by testing how well it would have predicted past values. 
    This is crucial for understanding how reliable your forecast might be.
    
    **What to look for:**
    - **Error Consistency**: Consistent errors (all bars similar height) suggest systematic bias that could be corrected
    - **Error Direction**: Predominant over-forecasting (red bars) or under-forecasting (green bars) indicates directional bias
    - **Error Magnitude**: MAPE under 5% is excellent, 5-10% is good, over 10% suggests caution is needed
    """)
    
    if results['backtest_results'] is not None and not results['backtest_results'].empty:
        backtest_results = results['backtest_results']
        
        # Calculate metrics
        mape = np.mean(np.abs(backtest_results['Percent_Error']))
        rmse = np.sqrt(np.mean(backtest_results['Error'] ** 2))
        
        # Display summary metrics in columns
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
        col2.metric("Root Mean Squared Error (RMSE)", f"${rmse/1e9:.2f}B")
        
        st.markdown("""
        **Understanding these metrics:**
        - **MAPE**: Average percentage error regardless of direction (lower is better)
        - **RMSE**: Error in currency terms, giving more weight to large errors
        """)
        
        # Display detailed backtest results table
        st.subheader("Detailed Backtest Results")
        st.markdown("""
        This table shows the numeric results for each backtest year, allowing you to examine 
        specific years more closely.
        """)
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
        
        # Add download button for CSV
        csv = backtest_results.to_csv().encode('utf-8')
        st.download_button(
            label="Download Backtest Results as CSV",
            data=csv,
            file_name=f"{forecaster.country_code}_backtest_results.csv",
            mime="text/csv",
        )
    else:
        st.info("No backtest results available. Run with backtesting enabled to see performance metrics.")
        st.markdown("""
        Backtesting is essential for validating model performance. Without backtest results, it's 
        difficult to assess how reliable the forecast might be. Consider enabling backtesting in 
        your next run to generate these metrics.
        """)

def display_economic_indicators_tab(forecaster, results):
    """
    Display the Economic Indicators tab with all features
    
    Args:
        forecaster: The GDPForecaster instance
        results: Dictionary containing forecast results
    """
    st.header("Economic Indicators Forecast")
    
    # Add explanation of economic indicators
    st.markdown("""
    This tab shows forecasts for all economic indicators used in the GDP prediction model. 
    These forecasts drive the GDP prediction and provide context for understanding the economic 
    conditions underlying the GDP forecast.
    
    **How to interpret these charts:**
    - **Blue lines** show historical values
    - **Red lines** show forecasted values
    - **Trend changes** between historical and forecast periods may indicate model assumptions
    - **Unusual patterns** in forecasts might require further investigation
    
    Pay special attention to indicators identified as important in the Feature Importance tab.
    """)
    
    if results['forecasted_features'] is not None:
        features_df = results['forecasted_features']
        
        # Group indicators into categories for better organization
        categories = {
            "GDP Components": ["NV.IND.TOTL.ZS", "NV.SRV.TOTL.ZS", "NV.AGR.TOTL.ZS", 
                              "NE.EXP.GNFS.ZS", "NE.IMP.GNFS.ZS"],
            "Population & Social": ["SP.POP.TOTL", "SI.POV.GINI", "SL.UEM.TOTL.ZS"],
            "Government & Finance": ["GC.DOD.TOTL.GD.ZS", "BX.KLT.DINV.WD.GD.ZS"],
            "Prices & Inflation": ["FP.CPI.TOTL.ZG"],
            "Expenditure": ["GB.XPD.RSDV.GD.ZS", "SH.XPD.CHEX.GD.ZS", "SE.XPD.TOTL.GD.ZS"],
            "Energy & Resources": ["EG.USE.PCAP.KG.OE"],
            "Other": []
        }
        
        # Assign features to categories
        feature_categories = {}
        for feature in features_df.columns:
            assigned = False
            for category, feature_list in categories.items():
                if any(code in feature for code in feature_list):
                    feature_categories[feature] = category
                    assigned = True
                    break
            if not assigned:
                feature_categories[feature] = "Other"
        
        # Create tabs for each category
        unique_categories = sorted(set(feature_categories.values()))
        category_tabs = st.tabs(unique_categories)
        
        for i, category in enumerate(unique_categories):
            with category_tabs[i]:
                # Add category description
                if category == "GDP Components":
                    st.markdown("""
                    These indicators show the sectoral composition of GDP and trade flows. Changes in these components 
                    often reflect structural changes in the economy.
                    """)
                elif category == "Population & Social":
                    st.markdown("""
                    Demographic and social indicators influence long-term economic growth potential and 
                    social welfare measures.
                    """)
                elif category == "Government & Finance":
                    st.markdown("""
                    These indicators reflect fiscal position and investment flows that can significantly 
                    impact economic stability and growth.
                    """)
                elif category == "Prices & Inflation":
                    st.markdown("""
                    Inflation metrics show price stability and monetary conditions, which affect purchasing 
                    power and economic certainty.
                    """)
                elif category == "Expenditure":
                    st.markdown("""
                    These show spending patterns in key sectors that can drive innovation, human capital 
                    development, and social welfare.
                    """)
                elif category == "Energy & Resources":
                    st.markdown("""
                    Resource utilization indicators reflect economic efficiency and environmental impact.
                    """)
                
                # Get features in this category
                cat_features = [f for f, cat in feature_categories.items() if cat == category]
                
                if not cat_features:
                    st.info(f"No indicators in the {category} category.")
                    continue
                
                # Create columns to show multiple plots side by side
                cols_per_row = 2
                for j in range(0, len(cat_features), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for k, col in enumerate(cols):
                        if j + k < len(cat_features):
                            feature = cat_features[j + k]
                            with col:
                                # Get historical and forecasted data
                                historical_series = forecaster.historical_data[feature].copy()
                                # Limit to last 15 years for better visualization
                                current_year = datetime.now().year
                                start_year = current_year - 15
                                start_date = pd.Timestamp(f"{start_year}-01-01")
                                if len(historical_series) > 15:
                                    historical_series = historical_series[historical_series.index >= start_date]
                                
                                forecasted_series = features_df[feature]
                                
                                # Create line chart
                                fig, ax = plt.subplots(figsize=(10, 5))
                                
                                # Plot historical and forecasted data
                                ax.plot(historical_series.index, historical_series.values, 'o-', 
                                        color='#3498db', label='Historical')
                                ax.plot(forecasted_series.index, forecasted_series.values, 'o-', 
                                        color='#e74c3c', label='Forecast')
                                
                                # Format chart
                                readable_name = feature.replace('.', ' ').replace('_', ' ')
                                
                                # Try to map code to more readable name
                                readable_labels = {
                                    "NY.GDP.MKTP.CD": "GDP (current US$)",
                                    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
                                    "SP.POP.TOTL": "Population",
                                    "SI.POV.GINI": "Gini index",
                                    "NE.EXP.GNFS.ZS": "Exports (% of GDP)",
                                    "NE.IMP.GNFS.ZS": "Imports (% of GDP)",
                                    "BX.KLT.DINV.WD.GD.ZS": "FDI inflows (% of GDP)",
                                    "GC.DOD.TOTL.GD.ZS": "Government debt (% of GDP)",
                                    "SL.UEM.TOTL.ZS": "Unemployment (%)",
                                    "FP.CPI.TOTL.ZG": "Inflation (%)",
                                    "GB.XPD.RSDV.GD.ZS": "R&D expenditure (% of GDP)",
                                    "SH.XPD.CHEX.GD.ZS": "Health expenditure (% of GDP)",
                                    "SE.XPD.TOTL.GD.ZS": "Education expenditure (% of GDP)",
                                    "EG.USE.PCAP.KG.OE": "Energy use per capita",
                                    "NV.IND.TOTL.ZS": "Industry (% of GDP)",
                                    "NV.SRV.TOTL.ZS": "Services (% of GDP)",
                                    "NV.AGR.TOTL.ZS": "Agriculture (% of GDP)"
                                }
                                
                                if feature in readable_labels:
                                    readable_name = readable_labels[feature]
                                
                                ax.set_title(readable_name, fontsize=12)
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                # Format x-axis
                                locator = mdates.YearLocator(2)  # Show every 2 years for less crowding
                                ax.xaxis.set_major_locator(locator)
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                                plt.xticks(rotation=45)
                                
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
        
        # All indicators table
        st.subheader("All Forecasted Indicators")
        st.markdown("""
        This table provides a comprehensive view of all forecasted indicators. You can download this data
        for further analysis or reporting.
        
        **Reading the data:**
        - Values formatted with **%** are percentages (like growth rates or shares of GDP)
        - Values formatted with **M** are in millions (like population)
        - Examine trends across years to identify expected structural changes
        """)
        
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
            file_name=f"{forecaster.country_code}_economic_indicators.csv",
            mime="text/csv",
        )
    else:
        st.info("No forecasted features available.")

def display_feature_importance_tab(forecaster, results):
    """
    Display the Feature Importance tab with all features
    
    Args:
        forecaster: The GDPForecaster instance
        results: Dictionary containing forecast results
    """
    st.header("Feature Importance Analysis")
    
    # Add explanation of feature importance
    st.markdown("""
    Feature importance shows which economic indicators have the strongest influence on the GDP forecast.
    This provides insight into the economic drivers behind the forecast and the model's decision-making.
    
    **How to interpret feature importance:**
    - **Higher values** indicate stronger influence on GDP predictions
    - **Positive coefficients** mean the indicator positively correlates with GDP (increases together)
    - **Negative coefficients** mean the indicator negatively correlates with GDP (inverse relationship)
    - **Feature ranking** helps identify which indicators are most critical to monitor
    
    The model is using ElasticNet regression, which combines L1 (Lasso) and L2 (Ridge) regularization to
    select relevant features and handle multicollinearity among economic indicators.
    """)
    
    # Get feature importance data - show all features
    feature_importance = results['feature_importance']
    
    # Try to use the plot_feature_importance method from GDPForecaster
    try:
        st.subheader("Feature Importance Visualization")
        fig = forecaster.plot_feature_importance()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Using standard visualization: {str(e)}")
        
        # Standard visualization as fallback
        fig, ax = plt.subplots(figsize=(12, max(8, len(feature_importance) * 0.4)))
        
        # Plot horizontal bar chart - show all features
        feature_importance = feature_importance.iloc[::-1]  # Reverse for bottom-to-top display
        bars = ax.barh(feature_importance['Feature'], feature_importance['Abs_Coefficient'], color='#3498db')
        
        ax.set_title('Feature Importance (Coefficient Magnitude)', fontsize=15)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("""
    **Key insights from this visualization:**
    - Features at the top have the strongest influence on GDP predictions
    - The magnitude shows the relative impact of a one-unit change in the indicator
    - A balanced mix of different indicator types suggests a comprehensive model
    
    If a single indicator dominates, it may indicate the model is overreliant on one factor,
    which could reduce forecast robustness to economic shocks.
    """)
    
    # Display feature importance table - show all features
    st.subheader("Feature Importance Table")
    st.markdown("""
    This table provides the numerical values behind the visualization. The "Effect" column shows
    whether an increase in the indicator is associated with an increase in GDP (Positive) or
    a decrease in GDP (Negative).
    """)
    
    # Create a formatted version of the dataframe for display
    display_data = []
    for i, (_, row) in enumerate(feature_importance.iterrows()):
        # Try to map code to more readable name
        feature_name = row['Feature']
        readable_labels = {
            "NY.GDP.MKTP.CD": "GDP (current US$)",
            "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
            "SP.POP.TOTL": "Population",
            "SI.POV.GINI": "Gini index",
            "NE.EXP.GNFS.ZS": "Exports (% of GDP)",
            "NE.IMP.GNFS.ZS": "Imports (% of GDP)",
            "BX.KLT.DINV.WD.GD.ZS": "FDI inflows (% of GDP)",
            "GC.DOD.TOTL.GD.ZS": "Government debt (% of GDP)",
            "SL.UEM.TOTL.ZS": "Unemployment (%)",
            "FP.CPI.TOTL.ZG": "Inflation (%)",
            "GB.XPD.RSDV.GD.ZS": "R&D expenditure (% of GDP)",
            "SH.XPD.CHEX.GD.ZS": "Health expenditure (% of GDP)",
            "SE.XPD.TOTL.GD.ZS": "Education expenditure (% of GDP)",
            "EG.USE.PCAP.KG.OE": "Energy use per capita",
            "NV.IND.TOTL.ZS": "Industry (% of GDP)",
            "NV.SRV.TOTL.ZS": "Services (% of GDP)",
            "NV.AGR.TOTL.ZS": "Agriculture (% of GDP)"
        }
        
        readable_name = readable_labels.get(feature_name, feature_name)
        
        display_data.append({
            "Rank": i+1,
            "Feature": readable_name,
            "Importance": f"{row['Normalized_Importance']:.2%}",
            "Coefficient": f"{row['Coefficient']:.4f}",
            "Effect": "Positive" if row['Coefficient'] > 0 else "Negative"
        })
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True)
    
    # Add download button for CSV
    csv = feature_importance.to_csv().encode('utf-8')
    st.download_button(
        label="Download Feature Importance as CSV",
        data=csv,
        file_name=f"{forecaster.country_code}_feature_importance.csv",
        mime="text/csv",
    )

def display_forecast_visualization_tab(forecaster, results, show_history_years):
    """
    Display the GDP Forecast Visualization tab
    
    Args:
        forecaster: The GDPForecaster instance
        results: Dictionary containing forecast results
        show_history_years: Number of historical years to show
    """
    st.header("GDP Forecast Visualizations")
    
    # Add explanation of forecast visualization
    st.markdown("""
    This visualization shows the complete GDP forecast alongside historical data, giving
    you the full picture of projected economic performance.
    
    **How to interpret this chart:**
    - **Upper Chart**: Shows GDP values in absolute terms (billions of USD)
      - Blue line/points: Historical GDP
      - Red line/points: Forecasted GDP
      - Green line/points (if available): IMF forecasts for comparison
    
    - **Lower Chart**: Shows year-over-year growth rates (%)
      - Blue line: Historical growth rates
      - Red line: Forecasted growth rates
      - Green line (if available): IMF growth forecasts
    
    **What to look for:**
    - **Trend changes**: Major shifts between historical and forecasted trends
    - **Growth volatility**: How stable or volatile the forecast growth rates are
    - **Comparison to IMF**: How your model compares to official forecasts
    - **Long-term trajectory**: The overall direction of economic performance
    """)
    
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
                file_name=f"{forecaster.country_code}_gdp_forecast.png",
                mime="image/png",
            )
        os.remove("temp_forecast_plot.png")  # Clean up
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        
    # Add GDP forecast table
    st.subheader("GDP Forecast Data")
    st.markdown("""
    This table provides the numerical values behind the visualization. It shows the forecasted
    GDP values and growth rates for each year in the forecast period.
    """)
    
    gdp_forecast = results['gdp_forecast']
    
    # Create a formatted version for display
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
        file_name=f"{forecaster.country_code}_gdp_forecast.csv",
        mime="text/csv",
    )
    
    # Add interpretation hints
    st.markdown("""
    **Key takeaways from this forecast:**
    
    1. **Compare with historical performance**: Is the forecast significantly different from recent historical trends?
       If so, what economic factors might explain the change?
       
    2. **Growth rate stability**: Are growth rates stable or volatile? Stable growth suggests structural factors,
       while volatile growth might indicate cyclical factors or potential forecasting issues.
       
    3. **Comparison with official forecasts**: How does your forecast compare with IMF or other official forecasts?
       Major divergences should be investigated and explained.
       
    4. **Range of uncertainty**: Remember that all forecasts have uncertainty. The further into the future,
       the wider the potential range of outcomes.
    """)