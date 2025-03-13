"""
Utility functions for the GDP forecaster project.
"""
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_report(country_code, results_dir, metrics, gdp_forecast, feature_importance,
                   forecast_horizon, backtest_years, backtest_results=None):
    """
    Generate a comprehensive HTML report with all GDP forecast results.
    
    Args:
        country_code: ISO 3-letter country code
        results_dir: Directory where results are saved
        metrics: Dictionary with model performance metrics
        gdp_forecast: DataFrame with GDP forecast data
        feature_importance: DataFrame with feature importance data
        forecast_horizon: Number of years forecasted
        backtest_years: Number of years used for backtesting
        backtest_results: DataFrame with rolling backtest results
        
    Returns:
        Path to the generated report
    """
    report_path = Path(results_dir) / f"{country_code}_forecast_report.html"
    
    with open(report_path, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>GDP Forecast Report for {country_code}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; font-size: 28px; }}
        h2 {{ color: #3498db; margin-top: 30px; font-size: 24px; }}
        h3 {{ color: #2980b9; margin-top: 25px; font-size: 20px; }}
        h4 {{ color: #16a085; margin-top: 20px; font-size: 18px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; margin-right: 15px; }}
        .image-container {{ margin: 20px 0; text-align: center; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
        .footer {{ margin-top: 50px; font-size: 12px; color: #7f8c8d; text-align: center; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .card-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; margin: 20px 0; }}
        .card {{ background-color: #f8f9fa; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
               padding: 16px; margin-bottom: 16px; width: calc(50% - 16px); box-sizing: border-box; }}
        .card h3 {{ color: #2c3e50; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
        .card p {{ margin: 8px 0; }}
        .card-label {{ font-weight: bold; color: #7f8c8d; }}
        .card-value {{ font-size: 20px; color: #2c3e50; }}
        .plot {{ width: 100%; margin: 20px 0; }}
        .table-of-contents {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .table-of-contents h3 {{ margin-top: 0; }}
        .table-of-contents ul {{ list-style-type: none; padding-left: 10px; }}
        .table-of-contents li {{ margin: 8px 0; }}
        .table-of-contents a {{ text-decoration: none; color: #3498db; }}
        .table-of-contents a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>GDP Forecast Report for {country_code}</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <!-- Table of Contents -->
    <div class="table-of-contents">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#forecast-summary">1. Forecast Summary</a></li>
            <li><a href="#forecast-table">2. GDP Forecast Table</a></li>
""")

        # Add backtest results to TOC if available
        if backtest_results is not None and not backtest_results.empty:
            f.write("""
            <li><a href="#backtest-results">3. Rolling Backtest Results</a></li>
""")
            section_num = 4
        else:
            section_num = 3

        # Continue TOC
        f.write(f"""
            <li><a href="#covariates-analysis">{section_num}. Covariates Analysis</a></li>
            <li><a href="#key-indicators-viz">{section_num+1}. Key Indicators Visualization</a></li>
            <li><a href="#feature-importance">{section_num+2}. Feature Importance</a></li>
            <li><a href="#gdp-forecast-viz">{section_num+3}. GDP Forecast Visualization</a></li>
            <li><a href="#gdp-growth-viz">{section_num+4}. GDP Growth Visualization</a></li>
            <li><a href="#model-interpretation">{section_num+5}. Model Interpretation</a></li>
        </ul>
    </div>

    <h2 id="forecast-summary">1. Forecast Summary</h2>
    <div class="card-container">
""")

        # Create cards for the summary metrics
        f.write(f"""
        <div class="card">
            <h3>Forecast Parameters</h3>
            <p><span class="card-label">Forecast Horizon:</span> <span class="card-value">{forecast_horizon} years</span></p>
            <p><span class="card-label">Backtest Years:</span> <span class="card-value">{backtest_years}</span></p>
        </div>
""")

        # Only show backtest metrics if available
        if 'Backtest_MAPE' in metrics:
            f.write(f"""
        <div class="card">
            <h3>Backtest Performance</h3>
            <p><span class="card-label">Backtest MAPE:</span> <span class="card-value">{metrics['Backtest_MAPE']:.2f}%</span></p>
            <p><span class="card-label">Backtest RMSE:</span> <span class="card-value">${metrics['Backtest_RMSE']/1e9:.2f}B</span></p>
        </div>
""")

        f.write("""
    </div>
    
    <h2 id="forecast-table">2. GDP Forecast Table</h2>
    <table>
        <tr>
            <th>Year</th>
            <th>GDP Forecast (Billions USD)</th>
            <th>Growth Rate (%)</th>
        </tr>
""")

        # Add forecast rows
        for idx, row in gdp_forecast.iterrows():
            year = idx.year
            gdp_billions = row['GDP_Forecast'] / 1e9
            growth_rate = row['Growth_Rate']
            
            # Determine CSS class for growth rate
            growth_class = "positive" if growth_rate > 0 else "negative" if growth_rate < 0 else ""
            
            f.write(f"""
        <tr>
            <td>{year}</td>
            <td>${gdp_billions:.2f}</td>
            <td class="{growth_class}">{growth_rate:.2f}%</td>
        </tr>
""")
        
        f.write("""</table>""")
        
        # Add rolling backtest results if available
        section_counter = 3
        if backtest_results is not None and not backtest_results.empty:
            f.write(f"""
    <h2 id="backtest-results">{section_counter}. Rolling Backtest Results</h2>
    <p>Results from forecasting GDP one year at a time using only data available at that point:</p>
    <table>
        <tr>
            <th>Year</th>
            <th>Actual GDP (Billions USD)</th>
            <th>Predicted GDP (Billions USD)</th>
            <th>Error (Billions USD)</th>
            <th>Percent Error</th>
        </tr>
""")
            
            # Add backtest rows
            for year, row in backtest_results.iterrows():
                error_class = "negative" if abs(row['Percent_Error']) > 5 else ""
                
                f.write(f"""
        <tr>
            <td>{year}</td>
            <td>${row['Actual']/1e9:.2f}</td>
            <td>${row['Predicted']/1e9:.2f}</td>
            <td>${row['Error']/1e9:.2f}</td>
            <td class="{error_class}">{row['Percent_Error']:.2f}%</td>
        </tr>
""")
            
            f.write("""</table>""")
            section_counter += 1
        
        # Add forecasted covariates section (always include this now as specifically requested)
        f.write(f"""
    <h2 id="covariates-analysis">{section_counter}. Covariates Analysis</h2>
    <p>Key economic indicators used in the GDP forecast model:</p>
""")
        
        section_counter += 1
        
        # Get covariates from feature_importance
        # First check if we have covariates in the gdp_forecast DataFrame
        has_covariates_in_forecast = (gdp_forecast is not None and 
                                     hasattr(gdp_forecast, 'columns') and 
                                     len([c for c in gdp_forecast.columns if c not in ['GDP_Forecast', 'Previous_GDP', 'Growth_Rate']]) > 0)
        
        if has_covariates_in_forecast:
            # Get covariates from gdp_forecast
            covariates = [col for col in gdp_forecast.columns if col not in ['GDP_Forecast', 'Previous_GDP', 'Growth_Rate']]
            if covariates:
                f.write(f"""
    <h3>Forecasted Covariates by Year</h3>
    <p>Values of key economic indicators used for each forecasted year:</p>
    <table>
        <tr>
            <th>Year</th>
""")
                
                # Add covariate headers
                for covariate in covariates:
                    display_name = covariate.replace('_', ' ').title()
                    f.write(f"""            <th>{display_name}</th>\n""")
                
                f.write("""        </tr>""")
                
                # Add covariate rows for each year
                for idx, row in gdp_forecast.iterrows():
                    year = idx.year
                    f.write(f"""
        <tr>
            <td>{year}</td>
""")
                    # Add each covariate value
                    for covariate in covariates:
                        value = row[covariate]
                        # Format based on likely type
                        if 'PERCENT' in covariate.upper() or 'RATE' in covariate.upper() or 'ZS' in covariate.upper() or 'ZG' in covariate.upper():
                            f.write(f"""            <td>{value:.2f}%</td>\n""")
                        elif 'POP' in covariate.upper():
                            f.write(f"""            <td>{value/1e6:.2f}M</td>\n""")
                        else:
                            f.write(f"""            <td>{value:.2f}</td>\n""")
                    
                    f.write("""        </tr>""")
                
                f.write("""
    </table>
""")
        
        # If no covariates in forecast, explain
        else:
            # Get top features from feature_importance
            top_features = feature_importance.head(5)['Feature'].tolist()
            
            f.write(f"""
    <p>The model uses the following top 5 economic indicators for forecasting:</p>
    <ul>
""")
            for feature in top_features:
                display_name = feature.replace('_', ' ').title()
                f.write(f"""        <li>{display_name}</li>\n""")
            
            f.write("""
    </ul>
    <p>These indicators are forecasted separately and then used as inputs to the GDP forecast model.</p>
""")

        # Add Key Indicators Visualization section
        f.write(f"""
    <h2 id="key-indicators-viz">{section_counter}. Key Indicators Visualization</h2>
    <p>Historical (last 15 years) and forecasted trends of the top 5 economic indicators used in the model:</p>
    <div class="image-container">
        <img src="{country_code}_key_indicators_forecast.png" alt="Key Economic Indicators" class="plot">
    </div>
""")
        
        section_counter += 1

        # Feature importance section with coefficients in millions or billions
        f.write(f"""
    <h2 id="feature-importance">{section_counter}. Feature Importance</h2>
    <p>Top factors influencing GDP predictions:</p>
    <table>
        <tr>
            <th>Rank</th>
            <th>Feature</th>
            <th>Importance</th>
            <th>Coefficient</th>
        </tr>
""")

        section_counter += 1

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

        # Add top 10 features
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            scaled_coef = row['Coefficient'] / coef_scale
            f.write(f"""
        <tr>
            <td>{i+1}</td>
            <td>{row['Feature']}</td>
            <td>{row['Normalized_Importance']:.2%}</td>
            <td>{scaled_coef:.4f} {coef_unit}</td>
        </tr>
""")
        
        f.write(f"""
    </table>
    <div class="image-container">
        <img src="{country_code}_feature_importance.png" alt="Feature Importance" class="plot">
    </div>
""")
        
        # Add GDP Forecast visualization
        f.write(f"""
    <h2 id="gdp-forecast-viz">{section_counter}. GDP Forecast Visualization</h2>
    <p>Shows Actual GDP, Backtested GDP Forecast, IMF Forecast, and Model Forecast</p>
    <div class="image-container">
        <img src="{country_code}_gdp_forecast.png" alt="GDP Forecast" class="plot">
    </div>
""")
        section_counter += 1

        # Add model interpretation
        backtest_quality = "strong" if metrics.get('Backtest_MAPE', 100) < 5 else "moderate" if metrics.get('Backtest_MAPE', 100) < 10 else "acceptable"
        
        f.write(f"""
    <h2 id="model-interpretation">{section_counter}. Model Interpretation</h2>
    <p>
        This forecast was generated using an ElasticNet regression model, which combines L1 and L2 regularization
        to handle complex relationships between economic indicators. The model was trained on historical data and
        validated through rolling backtesting on {backtest_years} years of data.
    </p>
""")

        if 'Backtest_MAPE' in metrics:
            f.write(f"""
    <p>
        The rolling backtest MAPE of {metrics['Backtest_MAPE']:.2f}% shows {backtest_quality} predictive performance
        when forecasting with only historical data available at each point in time.
    </p>
""")

        f.write(f"""
    <p>
        The forecast shows {'an increasing' if gdp_forecast['Growth_Rate'].mean() > 0 else 'a decreasing'} trend
        for {country_code}'s GDP over the next {forecast_horizon} years, with an average annual growth rate of
        {gdp_forecast['Growth_Rate'].mean():.2f}%.
    </p>
""")

        # Close HTML
        f.write(f"""
    <div class="footer">
        <p>Generated using GDP Forecaster &copy; {datetime.now().year}</p>
    </div>
</body>
</html>
""")

    return report_path

def export_forecast_data(forecaster, output_dir='./output'):
    """
    Export forecast data to CSV files.
    
    Args:
        forecaster: GDPForecaster instance
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with paths to exported files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    export_paths = {}
    
    # Export GDP forecast
    if forecaster.gdp_forecast is not None:
        forecast_path = os.path.join(output_dir, f"{forecaster.country_code}_gdp_forecast.csv")
        forecaster.gdp_forecast.to_csv(forecast_path)
        export_paths['gdp_forecast'] = forecast_path
    
    # Export feature importance
    if forecaster.model is not None:
        feature_importance = forecaster.get_model_coefficients()
        importance_path = os.path.join(output_dir, f"{forecaster.country_code}_feature_importance.csv")
        feature_importance.to_csv(importance_path)
        export_paths['feature_importance'] = importance_path
    
    # Export backtest results
    if forecaster.backtest_results is not None:
        backtest_path = os.path.join(output_dir, f"{forecaster.country_code}_backtest_results.csv")
        forecaster.backtest_results.to_csv(backtest_path)
        export_paths['backtest_results'] = backtest_path
    
    return export_paths