"""
Utility functions for the GDP forecaster project.
"""

import os
from datetime import datetime
from pathlib import Path


def generate_report(country_code, results_dir, metrics, gdp_forecast, feature_importance, 
                   forecast_horizon, backtest_years, comparison_plots=None):
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
        comparison_plots: List of tuples (title, image_path) for country comparisons
    
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
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; margin-top: 25px; }}
                h4 {{ color: #16a085; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; margin-right: 15px; }}
                .image-container {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
                .footer {{ margin-top: 50px; font-size: 12px; color: #7f8c8d; text-align: center; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>GDP Forecast Report for {country_code}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>Forecast Summary</h2>
            <p>
                <span class="metric">Forecast Horizon:</span> {forecast_horizon} years<br>
                <span class="metric">Backtest Years:</span> {backtest_years}<br>
                <span class="metric">Model Accuracy (MAPE):</span> {metrics['MAPE']:.2f}%<br>
                <span class="metric">R-squared:</span> {metrics['R2']:.4f}<br>
                <span class="metric">RMSE:</span> ${metrics['RMSE']/1e9:.2f} billion
            </p>
            
            <h2>GDP Forecast Table</h2>
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
        
        # Feature importance section
        f.write(f"""
            <h2>Feature Importance</h2>
            <p>Top factors influencing GDP predictions:</p>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance</th>
                    <th>Coefficient</th>
                </tr>
        """)
        
        # Add top 10 features
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            f.write(f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{row['Feature']}</td>
                    <td>{row['Normalized_Importance']:.2%}</td>
                    <td>{row['Coefficient']:.4f}</td>
                </tr>
            """)
        
        f.write("""</table>""")
        
        # Add plots
        f.write(f"""
            <h2>Forecast Visualizations</h2>
            
            <h3>GDP Forecast</h3>
            <div class="image-container">
                <img src="{country_code}_gdp_forecast.png" alt="GDP Forecast">
            </div>
            
            <h3>Confidence Intervals</h3>
            <div class="image-container">
                <img src="{country_code}_confidence_intervals.png" alt="Confidence Intervals">
            </div>
            
            <h3>Feature Importance</h3>
            <div class="image-container">
                <img src="{country_code}_feature_importance.png" alt="Feature Importance">
            </div>
        """)
        
        # Add country comparison if available
        if comparison_plots:
            f.write(f"""<h3>Country Comparisons</h3>""")
            for title, image_path in comparison_plots:
                f.write(f"""
                <h4>{title}</h4>
                <div class="image-container">
                    <img src="{image_path}" alt="{title}">
                </div>
                """)
        
        # Add model interpretation
        f.write(f"""
            <h2>Model Interpretation</h2>
            <p>
                This forecast was generated using an ElasticNet regression model, which combines L1 and L2 regularization
                to handle complex relationships between economic indicators. The model was trained on historical data and
                validated through backtesting on {backtest_years} years of data.
            </p>
            <p>
                With a Mean Absolute Percentage Error (MAPE) of {metrics['MAPE']:.2f}%, the model demonstrates 
                {'strong' if metrics['MAPE'] < 5 else 'moderate' if metrics['MAPE'] < 10 else 'acceptable'} 
                predictive accuracy. The R-squared value of {metrics['R2']:.4f} indicates that the model explains
                {metrics['R2']*100:.1f}% of the variance in GDP values.
            </p>
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