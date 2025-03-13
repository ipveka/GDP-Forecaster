# GDP Forecaster

A comprehensive toolkit for forecasting GDP of countries worldwide using ElasticNet regression with multiple economic indicators.

## Project Overview

The GDP Forecaster is a Python-based forecasting system that leverages historical economic data to predict future GDP values for any country. It incorporates multiple covariables such as population, Gini index, inflation rates, and other macroeconomic indicators to improve forecast accuracy.

### Key Features

- Forecasts GDP for any country for the next 5 years
- Incorporates multiple economic indicators as covariables
- Uses ElasticNet regression with hyperparameter tuning
- Includes backtesting capabilities to validate model performance
- Compares forecasts with official projections from sources like IMF
- Forecasts covariables using appropriate time series methods
- Visualizes forecasts with interactive plots
- Performs sensitivity analysis for key variables
- Provides confidence intervals for forecast uncertainty
- Enables cross-country and regional comparisons
- Streamlit app to run the project

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gdp-forecaster.git
   cd gdp-forecaster
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data output logs
   ```

## Quick Start

To get started with the GDP Forecaster:

```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/yourusername/gdp-forecaster.git
cd gdp-forecaster
pip install -r requirements.txt

# 2. Create necessary directories
mkdir -p data results logs

# 3. Run the forecaster for the United States
python src/runner.py
# Results will be saved to results/USA/

# 4. To analyze a different country, edit src/runner.py
# Change COUNTRY_CODE = "USA" to any other country code
# For example, COUNTRY_CODE = "JPN" for Japan
```

For an interactive analysis, open the Jupyter notebook:
```bash
jupyter notebook notebooks/gdp_forecast_example.ipynb
```

## Data Sources

The forecaster pulls data from several authoritative sources:

1. **World Bank Open Data API**
   - GDP (current US$)
   - GDP growth (annual %)
   - Population data
   - Gini index
   - Various economic and development indicators

2. **International Monetary Fund (IMF)**
   - World Economic Outlook Database
   - Official GDP growth forecasts

3. **Organization for Economic Co-operation and Development (OECD)**
   - Economic indicators for OECD member countries

4. **United Nations Statistics Division**
   - Demographic and economic data

5. **Federal Reserve Economic Data (FRED)**
   - Additional economic time series

## Methodology

### Data Collection
- Data is collected from APIs and cached locally for faster access
- Historical data for each indicator is cleaned and preprocessed

### Feature Selection
- Population trends (using historical growth rates)
- Income inequality (Gini index)
- Trade balance (exports/imports as % of GDP)
- Foreign direct investment
- Government debt
- Unemployment rates
- Inflation rates
- Sectoral composition (agriculture, industry, services)
- Research and development expenditure
- Education and healthcare spending
- Energy consumption

### Forecasting Approach
1. **Covariable Forecasting**
   - Population: Growth rate method
   - Economic indicators: ARIMA models
   - Structural indicators: Exponential smoothing
   
2. **GDP Forecasting**
   - Model: ElasticNet regression (combines L1 and L2 regularization)
   - Hyperparameter tuning: Grid search with time series cross-validation
   - Evaluation metrics: MAPE, RMSE, R-squared

3. **Backtesting**
   - Historical validation on the past 3 years
   - Comparison with official forecasts

## Project Structure

- `data/`: Cached data from various sources
- `notebooks/`: Jupyter notebooks with example analyses
  - `data_exploration.ipynb`: Interactive notebook for economic analysis
  - `forecaster.ipynb`: Interactive notebook for forecasting GDP
- `src/`: Source code including runnable scripts
  - `runner.py`: Main script for running GDP forecasts with customizable settings
- `utils/`: Utility modules for data collection and forecasting
  - `data_collector.py`: API interfaces to collect economic data
  - `variable_forecaster.py`: Time series forecasting for individual variables
  - `gdp_forecaster.py`: Main forecasting model using ElasticNet regression
  - `utils.py`: Utility functions including report generation
- `results/`: Generated forecasts, visualizations, and reports (organized by country)
- `logs/`: Log files from forecasting runs

## Running the Forecaster

There are two main ways to use the GDP Forecaster:

### 1. Using the Runner Script

Simply modify the configuration variables at the top of `src/runner.py` and run:

```bash
python src/runner.py
```

The script will:
1. Download and cache economic data for the specified country
2. Train an ElasticNet regression model with backtest validation
3. Generate GDP forecasts for the specified horizon
4. Create visualizations for the forecast, confidence intervals, and feature importance
5. Compare with other countries if specified
6. Generate a comprehensive HTML report with all results
7. Save all outputs to the `results/[COUNTRY_CODE]` directory

### 2. Using the Jupyter Notebook

For a more interactive experience, open `notebooks/gdp_forecast_example.ipynb` in Jupyter:

```bash
jupyter notebook notebooks/gdp_forecast_example.ipynb
```

The notebook allows you to:
1. Explore available countries and their data
2. Visualize historical GDP and economic indicators
3. Step through the forecasting process with detailed explanations
4. Customize the forecast parameters
5. Examine model performance and feature importance
6. Generate the same comprehensive outputs as the runner script

## Detailed Usage

### Command Line Arguments

The example.py script supports the following arguments:

```
usage: example.py [-h] [--country COUNTRY] [--horizon HORIZON]
                 [--backtest-years BACKTEST_YEARS] [--list-countries]
                 [--save-plots] [--output-dir OUTPUT_DIR]
                 [--compare-with COMPARE_WITH [COMPARE_WITH ...]]
                 [--sensitivity-analysis SENSITIVITY_ANALYSIS]
                 [--confidence-interval CONFIDENCE_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --country COUNTRY     ISO 3-letter country code (default: USA)
  --horizon HORIZON     Forecast horizon in years (default: 5)
  --backtest-years BACKTEST_YEARS
                        Number of years to use for backtesting (default: 3)
  --list-countries      List available countries and exit
  --save-plots          Save plots to files
  --output-dir OUTPUT_DIR
                        Directory to save outputs (default: ./output)
  --compare-with COMPARE_WITH [COMPARE_WITH ...]
                        Compare with other countries (e.g., --compare-with DEU FRA GBR)
  --sensitivity-analysis SENSITIVITY_ANALYSIS
                        Perform sensitivity analysis on a specific feature
  --confidence-interval CONFIDENCE_INTERVAL
                        Show forecasts with confidence intervals (0.0-1.0, default: 0.90)
```

### Programmatic Usage

```python
from utils.gdp_forecaster import GDPForecaster

# Initialize the forecaster
forecaster = GDPForecaster()

# Load data for a specific country (using ISO 3-letter code)
forecaster.load_data("USA")

# Train the model with 3 years of backtesting
metrics = forecaster.train_model(test_years=3)
print(f"Backtest Results: MAPE = {metrics['MAPE']:.2f}%, R² = {metrics['R2']:.4f}")

# Forecast GDP for the next 5 years
gdp_forecast = forecaster.forecast_gdp(horizon=5)

# Get official forecasts for comparison
official_forecast = forecaster.get_official_forecasts()

# Generate visualization
fig = forecaster.plot_forecast()
fig.savefig("usa_gdp_forecast.png")

# Print detailed forecast with comparisons
forecaster.print_forecast_summary()

# Analyze feature importance
importance = forecaster.get_model_coefficients()
print("Top 3 most important factors:")
for i in range(3):
    print(f"{i+1}. {importance['Feature'].iloc[i]}: {importance['Normalized_Importance'].iloc[i]:.2%}")

# Perform sensitivity analysis on population
sensitivity_fig = forecaster.perform_sensitivity_analysis('SP.POP.TOTL')
sensitivity_fig.savefig("population_sensitivity.png")

# Export all results
forecaster.export_results("./output")
```

## Advanced Features

### Sensitivity Analysis

The forecaster can analyze how changes in key variables affect GDP forecasts:

```python
# Analyze impact of 10% change in population
forecaster.perform_sensitivity_analysis('SP.POP.TOTL', variation_pct=10.0)

# Analyze impact of 5% change in Gini index
forecaster.perform_sensitivity_analysis('SI.POV.GINI', variation_pct=5.0)
```

### Country Comparison

Compare GDP growth across multiple countries:

```python
# Compare growth rates
GDPForecaster.compare_countries(['USA', 'CHN', 'DEU'], metric='growth')

# Compare normalized GDP
GDPForecaster.compare_countries(['USA', 'CHN', 'DEU'], metric='gdp')
```

### Regional Analysis

Analyze regional economic performance:

```python
regions = {
    'North America': ['USA', 'CAN', 'MEX'],
    'Western Europe': ['DEU', 'FRA', 'GBR', 'ITA'],
    'East Asia': ['JPN', 'CHN', 'KOR']
}
region_results = forecaster.run_regional_analysis(regions)
```

### Forecast Uncertainty

Visualize forecast uncertainty with confidence intervals:

```python
# 90% confidence interval
forecaster.plot_confidence_intervals(confidence_level=0.90)

# 95% confidence interval
forecaster.plot_confidence_intervals(confidence_level=0.95)
```

## Installation Requirements

All dependencies are listed in the `requirements.txt` file. The main requirements are:

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
requests>=2.26.0
tabulate>=0.8.9
scipy>=1.7.0
```

You can install all dependencies at once using:
```bash
pip install -r requirements.txt
```

## Limitations and Considerations

- Model performance varies by country and data availability
- Economic shocks or structural changes may affect forecast accuracy
- Results should be interpreted alongside expert domain knowledge
- Some countries have limited historical data for certain indicators
- The World Bank API may have rate limits for frequent requests

## Future Enhancements

- Incorporate additional data sources
- Implement more advanced ML models (LSTM, XGBoost)
- Add geopolitical risk indicators
- Include global economic cycle effects
- Enhance confidence interval estimation with bootstrapping
- Add interactive web dashboard

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.