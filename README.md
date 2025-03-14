# GDP Forecaster

A comprehensive toolkit for forecasting GDP of countries worldwide using ElasticNet regression with multiple economic indicators.

## Project Overview

The GDP Forecaster is a Python-based forecasting system that leverages historical economic data to predict future GDP values for any country. It incorporates multiple economic indicators such as population, Gini index, inflation rates, and other macroeconomic factors to improve forecast accuracy.

### Key Features

- Forecasts GDP for any country for a user-defined horizon (typically 5 years)
- Incorporates 15+ economic indicators from World Bank data
- Uses ElasticNet regression with automatic hyperparameter tuning
- Includes backtesting capabilities to validate model performance
- Compares forecasts with official IMF projections when available
- Forecasts individual economic indicators using appropriate time series methods
- Visualizes forecasts with informative plots
- Analyzes feature importance to understand key GDP drivers
- Interactive Streamlit app for user-friendly forecasting

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

# 3. Launch the interactive Streamlit app
streamlit run app.py
# The app will open in your default web browser
```

## Project Structure

The GDP Forecaster is organized into the following structure:

- `app.py`: Main Streamlit application file for interactive forecasting
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
  - `app_utils.py`: Helper functions for the Streamlit application
  - `utils.py`: Utility functions including report generation
- `results/`: Generated forecasts, visualizations, and reports (organized by country)
- `logs/`: Log files from forecasting runs

## Streamlit App

The project includes an interactive Streamlit app (`app.py`) that provides a user-friendly interface for GDP forecasting.

### Running the App

To launch the Streamlit app:

```bash
streamlit run app.py
```

### App Features

The app is organized into tabs for easy navigation:

1. **Summary**: Overview of forecast results and model performance metrics
2. **Backtesting**: Detailed analysis of model accuracy using historical validation
3. **Economic Indicators**: Forecasts for all underlying economic variables
4. **Feature Importance**: Ranking of indicators by their impact on GDP
5. **Forecast Visualization**: Comprehensive GDP projection charts

Each tab includes detailed explanations to help interpret the results and understand the economic implications of the forecasts.

## Data Sources

The forecaster primarily uses data from the World Bank API:

- **World Bank Open Data API**
  - GDP (current US$)
  - GDP growth (annual %)
  - Population data
  - Gini index
  - Trade indicators (exports/imports as % of GDP)
  - Foreign direct investment
  - Government debt
  - Unemployment rates
  - Inflation rates
  - Sectoral composition (agriculture, industry, services)
  - Research and development expenditure
  - Education and healthcare spending
  - Energy consumption

- **International Monetary Fund (IMF)**
  - World Economic Outlook Database forecasts (simulated in current implementation)

## Core Classes

The GDP Forecaster consists of three main classes:

### 1. DataCollector

Responsible for retrieving economic data from various sources, primarily the World Bank API.

```python
from utils.data_collector import DataCollector

# Initialize the data collector
collector = DataCollector(cache_dir='./data')

# Get list of available countries
countries = collector.get_country_list()
print(f"Available countries: {len(countries)}")

# Fetch data for a specific country
data = collector.fetch_world_bank_data("USA")
print(f"Retrieved {len(data.columns)} indicators for USA")

# Get IMF forecasts for comparison
imf_data = collector.fetch_imf_forecasts("USA")
```

### 2. VariableForecaster

Handles the forecasting of individual economic variables using appropriate time series methods.

```python
from utils.variable_forecaster import VariableForecaster
import pandas as pd

# Initialize the variable forecaster
var_forecaster = VariableForecaster()

# Example: Forecast population for 5 years
historical_pop = pd.Series([331002651, 329466283, 326687501, 324985539, 323015995],
                           index=pd.DatetimeIndex(['2020-01-01', '2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01']))
pop_forecast = var_forecaster.forecast_population(historical_pop, horizon=5)
print("Population forecast:")
print(pop_forecast)

# Example: Forecast an economic indicator using ARIMA
gdp_growth = pd.Series([2.3, 2.9, 2.3, 1.7, 1.7],
                       index=pd.DatetimeIndex(['2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01', '2015-01-01']))
growth_forecast = var_forecaster.forecast_arima(gdp_growth, horizon=5)
print("GDP growth forecast:")
print(growth_forecast)

# Select appropriate method based on variable type
inflation = pd.Series([1.8, 2.4, 2.1, 1.3, 0.1],
                     index=pd.DatetimeIndex(['2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01', '2015-01-01']))
method = var_forecaster.select_forecast_method("FP.CPI.TOTL.ZG", inflation, horizon=5)
print("Inflation forecast:")
print(method)
```

### 3. GDPForecaster

The main class that integrates data collection, variable forecasting, and GDP prediction using ElasticNet regression.

```python
from utils.gdp_forecaster import GDPForecaster

# Initialize the GDP forecaster
forecaster = GDPForecaster()

# Load data for a specific country
historical_data = forecaster.load_data("USA")
print(f"Loaded {len(historical_data)} years of data with {historical_data.shape[1]} indicators")

# First run backtests to assess model performance
backtest_results = forecaster.run_rolling_backtests(n_years=3)
mape = np.mean(np.abs(backtest_results['Percent_Error']))
print(f"Backtest Results - Average MAPE: {mape:.2f}%")

# Then train the final model for forecasting (using all available data)
metrics = forecaster.train_model(test_years=0)
print("Model trained for forecasting")

# Forecast economic indicators for 5 years
forecasted_features = forecaster.forecast_features(horizon=5)
print(f"Forecasted {len(forecasted_features.columns)} economic indicators")

# Generate GDP forecast
gdp_forecast = forecaster.forecast_gdp(horizon=5)
print("GDP Forecast:")
print(gdp_forecast)

# Get feature importance
feature_importance = forecaster.get_model_coefficients()
print("Top 3 most important factors:")
for i in range(min(3, len(feature_importance))):
    print(f"{i+1}. {feature_importance['Feature'].iloc[i]}: {feature_importance['Normalized_Importance'].iloc[i]:.2%}")

# Create visualization
fig = forecaster.plot_forecast(show_history_years=10)
fig.savefig("usa_gdp_forecast.png")

# Generate detailed summary
forecaster.print_forecast_summary()

# Export results to files
output_dir = forecaster.export_results("./output")
print(f"Results exported to {output_dir}")
```

## Methodology

### Data Collection and Preprocessing
- Historical data is collected from the World Bank API
- Data is cleaned, with missing values handled using interpolation and appropriate fill methods
- Data is cached locally for faster access in future runs

### Forecasting Approach
1. **Backtesting**
   - Run rolling backtests to assess model performance on historical data
   - Calculate metrics like MAPE and RMSE to understand forecast reliability

2. **Indicator Forecasting**
   - Population: Growth rate method
   - Economic indicators: ARIMA models
   - Structural indicators: Exponential smoothing

3. **GDP Forecasting**
   - Model: ElasticNet regression (combines L1 and L2 regularization)
   - Hyperparameter tuning: Grid search with time series cross-validation
   - Recent data is weighted more heavily than older data
   - Train final model on all available data for future forecasting

## Running the Forecaster

There are two main ways to use the GDP Forecaster:

### 1. Using the Streamlit App (Recommended)

For the most user-friendly experience, run the Streamlit app:

```bash
streamlit run app.py
```

The app allows you to:
1. Select a country from the dropdown menu
2. Set forecast parameters (horizon, backtest years)
3. Adjust advanced model parameters (alpha, L1 ratio)
4. Run backtests to assess model reliability
5. Generate forecasts for future GDP
6. Explore economic indicators and feature importance
7. Download results for further analysis

### 2. Using the GDPForecaster Programmatically

For more control and custom integration, use the GDPForecaster class directly:

```python
from utils.gdp_forecaster import GDPForecaster

# Initialize and configure the forecaster
forecaster = GDPForecaster()
data = forecaster.load_data("JPN")  # Japan

# First run backtests to evaluate performance
backtest_results = forecaster.run_rolling_backtests(n_years=3)
print(f"Backtest MAPE: {np.mean(np.abs(backtest_results['Percent_Error'])):.2f}%")

# Then train the final model using all data
forecaster.train_model()

# Generate and visualize forecast
gdp_forecast = forecaster.forecast_gdp(horizon=7)
fig = forecaster.plot_forecast()

# Save results
forecaster.export_results("./output/japan")
```

## Installation Requirements

All dependencies are listed in the `requirements.txt` file. The main requirements are:

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
requests>=2.26.0
tabulate>=0.8.9
scipy>=1.7.0
streamlit>=1.12.0
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
- Add confidence intervals for forecast uncertainty
- Implement cross-country and regional comparison capabilities
- Add sensitivity analysis for key economic variables
- Enhance the Streamlit app with more interactive features

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.