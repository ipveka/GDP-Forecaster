"""
DataCollector module for retrieving economic data from various sources.
"""

import os
import json
import logging
import pandas as pd
import requests
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collector')


class DataCollector:
    """Class to collect and preprocess data from various sources."""
    
    def __init__(self, cache_dir: str = './data'):
        """
        Initialize the DataCollector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.wb_indicators = {
            'NY.GDP.MKTP.CD': 'GDP (current US$)',
            'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
            'SP.POP.TOTL': 'Population, total',
            'SI.POV.GINI': 'Gini index',
            'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',
            'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
            'BX.KLT.DINV.WD.GD.ZS': 'Foreign direct investment, net inflows (% of GDP)',
            'GC.DOD.TOTL.GD.ZS': 'Central government debt, total (% of GDP)',
            'SL.UEM.TOTL.ZS': 'Unemployment, total (% of total labor force)',
            'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
            'GB.XPD.RSDV.GD.ZS': 'Research and development expenditure (% of GDP)',
            'SH.XPD.CHEX.GD.ZS': 'Current health expenditure (% of GDP)',
            'SE.XPD.TOTL.GD.ZS': 'Government expenditure on education, total (% of GDP)',
            'EG.USE.PCAP.KG.OE': 'Energy use (kg of oil equivalent per capita)',
            'NV.IND.TOTL.ZS': 'Industry (including construction), value added (% of GDP)',
            'NV.SRV.TOTL.ZS': 'Services, value added (% of GDP)',
            'NV.AGR.TOTL.ZS': 'Agriculture, forestry, and fishing, value added (% of GDP)'
        }
        
    def _cache_path(self, filename: str) -> str:
        """Generate path for cached file."""
        return os.path.join(self.cache_dir, filename)
    
    def fetch_world_bank_data(self, country_code: str) -> pd.DataFrame:
        """
        Fetch data from World Bank API for specified country.
        
        Args:
            country_code: ISO 3-letter country code
            
        Returns:
            DataFrame with all indicators for the country
        """
        cache_file = self._cache_path(f'wb_{country_code}.csv')
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            logger.info(f"Loading cached World Bank data for {country_code}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        logger.info(f"Fetching World Bank data for {country_code}")
        all_data = []
        
        for indicator, name in self.wb_indicators.items():
            url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=100"
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch indicator {indicator}: {response.status_code}")
                    continue
                
                data = response.json()
                if len(data) < 2 or not data[1]:
                    logger.warning(f"No data available for indicator {indicator}")
                    continue
                
                # Extract and transform the data
                indicator_data = [(item['date'], item['value']) for item in data[1] if item['value'] is not None]
                if not indicator_data:
                    continue
                    
                temp_df = pd.DataFrame(indicator_data, columns=['year', indicator])
                temp_df['year'] = pd.to_datetime(temp_df['year'], format='%Y')
                all_data.append(temp_df.set_index('year'))
                
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {str(e)}")
        
        if not all_data:
            raise ValueError(f"No data could be fetched for {country_code}")
        
        # Combine all indicators
        df = pd.concat(all_data, axis=1)
        df = df.sort_index()
        
        # Cache the result
        df.to_csv(cache_file)
        return df
    
    def fetch_imf_forecasts(self, country_code: str) -> pd.DataFrame:
        """
        Fetch IMF World Economic Outlook forecasts.
        
        Args:
            country_code: ISO 3-letter country code
            
        Returns:
            DataFrame with IMF forecasts
        """
        # Note: In a real implementation, you'd interface with IMF data
        # For this example, we'll simulate it with a placeholder
        cache_file = self._cache_path(f'imf_{country_code}.csv')
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            logger.info(f"Loading cached IMF forecast data for {country_code}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Placeholder - in a real implementation, this would fetch from IMF API
        logger.info(f"Note: Using simulated IMF forecasts for {country_code}")
        
        # Current year
        current_year = pd.Timestamp.now().year
        years = range(current_year, current_year + 6)
        
        # Simulate some forecasts based on country code
        # This would be replaced with actual API calls
        import numpy as np
        np.random.seed(hash(country_code) % 10000)  # Deterministic based on country
        growth_rates = 2 + np.random.normal(0, 1, len(years))  # Mean of 2% with variance
        
        forecasts = pd.DataFrame({
            'year': [pd.Timestamp(year=y, month=1, day=1) for y in years],
            'IMF_GDP_Growth': growth_rates
        })
        forecasts.set_index('year', inplace=True)
        
        # Cache the simulated forecasts
        forecasts.to_csv(cache_file)
        return forecasts
    
    def get_country_list(self) -> List[Dict[str, str]]:
        """
        Get list of countries with their codes.
        
        Returns:
            List of dictionaries with country information
        """
        cache_file = self._cache_path('country_list.json')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        url = "http://api.worldbank.org/v2/country?format=json&per_page=300"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch country list: {response.status_code}")
        
        data = response.json()
        
        # Filter for only active countries
        countries = [
            {"name": country["name"], 
             "id": country["id"], 
             "iso3code": country.get("iso2Code", "")}
            for country in data[1]
            if country["region"]["value"] != "Aggregates"
        ]
        
        with open(cache_file, 'w') as f:
            json.dump(countries, f)
            
        return countries