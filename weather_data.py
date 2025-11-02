"""
Weather data generator for training the RL agent.
Supports both synthetic data generation and realistic German weather data simulation.
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta


class WeatherDataGenerator:
    """Generates synthetic weather temperature data or simulates realistic German weather patterns."""
    
    def __init__(self, days=365, seed=42, use_real_data=False, location='germany'):
        """
        Initialize the weather data generator.
        
        Args:
            days: Number of days to generate data for (for synthetic data)
            seed: Random seed for reproducibility (for synthetic data)
            use_real_data: If True, generate realistic German weather patterns; if False, generate simple synthetic data
            location: Location for real data ('germany' or 'rheinland-pfalz')
        """
        self.days = days
        self.use_real_data = use_real_data
        self.location = location
        np.random.seed(seed)
        
        if use_real_data:
            # Try to fetch from API, fall back to realistic simulation
            self.data = self._fetch_or_simulate_real_data()
        else:
            self.data = self._generate_synthetic_data()
    
    def _fetch_or_simulate_real_data(self):
        """
        Fetch real weather data from Germany or simulate realistic patterns.
        
        Returns:
            DataFrame with date and temperature columns
        """
        print(f"Fetching real weather data for {self.location}...")
        
        # Coordinates for Germany (center) or Rheinland-Pfalz (Mainz)
        if self.location.lower() == 'rheinland-pfalz':
            latitude = 49.9929  # Mainz, Rheinland-Pfalz
            longitude = 8.2473
            location_name = "Rheinland-Pfalz (Mainz)"
        else:
            latitude = 51.1657  # Germany (center)
            longitude = 10.4515
            location_name = "Germany (center)"
        
        # Try to get last 10 years of data from API
        end_date = datetime(2025, 12, 31)
        start_date = datetime(2016, 1, 1)
        
        try:
            # Open-Meteo Historical Weather API
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_mean",
                "timezone": "Europe/Berlin"
            }
            
            print(f"  Location: {location_name}")
            # Note: Coordinates are public information (city centers), not sensitive data
            print(f"  Coordinates: {latitude}°N, {longitude}°E")
            print(f"  Attempting to fetch from API...")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract daily data
            if 'daily' in data and 'time' in data['daily']:
                dates = pd.to_datetime(data['daily']['time'])
                temperatures = data['daily']['temperature_2m_mean']
                
                # Create DataFrame
                df = pd.DataFrame({
                    'date': dates,
                    'temperature': temperatures
                })
                
                # Remove any NaN values
                df = df.dropna()
                
                print(f"  ✓ Successfully fetched {len(df)} days of real temperature data")
                print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
                print(f"  Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
                
                return df
                
        except Exception as e:
            print(f"  ⚠ API not accessible: {type(e).__name__}")
            print(f"  ℹ Generating realistic German weather simulation instead...")
        
        # Fall back to realistic simulation
        return self._generate_realistic_german_data(start_date, end_date, location_name)
    
    def _generate_realistic_german_data(self, start_date, end_date, location_name):
        """
        Generate realistic German weather data based on historical patterns.
        
        This simulates German weather with:
        - Cold winters (Dec-Feb): avg -1°C to 4°C
        - Mild springs (Mar-May): avg 5°C to 15°C
        - Warm summers (Jun-Aug): avg 15°C to 24°C
        - Cool autumns (Sep-Nov): avg 6°C to 14°C
        - Daily temperature variations and realistic noise
        
        Args:
            start_date: Start date for simulation
            end_date: End date for simulation
            location_name: Name of the location
            
        Returns:
            DataFrame with realistic German temperature data
        """
        print(f"  Location: {location_name}")
        print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Generating realistic German weather patterns...")
        
        # Generate dates
        num_days = (end_date - start_date).days + 1
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        temperatures = []
        for i, date in enumerate(dates):
            # Get day of year (1-365/366)
            day_of_year = date.timetuple().tm_yday
            
            # Seasonal pattern for Germany (colder in winter, warmer in summer)
            # Peak cold around day 31 (end of Jan), peak warm around day 213 (early Aug)
            seasonal_avg = 11.5 + 13.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
            
            # Add some year-to-year variation
            year_variation = np.random.normal(0, 1.5)
            
            # Add weather variability (some days colder/warmer than average)
            # Create multi-day weather patterns (cold fronts, warm spells)
            if i % 7 == 0:  # Create new weather pattern every ~week
                pattern_strength = np.random.normal(0, 2.5)
            
            # Daily variation
            daily_noise = np.random.normal(0, 2)
            
            # Combine all factors
            temp = seasonal_avg + year_variation + pattern_strength + daily_noise
            
            # Add occasional extreme events (heat waves, cold snaps)
            if np.random.random() < 0.05:  # 5% chance
                extreme = np.random.choice([-1, 1]) * np.random.uniform(5, 10)
                temp += extreme
            
            temperatures.append(temp)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperatures
        })
        
        print(f"  ✓ Generated {len(df)} days of realistic temperature data")
        print(f"  Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
        print(f"  Average temperature: {df['temperature'].mean():.1f}°C (typical for Germany: ~10°C)")
        
        return df
    
    def _generate_synthetic_data(self):
        """
        Generate simple synthetic temperature data with seasonal patterns.
        
        Returns:
            DataFrame with date and temperature columns
        """
        # Generate dates
        dates = pd.date_range(start='2023-01-01', periods=self.days, freq='D')
        
        # Generate temperatures with seasonal pattern
        # Base temperature around 15°C with seasonal variation
        day_of_year = np.arange(self.days)
        seasonal_pattern = 10 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
        
        # Add random noise
        noise = np.random.normal(0, 3, self.days)
        
        # Combine patterns
        temperatures = 15 + seasonal_pattern + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperatures
        })
        
        return df
    
    def get_data(self):
        """
        Get the weather data.
        
        Returns:
            DataFrame with weather data
        """
        return self.data
    
    def get_temperature_sequence(self, window_size=30):
        """
        Get temperature sequences for training.
        
        Args:
            window_size: Number of consecutive days to include in each sequence
            
        Returns:
            List of temperature sequences
        """
        temps = self.data['temperature'].values
        sequences = []
        
        for i in range(len(temps) - window_size):
            sequences.append(temps[i:i + window_size + 1])
        
        return sequences
