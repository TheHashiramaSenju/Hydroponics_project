# data/generate_hydroponics_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_hydroponics_dataset(num_samples=50000):
    """
    Generate a comprehensive hydroponic system dataset
    """
    np.random.seed(42)
    random.seed(42)
    
    # Time series data
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)]
    
    data = []
    
    for i, date in enumerate(dates):
        # Environmental factors
        hour = date.hour
        day_of_year = date.timetuple().tm_yday
        
        # Seasonal patterns
        seasonal_temp = 22 + 8 * np.sin(2 * np.pi * day_of_year / 365)
        seasonal_humidity = 60 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/4)
        
        # Daily patterns
        daily_temp_variation = 3 * np.sin(2 * np.pi * hour / 24)
        daily_humidity_variation = -10 * np.sin(2 * np.pi * hour / 24)
        
        # Base measurements with realistic noise
        temperature = seasonal_temp + daily_temp_variation + np.random.normal(0, 1.5)
        humidity = seasonal_humidity + daily_humidity_variation + np.random.normal(0, 3)
        
        # pH levels (optimal around 5.5-6.5 for most plants)
        ph_base = 6.0 + 0.3 * np.sin(2 * np.pi * i / (24*7)) # Weekly cycle
        ph_level = max(4.5, min(8.0, ph_base + np.random.normal(0, 0.2)))
        
        # Electrical conductivity (nutrient concentration)
        ec_base = 1.8 + 0.4 * np.sin(2 * np.pi * i / (24*14)) # Bi-weekly cycle
        ec_level = max(0.8, min(3.2, ec_base + np.random.normal(0, 0.15)))
        
        # Water temperature
        water_temp = temperature - 2 + np.random.normal(0, 1)
        
        # Light intensity (PAR - Photosynthetically Active Radiation)
        if 6 <= hour <= 18:  # Daylight hours
            light_base = 800 + 400 * np.sin(np.pi * (hour - 6) / 12)
            light_intensity = max(0, light_base + np.random.normal(0, 100))
        else:
            light_intensity = np.random.normal(20, 10)  # Minimal light at night
        
        # CO2 levels
        co2_base = 400 + 200 * (1 - (hour - 12)**2 / 144)  # Peak at noon
        co2_level = max(300, co2_base + np.random.normal(0, 30))
        
        # Water level (percentage)
        water_level = max(10, min(100, 80 + 15 * np.sin(2 * np.pi * i / (24*3)) + np.random.normal(0, 5)))
        
        # Nutrient levels
        nitrogen = max(50, min(300, 180 + 50 * np.sin(2 * np.pi * i / (24*10)) + np.random.normal(0, 15)))
        phosphorus = max(20, min(150, 80 + 30 * np.sin(2 * np.pi * i / (24*12)) + np.random.normal(0, 8)))
        potassium = max(80, min(400, 220 + 60 * np.sin(2 * np.pi * i / (24*8)) + np.random.normal(0, 20)))
        
        # System status indicators
        pump_status = 1 if random.random() > 0.05 else 0  # 95% uptime
        fan_status = 1 if temperature > 25 else (1 if random.random() > 0.2 else 0)
        heater_status = 1 if temperature < 18 else 0
        
        # Plant health indicators (target variables)
        # Growth rate (cm/day)
        optimal_conditions = (
            (5.5 <= ph_level <= 6.5) * 0.3 +
            (1.4 <= ec_level <= 2.2) * 0.3 +
            (20 <= temperature <= 26) * 0.2 +
            (light_intensity > 400) * 0.2
        )
        growth_rate = max(0, 2.5 + 3 * optimal_conditions + np.random.normal(0, 0.5))
        
        # Yield prediction (grams per plant)
        yield_factor = optimal_conditions + np.random.normal(0, 0.1)
        expected_yield = max(50, 200 + 300 * yield_factor + np.random.normal(0, 30))
        
        # Health score (0-100)
        health_score = max(0, min(100, 70 + 25 * optimal_conditions + np.random.normal(0, 5)))
        
        # Disease probability
        stress_factors = (
            (ph_level < 5.0 or ph_level > 7.5) * 0.3 +
            (humidity > 85 or humidity < 40) * 0.3 +
            (temperature > 30 or temperature < 15) * 0.4
        )
        disease_risk = min(1, max(0, 0.1 + stress_factors + np.random.normal(0, 0.05)))
        
        data.append({
            'timestamp': date,
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'ph_level': round(ph_level, 2),
            'ec_level': round(ec_level, 2),
            'water_temp': round(water_temp, 2),
            'light_intensity': round(light_intensity, 1),
            'co2_level': round(co2_level, 1),
            'water_level': round(water_level, 1),
            'nitrogen_ppm': round(nitrogen, 1),
            'phosphorus_ppm': round(phosphorus, 1),
            'potassium_ppm': round(potassium, 1),
            'pump_status': pump_status,
            'fan_status': fan_status,
            'heater_status': heater_status,
            'growth_rate': round(growth_rate, 3),
            'expected_yield': round(expected_yield, 1),
            'health_score': round(health_score, 1),
            'disease_risk': round(disease_risk, 3)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating hydroponics dataset...")
    df = generate_hydroponics_dataset(50000)

    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    missing_columns = ['temperature', 'humidity', 'ph_level', 'ec_level']
    
    for idx in missing_indices:
        col = np.random.choice(missing_columns)
        df.at[idx, col] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.001), replace=False)
    for idx in outlier_indices:
        df.at[idx, 'temperature'] = np.random.choice([5, 45])  # Extreme temperatures
    
    df.to_csv('hydroponics_dataset.csv', index=False)
    print(f"Dataset saved with {len(df)} samples and {len(df.columns)} features")
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
