# example_usage.py
from src.DS_Hydroponics import HydroponicsMLPipeline
import pandas as pd
import numpy as np

# Load the trained pipeline
pipeline = HydroponicsMLPipeline()
pipeline.load_models('models/hydroponics_model.pkl')

# Example prediction
new_data = pd.DataFrame({
    'temperature': [24.5],
    'humidity': [65.2],
    'ph_level': [6.1],
    'ec_level': [1.8],
    'water_temp': [22.3],
    'light_intensity': [650.0],
    'co2_level': [450.0],
    'water_level': [85.0],
    'nitrogen_ppm': [180.0],
    'phosphorus_ppm': [80.0],
    'potassium_ppm': [220.0],
    'pump_status': [1],
    'fan_status': [1],
    'heater_status': 
})

# Make predictions
growth_pred = pipeline.predict(new_data, 'growth_rate', 'random_forest')
yield_pred = pipeline.predict(new_data, 'expected_yield', 'gradient_boosting')

print(f"Predicted growth rate: {growth_pred:.3f} cm/day")
print(f"Predicted yield: {yield_pred:.1f} grams")
