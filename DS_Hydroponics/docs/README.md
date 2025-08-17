# DS_Hydroponics: Intelligent Hydroponics System Optimization

## Overview

DS_Hydroponics is a data science and machine learning project designed to simulate, analyze, and optimize hydroponic crop growth using synthetic sensor data and robust predictive models. It enables monitoring of critical variables, predicts plant health/yield, and demonstrates advanced ML techniques for agricultural IoT.

---

## Objectives

- Generate realistic hydroponics sensor data (with noise, missing values, and outliers).
- Train multiple machine learning models to predict plant growth, yield, health scores, and disease risk.
- Ensure robust data handling: error catching, imputation, outlier correction, and thorough validation.
- Visualize system status and trends using clear plots and reports.
- Provide ready-to-use scripts for predictions and analytics on new sensor data.

---

## Project Structure

DS_Hydroponics/
│
├── src/
│ ├── DS_Hydroponics.ipynb # Main ML pipeline (Jupyter/Script)
│ └── example_usage.py # Demo for model prediction with new data
│
├── data/
│ ├── generate_hydroponics_dataset.py # Synthetic dataset generator
│ └── hydroponics_dataset.csv # Generated dataset
│
├── models/
│ └── hydroponics_model.pkl # Saved ML models (pickle)
│
├── docs/
│ ├── README.md # Documentation (this file)
│ └── report.pdf # Detailed report (optional)
│
├── output/
│ ├── simulation_report_YYYYMMDD_HHMMSS.csv
│ ├── dataset_summary.csv
│ ├── all_sensors_together.png
│ ├── individual_sensors/
│ │ ├── temperature.png
│ │ ├── humidity.png
│ │ └── ... (other sensors)
│ └── plant_health_trend.png
│
└── requirements.txt # Python library requirements



---

## Key Script Descriptions

- **`generate_hydroponics_dataset.py`**  
  Generates a time-series sensor dataset with realistic noise, seasonal patterns, and random missing values/outliers. Output: `hydroponics_dataset.csv`.

- **`DS_Hydroponics.ipynb` / `DS_Hydroponics.py`**  
  ML pipeline to clean data, engineer features, train/validate models (RandomForest, GradientBoosting, Ridge), handle inconsistencies, and save the model.

- **`example_usage.py`**  
  Example template for using the trained model to predict results from new sensor readings.

- **`hydroponics_model.pkl`**  
  Serialized models and preprocessing pipeline; load this for real-time predictions.

- **`output/` (folder)**  
  Contains performance reports, CSV summaries, and sensor/health visualizations.

---

## Usage Guide

### 1. **Install Dependencies**
    pip install -r requirements.txt

### 2. **Generate Dataset**
    python data/generate_hydroponics_dataset.py

*Generates `hydroponics_dataset.csv` inside the data folder.*

### 3. **Train Models**
    python src/DS_Hydroponics.py


*Saves models in `models/hydroponics_model.pkl` and generates outputs in the `output/` folder.*

### 4. **Predict with New Data**
See `src/example_usage.py` for reference code.

---

## Features

- Realistic hydroponic sensor data simulation (environmental, nutrient, operational variables)
- Robust preprocessing: missing value imputation, outlier correction, feature engineering
- Multiple model training and hyperparameter optimization
- Performance metrics and feature importance summaries
- Sensor trend and plant health visualization outputs
- Modular, customizable pipeline and scripts

---

## Authors

- [Mukesh T]

---

## License

MIT License

---

## Questions

Open a repository issue or contact the maintainer.
