#!/bin/bash

# Get the current directory name (whatever it's called)
ROOT_DIR=$(pwd)

echo "Creating hydroponics project structure in: $ROOT_DIR"

# Create the main DS_Hydroponics directory
mkdir -p "DS_Hydroponics"

# Create all subdirectories
mkdir -p "DS_Hydroponics/src"
mkdir -p "DS_Hydroponics/data" 
mkdir -p "DS_Hydroponics/models"
mkdir -p "DS_Hydroponics/docs"
mkdir -p "DS_Hydroponics/output"
mkdir -p "DS_Hydroponics/output/individual_sensors"

# Create all files exactly as shown in the structure
touch "DS_Hydroponics/src/DS Hydroponics.ipynb"
touch "DS_Hydroponics/data/hydroponics_dataset.csv"
touch "DS_Hydroponics/models/hydroponics_model.pkl"
touch "DS_Hydroponics/docs/README.md"
touch "DS_Hydroponics/docs/report.pdf"
touch "DS_Hydroponics/output/simulation_report_YYYYMMDD_HHMMSS.csv"
touch "DS_Hydroponics/output/dataset_summary.csv"
touch "DS_Hydroponics/output/all_sensors_together.png"
touch "DS_Hydroponics/output/individual_sensors/temperature.png"
touch "DS_Hydroponics/output/individual_sensors/humidity.png"
touch "DS_Hydroponics/output/individual_sensors/etc..."
touch "DS_Hydroponics/output/plant_health_trend.png"
touch "DS_Hydroponics/requirements.txt"

echo "✅ Complete hydroponics project structure created successfully!"
echo "📁 Structure created in: $ROOT_DIR/DS_Hydroponics"

# Show the created structure
tree DS_Hydroponics 2>/dev/null || find DS_Hydroponics -type f | sort

