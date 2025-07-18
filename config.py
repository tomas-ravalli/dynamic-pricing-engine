import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data Paths ---
# Path to the directory for synthetic data
DATA_DIR = os.path.join(BASE_DIR, 'data')
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, '03_synthetic', 'synthetic_match_data.csv')

# Path for processed data, ready for modeling
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, '02_processed', 'processed_match_data.csv')

# --- Model Paths ---
# Path to save the trained demand forecast model
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEMAND_FORECAST_MODEL_PATH = os.path.join(MODELS_DIR, 'demand_forecast_model.joblib')