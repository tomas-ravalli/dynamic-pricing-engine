from pathlib import Path

# --- DIRECTORY SETUP ---
# Use pathlib to ensure paths are OS-agnostic
ROOT_DIR = Path(__file__).resolve().parent

# Define key project directories
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
SRC_DIR = ROOT_DIR / "src"

# Define specific data paths
SYNTHETIC_DATA_PATH = DATA_DIR / "03_synthetic" / "synthetic_match_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "02_processed" / "processed_data.csv"

# Define model artifact paths
PRICE_ELASTICITY_MODEL_PATH = MODELS_DIR / "price_elasticity_model.joblib"
DEMAND_FORECAST_MODEL_PATH = MODELS_DIR / "demand_forecast_model.joblib"