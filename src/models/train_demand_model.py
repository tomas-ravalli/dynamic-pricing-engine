import pandas as pd
import joblib
import logging
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to train the demand forecast model.
    It loads processed data, splits it, trains a model, and saves it.
    """
    logging.info("Starting model training process...")

    # Load the processed dataset
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        logging.info(f"Successfully loaded processed data from {config.PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        logging.error(f"Error: Processed data file not found at {config.PROCESSED_DATA_PATH}. Please run src.features.build_features.py first.")
        return

    # Define target and features
    target_column = 'zone_historical_sales'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split into training and testing sets. Training set size: {X_train.shape[0]}")

    # Initialize and train the model
    # Using GradientBoostingRegressor as it's robust and often performs well
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    logging.info("Training the Gradient Boosting Regressor model...")
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Evaluate the model (optional, but good practice)
    score = model.score(X_test, y_test)
    logging.info(f"Model R^2 score on the test set: {score:.4f}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(config.DEMAND_FORECAST_MODEL_PATH), exist_ok=True)

    # Save the trained model
    joblib.dump(model, config.DEMAND_FORECAST_MODEL_PATH)
    logging.info(f"Model saved successfully to {config.DEMAND_FORECAST_MODEL_PATH}")

if __name__ == '__main__':
    main()
