import joblib
import pandas as pd
import numpy as np
import logging
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config
from src.decision_engine.constants import SAMPLE_BASE_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizationEngine:
    """
    Finds the revenue-maximizing price by simulating outcomes across a range of prices.
    """
    def __init__(self, model_path=config.DEMAND_FORECAST_MODEL_PATH, pipeline_path=config.FEATURE_PIPELINE_PATH):
        """
        Initializes the engine by loading the trained model and feature pipeline.
        """
        try:
            self.model = joblib.load(model_path)
            self.pipeline = joblib.load(pipeline_path)
            logging.info("OptimizationEngine initialized: Model and feature pipeline loaded successfully.")
        except FileNotFoundError as e:
            self.model = None
            self.pipeline = None
            logging.error(f"Error loading files: {e}. Please train the model and build features first.")
            sys.exit(1)

    def run_optimization(self, base_features: pd.DataFrame, price_range: tuple = (50, 251), step: int = 5) -> tuple:
        """
        Iterates through a price range to find the price that maximizes revenue.
        """
        if self.model is None:
            logging.error("Cannot run optimization, model not loaded.")
            return None, None

        logging.info(f"Starting price optimization across range {price_range} with step {step}...")
        
        best_price = 0
        max_revenue = 0
        
        # Iterate through the defined price range
        for price in range(price_range[0], price_range[1], step):
            # Create a full feature set for prediction
            sim_features = base_features.copy()
            sim_features['ticket_price'] = price
            
            # --- CRITICAL STEP ---
            # Transform the data using the loaded pipeline
            processed_features = self.pipeline.transform(sim_features)
            
            # Make prediction on the PROCESSED data
            predicted_sales = self.model.predict(processed_features)[0]
            predicted_sales = max(0, int(predicted_sales))
            projected_revenue = price * predicted_sales

            # Check if this price yields a higher revenue
            if projected_revenue > max_revenue:
                max_revenue = projected_revenue
                best_price = price
        
        logging.info(f"Optimization complete: Optimal price is €{best_price:.2f} with a max revenue of €{max_revenue:,.2f}.")
        return best_price, max_revenue

# --- Example Usage ---
if __name__ == '__main__':
    logging.info("Running OptimizationEngine as a standalone script...")

    # Create a DataFrame from the sample features
    features_df = pd.DataFrame([SAMPLE_BASE_FEATURES])

    # Initialize the optimization engine
    opt_engine = OptimizationEngine()
    
    # Find the optimal price
    optimal_price, max_revenue = opt_engine.run_optimization(base_features=features_df)
    
    # Print the results
    print("\n--- Optimization Result ---")
    if optimal_price is not None:
        print(f"Optimal Price Recommendation: €{optimal_price:.2f}")
        print(f"Maximum Estimated Revenue: €{max_revenue:,.2f}")
    else:
        print("Could not determine optimal price.")
    print("---------------------------\n")