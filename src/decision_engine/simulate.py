import joblib
import pandas as pd
import logging
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config
from src.decision_engine.constants import SAMPLE_BASE_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimulationEngine:
    """
    Handles 'what-if' scenarios by loading the trained model and feature pipeline
    to simulate the impact of a given price on demand and revenue.
    """
    def __init__(self, model_path=config.DEMAND_FORECAST_MODEL_PATH, pipeline_path=config.FEATURE_PIPELINE_PATH):
        """
        Initializes the engine by loading the trained model and the feature pipeline.
        """
        try:
            self.model = joblib.load(model_path)
            self.pipeline = joblib.load(pipeline_path)
            logging.info("SimulationEngine initialized: Model and feature pipeline loaded successfully.")
        except FileNotFoundError as e:
            self.model = None
            self.pipeline = None
            logging.error(f"Error loading files: {e}. Please train the model and build features first.")
            sys.exit(1)

    def run_simulation(self, price: float, base_features: pd.DataFrame) -> dict:
        """
        Predicts sales demand and calculates revenue for a hypothetical price
        after applying the correct feature transformations.
        """
        if self.model is None:
            return {"error": "Model not loaded."}

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

        result = {
            "simulated_price": price,
            "predicted_sales": predicted_sales,
            "projected_revenue": projected_revenue
        }
        
        logging.info(f"Simulation complete: For a price of {price:.2f}, predicted sales are {predicted_sales} units, with a projected revenue of {projected_revenue:.2f}.")
        return result

# --- Example Usage ---
if __name__ == '__main__':
    logging.info("Running SimulationEngine as a standalone script...")

    # Create a DataFrame from the sample features
    features_df = pd.DataFrame([SAMPLE_BASE_FEATURES])

    # Initialize the simulation engine
    sim_engine = SimulationEngine()
    
    # Simulate the outcome for a hypothetical price of €195.00
    simulation_result = sim_engine.run_simulation(price=195.00, base_features=features_df)
    
    # Print the results
    print("\n--- Simulation Result ---")
    if "error" in simulation_result:
        print(f"An error occurred: {simulation_result['error']}")
    else:
        print(f"Simulated Price: €{simulation_result['simulated_price']:.2f}")
        print(f"Predicted Sales: {simulation_result['predicted_sales']} tickets")
        print(f"Projected Revenue: €{simulation_result['projected_revenue']:.2f}")
    print("-------------------------\n")