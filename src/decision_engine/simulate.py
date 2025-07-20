import joblib
import pandas as pd
import logging
import os
import sys
from .constants import SAMPLE_BASE_FEATURES

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimulationEngine:
    """
    Handles 'what-if' scenarios by simulating the impact of a given price
    on demand and revenue.
    """
    def __init__(self, model_path=config.DEMAND_FORECAST_MODEL_PATH):
        """
        Initializes the engine by loading the trained demand forecast model.
        """
        try:
            self.model = joblib.load(model_path)
            logging.info("SimulationEngine initialized: Demand forecast model loaded successfully.")
        except FileNotFoundError:
            self.model = None
            logging.error(f"Error: Demand model not found at {model_path}. Please train the model first.")
            sys.exit(1)

    def run_simulation(self, price: float, base_features: pd.DataFrame) -> dict:
        """
        Predicts the sales demand and calculates revenue for a hypothetical price.

        Args:
            price (float): The hypothetical price to simulate.
            base_features (pd.DataFrame): A DataFrame with a single row containing all
                                          the other features needed for the model.
                                          It should NOT contain 'ticket_price'.

        Returns:
            dict: A dictionary containing the simulated price, predicted sales,
                  and projected revenue.
        """
        if self.model is None:
            return {"error": "Model not loaded."}

        sim_features = base_features.copy()
        sim_features['ticket_price'] = price
        predicted_sales = self.model.predict(sim_features)[0]
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

    # Create a sample feature set for a hypothetical match.
    # This dictionary contains all features the demand model expects, EXCEPT 'ticket_price'.
    sample_base_features = SAMPLE_BASE_FEATURES
    features_df = pd.DataFrame([sample_base_features])

    # Initialize the simulation engine
    sim_engine = SimulationEngine()
    
    # Simulate the outcome for a hypothetical price of €195.00
    simulation_result = sim_engine.run_simulation(price=195.00, base_features=features_df)
    
    # Print the results in a readable format
    print("\n--- Simulation Result ---")
    if "error" in simulation_result:
        print(f"An error occurred: {simulation_result['error']}")
    else:
        print(f"Simulated Price: €{simulation_result['simulated_price']:.2f}")
        print(f"Predicted Sales: {simulation_result['predicted_sales']} tickets")
        print(f"Projected Revenue: €{simulation_result['projected_revenue']:.2f}")
    print("-------------------------\n")
