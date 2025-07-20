import joblib
import numpy as np
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

class OptimizationEngine:
    """
    Finds the optimal ticket price to maximize revenue based on the demand model
    and business constraints.
    """
    def __init__(self, model_path=config.DEMAND_FORECAST_MODEL_PATH):
        """
        Initializes the engine by loading the trained demand forecast model.
        """
        try:
            self.model = joblib.load(model_path)
            logging.info("OptimizationEngine initialized: Demand forecast model loaded successfully.")
        except FileNotFoundError:
            self.model = None
            logging.error(f"Error: Demand model not found at {model_path}. Please train the model first.")
            sys.exit(1)

    def find_optimal_price(self, base_features: pd.DataFrame, price_range: tuple, increment: float = 1.0) -> dict:
        """
        Performs a grid search over a range of prices to find the one that maximizes revenue.

        Args:
            base_features (pd.DataFrame): A DataFrame with a single row containing all
                                          the other features needed for the model.
            price_range (tuple): A tuple defining the min and max price to test (e.g., (50, 400)).
            increment (float): The step size for iterating through the price range.

        Returns:
            dict: A dictionary containing the optimal price, predicted sales at that price,
                  and the maximum projected revenue.
        """
        if self.model is None:
            return {"error": "Model not loaded."}

        best_price = 0
        max_revenue = -1
        sales_at_best_price = 0
        
        logging.info(f"Starting price optimization search in range {price_range} with increment {increment}...")
        sim_features = base_features.copy()

        for price in np.arange(price_range[0], price_range[1] + increment, increment):
            sim_features['ticket_price'] = price
            predicted_sales = self.model.predict(sim_features)[0]
            predicted_sales = max(0, int(predicted_sales))
            current_revenue = price * predicted_sales

            if current_revenue > max_revenue:
                max_revenue = current_revenue
                best_price = price
                sales_at_best_price = predicted_sales
        
        result = {
            "optimal_price": best_price,
            "sales_at_optimal_price": sales_at_best_price,
            "max_revenue": max_revenue
        }
        
        logging.info(f"Optimization complete: Optimal price is {best_price:.2f}, yielding a max revenue of {max_revenue:.2f} from {sales_at_best_price} sales.")
        return result

# --- Example Usage ---
if __name__ == '__main__':
    logging.info("Running OptimizationEngine as a standalone script...")

    # Create a sample feature set for a hypothetical match.
    sample_base_features = SAMPLE_BASE_FEATURES
    features_df = pd.DataFrame([sample_base_features])

    # Initialize the optimization engine
    opt_engine = OptimizationEngine()
    
    # Define business constraints for the price search
    price_constraints = (75, 350)  # Min price €75, Max price €350
    
    # Find the optimal price
    optimization_result = opt_engine.find_optimal_price(
        base_features=features_df,
        price_range=price_constraints,
        increment=5  # Check price in steps of €5
    )
    
    # Print the results in a readable format
    print("\n--- Optimization Result ---")
    if "error" in optimization_result:
        print(f"An error occurred: {optimization_result['error']}")
    else:
        print(f"Optimal Price Recommendation: €{optimization_result['optimal_price']:.2f}")
        print(f"Predicted Sales at this Price: {optimization_result['sales_at_optimal_price']} tickets")
        print(f"Maximum Projected Revenue: €{optimization_result['max_revenue']:.2f}")
    print("---------------------------\n")
