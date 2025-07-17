import joblib
import numpy as np
import pandas as pd
import logging
import config  # Import the entire config module

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

    def run_simulation(self, price: float, base_features: pd.DataFrame) -> dict:
        """
        Predicts the sales demand and calculates revenue for a hypothetical price.

        Args:
            price (float): The hypothetical price to simulate.
            base_features (pd.DataFrame): A DataFrame with a single row containing all
                                          the other features needed for the model
                                          (e.g., opponent_tier, days_to_match).
                                          It should NOT contain 'ticket_price'.

        Returns:
            dict: A dictionary containing the simulated price, predicted sales,
                  and projected revenue.
        """
        if self.model is None:
            return {"error": "Model not loaded."}

        # Create a copy to avoid modifying the original DataFrame
        sim_features = base_features.copy()
        
        # Set the price for this specific simulation
        sim_features['ticket_price'] = price

        # Predict the demand (sales). The model expects a DataFrame.
        predicted_sales = self.model.predict(sim_features)[0]
        
        # Ensure sales are not negative
        predicted_sales = max(0, int(predicted_sales))

        # Calculate projected revenue
        projected_revenue = price * predicted_sales

        result = {
            "simulated_price": price,
            "predicted_sales": predicted_sales,
            "projected_revenue": projected_revenue
        }
        
        logging.info(f"Simulation complete: For a price of {price:.2f}, predicted sales are {predicted_sales} units, with a projected revenue of {projected_revenue:.2f}.")
        return result


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

    def find_optimal_price(self, base_features: pd.DataFrame, price_range: tuple, increment: float = 1.0) -> dict:
        """
        Performs a grid search over a range of prices to find the one that maximizes revenue.

        Args:
            base_features (pd.DataFrame): A DataFrame with a single row containing all
                                          the other features needed for the model.
                                          It should NOT contain 'ticket_price'.
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

        # Create a copy of the base features to modify in the loop
        sim_features = base_features.copy()

        for price in np.arange(price_range[0], price_range[1] + increment, increment):
            # Set the price for the current iteration
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
