import pandas as pd
import logging
from pathlib import Path
import joblib

# Import the specific path variable for the demand model
from config import DEMAND_FORECAST_MODEL_PATH

def make_demand_prediction(input_data):
    """
    Loads the trained demand forecast pipeline and makes a prediction on new data.
    """
    logging.info("Starting DEMAND prediction...")
    
    # Load the demand model pipeline
    model_pipeline_path = Path(DEMAND_FORECAST_MODEL_PATH)
    if not model_pipeline_path.exists():
        logging.error(f"Demand model artifact not found at {model_pipeline_path}. Please run train_demand_model.py first.")
        return

    model_pipeline = joblib.load(model_pipeline_path)
    logging.info("Demand model pipeline loaded successfully.")

    # Convert single row of input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model_pipeline.predict(input_df)
    
    # Format the output as an integer
    predicted_sales = int(prediction[0])
    logging.info(f"Prediction complete. Predicted Ticket Sales: {predicted_sales:,}")
    return predicted_sales

def main():
    """
    Main function to demonstrate making a prediction on a sample data point.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sample_data = {
        'ticket_price': 195.00,
        'zone': 'Lateral',
        'days_until_match': 10,
        'opponent_tier': 'A',
        'weather_forecast': 'Sunny',
        'competing_city_events': False,
        'flights_to_barcelona_index': 85,
        'google_trends_index': 75,
        'internal_search_trends': 3500,
        'web_visits': 60000,
        'web_conversion_rate': 0.04,
        'social_media_sentiment': 0.6,
        'competitor_avg_price': 220.50,
        'zone_seats_availability': 10000,
        'ticket_availability_pct': 33.3 # <-- ADD THIS NEW FEATURE
    }
    
    make_demand_prediction(sample_data)

if __name__ == '__main__':
    main()