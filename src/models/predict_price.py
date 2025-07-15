import pandas as pd
import logging
from pathlib import Path
import joblib

# Import the correct, specific path variable from config
from config import PRICE_ELASTICITY_MODEL_PATH

def make_price_prediction(input_data):
    """
    Loads the trained price elasticity pipeline and makes a prediction on new data.
    """
    logging.info("Starting price prediction...")
    
    # Load the price model pipeline
    model_pipeline_path = Path(PRICE_ELASTICITY_MODEL_PATH)
    if not model_pipeline_path.exists():
        logging.error(f"Price model artifact not found at {model_pipeline_path}. Please run train_price_model.py first.")
        return

    model_pipeline = joblib.load(model_pipeline_path)
    logging.info("Price model pipeline loaded successfully.")

    # Convert single row of input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model_pipeline.predict(input_df)
    
    logging.info(f"Prediction complete. Predicted price: {prediction[0]:.2f} €")
    return prediction[0]

def main():
    """
    Main function to demonstrate making a prediction on a sample data point.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Sample data for which to predict a price
    sample_data = {
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
        'zone_historical_sales': 20000,
        'zone_seats_availability': 10000
    }
    
    make_price_prediction(sample_data)

if __name__ == '__main__':
    main()