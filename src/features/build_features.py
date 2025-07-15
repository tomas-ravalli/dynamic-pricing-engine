import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Import paths from our config file
from config import SYNTHETIC_DATA_PATH, PROCESSED_DATA_PATH

def build_features(df):
    """
    Takes the detailed synthetic data and engineers features for modeling.
    """
    logging.info("Starting feature engineering on the new dataset...")

    # --- Pre-processing ---
    # Convert boolean to integer
    df['competing_city_events'] = df['competing_city_events'].astype(int)

    # Separate feature types
    categorical_cols = ['zone', 'opponent_tier', 'weather_forecast']
    numerical_cols = [
        'days_until_match', 'flights_to_barcelona_index', 'google_trends_index',
        'internal_search_trends', 'web_visits', 'web_conversion_rate',
        'social_media_sentiment', 'competitor_avg_price', 'zone_historical_sales',
        'zone_seats_availability', 'competing_city_events'
    ]
    
    # Keep identifiers and target separate
    passthrough_cols = ['match_id', 'ticket_price']
    
    # --- Encoding & Scaling ---
    # One-Hot Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

    # Scale numerical features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)
    
    # Combine all parts back into a final DataFrame
    processed_df = pd.concat([df[passthrough_cols], scaled_df, encoded_df], axis=1)

    logging.info("Feature engineering complete.")
    return processed_df

def main():
    """
    Main function to load data, build features, and save the processed data.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the synthetic data
    input_path = Path(SYNTHETIC_DATA_PATH)
    if not input_path.exists():
        logging.error(f"Synthetic data file not found at {input_path}. Please run make_dataset.py first.")
        return
    df = pd.read_csv(input_path)
    
    # Build features
    processed_df = build_features(df)

    # Save the processed data
    output_path = Path(PROCESSED_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
