import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add the project root to the Python path.
# This allows the script to be run from anywhere, not just the project root.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import path from our config file
from config import SYNTHETIC_DATA_PATH

def generate_synthetic_data(num_matches=50):
    """
    Generates a high-fidelity synthetic dataset for football match ticket pricing.
    Each row represents a specific seating zone for a given match.

    Args:
        num_matches (int): The number of unique matches to generate data for.

    Returns:
        pd.DataFrame: A pandas DataFrame with detailed synthetic data.
    """
    logging.info(f"Generating detailed synthetic dataset for {num_matches} matches...")

    # --- Define realistic categories and base values ---
    zones = {
        'Gol Nord': {'capacity': 15000, 'base_price': 70},
        'Gol Sud': {'capacity': 15000, 'base_price': 70},
        'Lateral': {'capacity': 30000, 'base_price': 150},
        'Tribuna': {'capacity': 25000, 'base_price': 250},
        'VIP': {'capacity': 4000, 'base_price': 500}
    }
    opponent_tiers = {'A++': 1.5, 'A': 1.2, 'B': 1.0, 'C': 0.8}
    weather_forecasts = ['Sunny', 'Cloudy', 'Rain']
    
    records = []

    for i in range(num_matches):
        match_id = 100 + i
        
        # --- Simulate match-level base factors ---
        tier_name = np.random.choice(list(opponent_tiers.keys()), p=[0.1, 0.3, 0.4, 0.2])
        tier_multiplier = opponent_tiers[tier_name]
        
        days_until_match = np.random.randint(1, 90)
        competing_city_events = np.random.choice([True, False], p=[0.2, 0.8])
        weather = np.random.choice(weather_forecasts, p=[0.6, 0.3, 0.1])

        # --- Simulate external signals based on match importance ---
        base_buzz = tier_multiplier + (90 - days_until_match) / 90.0
        
        social_media_sentiment = np.random.uniform(-0.5, 1.0) * base_buzz
        google_trends_index = np.clip(np.random.randint(40, 100) * base_buzz, 20, 100)
        internal_search_trends = np.clip(np.random.randint(500, 2000) * base_buzz, 100, 5000)
        web_visits = np.clip(np.random.randint(10000, 50000) * base_buzz, 5000, 100000)
        web_conversion_rate = np.clip(np.random.uniform(0.01, 0.05) * base_buzz, 0.005, 0.1)
        flights_to_barcelona_index = np.clip(np.random.randint(50, 100) * base_buzz, 30, 100)
        competitor_base_price = np.random.uniform(1.1, 1.5) * base_buzz
        
        # --- Generate data for each zone in the match ---
        for zone_name, zone_info in zones.items():
            # Simulate zone-specific sales
            zone_sales_factor = (90 - days_until_match) / 90.0 * tier_multiplier
            zone_historical_sales = int(np.clip(zone_info['capacity'] * zone_sales_factor * np.random.uniform(0.5, 1.0), 0, zone_info['capacity']))
            zone_seats_availability = zone_info['capacity'] - zone_historical_sales

            # Simulate final ticket price (our target variable)
            final_price = zone_info['base_price'] * (0.4 + competitor_base_price * 0.6) * np.random.uniform(0.85, 1.15)
            
            records.append({
                'match_id': match_id,
                'zone': zone_name,
                'days_until_match': days_until_match,
                'opponent_tier': tier_name,
                'weather_forecast': weather,
                'competing_city_events': competing_city_events,
                'flights_to_barcelona_index': int(flights_to_barcelona_index),
                'google_trends_index': int(google_trends_index),
                'internal_search_trends': int(internal_search_trends),
                'web_visits': int(web_visits),
                'web_conversion_rate': round(web_conversion_rate, 4),
                'social_media_sentiment': round(social_media_sentiment, 4),
                'competitor_avg_price': round(zone_info['base_price'] * competitor_base_price * np.random.uniform(0.9, 1.3), 2),  # Competitor price with more variance
                'zone_historical_sales': zone_historical_sales,
                'zone_seats_availability': zone_seats_availability,
                'ticket_price': round(final_price, 2)
            })

    df = pd.DataFrame(records)
    logging.info(f"Successfully generated DataFrame with {len(df)} rows.")
    return df

def main():
    """
    Main function to generate and save the synthetic dataset.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    output_path = Path(SYNTHETIC_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    synthetic_df = generate_synthetic_data()
    synthetic_df.to_csv(output_path, index=False)
    
    logging.info(f"Synthetic data saved to {output_path}")

if __name__ == '__main__':
    main()
