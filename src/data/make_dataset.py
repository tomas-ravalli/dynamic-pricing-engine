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

def generate_synthetic_data(num_matches=10, max_days=90):
    """
    Generates a high-fidelity synthetic dataset for football match ticket pricing.
    This version focuses on time-series depth, generating a full sales history
    for a smaller number of matches.

    Args:
        num_matches (int): The number of unique matches to generate.
        max_days (int): The number of days before a match to start generating data.

    Returns:
        pd.DataFrame: A pandas DataFrame with detailed synthetic data.
    """
    logging.info(f"Generating time-series dataset for {num_matches} matches over {max_days} days...")

    # --- Define realistic categories and base values ---
    zones = {
        'Gol Nord': {'capacity': 15000, 'base_price': 50, 'demand_factor': 0.9},
        'Gol Sud': {'capacity': 15000, 'base_price': 50, 'demand_factor': 0.9},
        'Lateral': {'capacity': 30000, 'base_price': 150, 'demand_factor': 1.0},
        'Tribuna': {'capacity': 25000, 'base_price': 250, 'demand_factor': 1.1},
        'VIP': {'capacity': 4000, 'base_price': 500, 'demand_factor': 0.8}
    }
    opponent_tiers = {'A++': 1.5, 'A': 1.2, 'B': 1.0, 'C': 0.8}
    weather_forecasts = ['Sunny', 'Windy', 'Rain']
    
    records = []

    for i in range(num_matches):
        match_id = 100 + i
        
        # --- Simulate static match-level factors (they don't change day-to-day) ---
        tier_name = np.random.choice(list(opponent_tiers.keys()), p=[0.1, 0.3, 0.4, 0.2])
        tier_multiplier = opponent_tiers[tier_name]
        
        if tier_name == 'A++': ea_strength = np.random.randint(88, 93)
        elif tier_name == 'A': ea_strength = np.random.randint(84, 89)
        elif tier_name == 'B': ea_strength = np.random.randint(80, 85)
        else: ea_strength = np.random.randint(75, 81)

        team_position = np.random.randint(1, 5)
        weekday_match = np.random.choice([True, False], p=[0.3, 0.7])
        top_player_injured = np.random.choice([True, False], p=[0.15, 0.85])
        league_winner_known = True if i > (num_matches * 0.9) and np.random.rand() > 0.5 else False
        holidays = np.random.choice([True, False], p=[0.1, 0.9])
        competing_city_events = np.random.choice([True, False], p=[0.2, 0.8])
        weather = np.random.choice(weather_forecasts, p=[0.7, 0.2, 0.1])

        excitement = tier_multiplier - (team_position - 1) * 0.05
        excitement -= 0.2 if top_player_injured and tier_name in ['A++', 'A'] else 0
        excitement -= 0.25 if league_winner_known and tier_name in ['B', 'C'] else 0
        excitement += 0.15 if holidays else 0
        excitement -= 0.1 if weekday_match else 0
        excitement = np.clip(excitement, 0.5, 2.0)

        # Initialize sales tracking for this match
        cumulative_sales = {zone_name: 0 for zone_name in zones.keys()}

        # --- Time-series loop: Iterate from 90 days out to match day ---
        for days_until_match in range(max_days, -1, -1):
            
            # --- Simulate dynamic signals based on excitement and time ---
            time_decay_factor = ((max_days - days_until_match) / max_days)**0.7
            base_buzz = excitement * time_decay_factor

            social_media_sentiment = np.clip(np.random.normal(0.2, 0.4) * base_buzz, -1.0, 1.0)
            google_trends_index = int(np.clip(np.random.randint(20, 60) * base_buzz, 20, 100))
            internal_search_trends = int(np.clip(np.random.randint(100, 1000) * base_buzz, 100, 5000))
            web_visits = int(np.clip(np.random.randint(5000, 20000) * base_buzz, 5000, 100000))
            web_conversion_rate = np.clip(np.random.normal(0.02, 0.01) * base_buzz, 0.005, 0.1)
            flights_to_barcelona_index = int(np.clip(np.random.randint(30, 80) * base_buzz, 20, 100))

            # --- Generate data for each zone ---
            for zone_name, zone_info in zones.items():
                
                # Simulate daily sales and update cumulative sales
                daily_demand_potential = (zone_info['capacity'] - cumulative_sales[zone_name]) * 0.02 * excitement * time_decay_factor
                daily_sales = int(np.clip(np.random.normal(daily_demand_potential, daily_demand_potential * 0.3), 0, None))
                cumulative_sales[zone_name] += daily_sales
                cumulative_sales[zone_name] = min(cumulative_sales[zone_name], zone_info['capacity'])

                zone_seats_availability = zone_info['capacity'] - cumulative_sales[zone_name]
                ticket_availability_pct = round(zone_seats_availability / zone_info['capacity'], 4)

                # Price evolves based on availability and excitement
                price_multiplier = excitement + (1 - ticket_availability_pct)**2
                ticket_price = round(zone_info['base_price'] * price_multiplier * np.random.normal(1.0, 0.05), 2)
                competitor_avg_price = round(ticket_price * np.random.uniform(0.8, 1.2), 2)

                records.append({
                    'match_id': match_id,
                    'seat_zone': zone_name,
                    'opponent_tier': tier_name,
                    'ea_opponent_strength': ea_strength,
                    'team_position': team_position,
                    'weekday_match': weekday_match,
                    'top_player_injured': top_player_injured,
                    'league_winner_known': league_winner_known,
                    'holidays': holidays,
                    'weather_forecast': weather,
                    'competing_city_events': competing_city_events,
                    'days_until_match': days_until_match,
                    'flights_to_barcelona_index': flights_to_barcelona_index,
                    'google_trends_index': google_trends_index,
                    'internal_search_trends': internal_search_trends,
                    'web_visits': web_visits,
                    'web_conversion_rate': round(web_conversion_rate, 4),
                    'social_media_sentiment': round(social_media_sentiment, 4),
                    'zone_historical_sales': cumulative_sales[zone_name], # This is now cumulative sales to date
                    'zone_seats_availability': zone_seats_availability,
                    'ticket_availability_pct': ticket_availability_pct,
                    'competitor_avg_price': competitor_avg_price,
                    'ticket_price': ticket_price
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
    
    synthetic_df = generate_synthetic_data(num_matches=10, max_days=90)
    synthetic_df.to_csv(output_path, index=False)
    
    logging.info(f"Synthetic data saved to {output_path}")

if __name__ == '__main__':
    main()