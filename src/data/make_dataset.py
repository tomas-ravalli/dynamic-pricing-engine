# src/data/make_dataset.py

import pandas as pd
import numpy as np
import random
from pathlib import Path

# --- Configuration ---
NUM_MATCHES = 10
DAYS_IN_ADVANCE = 90
ZONES = {
    'VIP': {'capacity': 500, 'base_price': 250},
    'Lateral': {'capacity': 8000, 'base_price': 120},
    'Corner': {'capacity': 6000, 'base_price': 90},
    'Gol Nord': {'capacity': 9000, 'base_price': 75},
    'Gol Sud': {'capacity': 9000, 'base_price': 75}
}
WEATHER_FORECASTS = ['Sunny', 'Windy', 'Rain']
WEATHER_WEIGHTS = [0.70, 0.20, 0.10]
ZONE_SKIP_PROBABILITY = {
    'VIP': 0.05,
    'Lateral': 0.10,
    'Corner': 0.12,
    'Gol Nord': 0.15,
    'Gol Sud': 0.15,
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'data/03_synthetic'
OUTPUT_FILE = OUTPUT_DIR / 'synthetic_match_data.csv'


def generate_match_details(num_matches):
    """Creates base attributes for each match."""
    matches = []
    tiers = ['C', 'B', 'A', 'A++']
    for i in range(1, num_matches + 1):
        tier = random.choice(tiers)
        strength_map = {'C': (65, 75), 'B': (76, 82), 'A': (83, 88), 'A++': (89, 95)}
        strength = random.randint(*strength_map[tier])
        is_international = True if i > num_matches - 3 else False

        matches.append({
            'match_id': i,
            'opponent_tier': tier,
            'ea_opponent_strength': strength,
            'is_weekday': random.choice([True, False]), # RENAMED
            'is_international': is_international,
            'top_player_injured': random.choice([True, False]) if tier in ['A', 'A++'] else False,
            'league_winner_known': random.choice([True, False]),
        })
    return matches

def calculate_excitement_factor(match):
    """Calculates the core demand driver for a match."""
    base_excitement = {'C': 0.4, 'B': 0.6, 'A': 0.8, 'A++': 1.0}[match['opponent_tier']]

    if match['is_international']:
        base_excitement *= 1.5
    base_excitement *= (1 + (match['ea_opponent_strength'] - 80) / 100)
    if match['is_weekday']: # RENAMED
        base_excitement *= 0.75
    if match['top_player_injured']:
        base_excitement *= 0.8
    if match['league_winner_known'] and not match['is_international']:
        base_excitement *= 0.7

    return np.clip(base_excitement, 0.2, 2.0)


def generate_daily_data(match, excitement_factor):
    """Generates time-series data for a single match."""
    records = []
    for zone_name, zone_info in ZONES.items():
        if random.random() < ZONE_SKIP_PROBABILITY[zone_name]:
            continue

        available_seats = zone_info['capacity']
        for day in range(DAYS_IN_ADVANCE, -1, -1):
            time_urgency = 1 + 2.5 * np.exp(-day / 20)
            base_web_visits = int(excitement_factor * 20000 * time_urgency)
            web_visits = int(base_web_visits * random.uniform(0.8, 1.2))
            
            weather = random.choices(WEATHER_FORECASTS, weights=WEATHER_WEIGHTS, k=1)[0]
            weather_multiplier = 1.0
            if day < 7:
                if weather == 'Rain':
                    weather_multiplier = 0.65
                elif weather == 'Windy':
                    weather_multiplier = 0.85

            price_multiplier = 1 + (excitement_factor - 1) * 0.8
            ticket_price = round(zone_info['base_price'] * price_multiplier * random.uniform(0.95, 1.15), 2)
            sales_potential = (excitement_factor * time_urgency * weather_multiplier) / (1 + (ticket_price / zone_info['base_price']) / 10)
            sales_potential *= (1 + web_visits / 50000)
            zone_appeal = ZONES[zone_name]['capacity'] / 9000
            daily_sales = int(sales_potential * zone_appeal * random.uniform(0.7, 1.3) * 30)
            daily_sales = min(daily_sales, available_seats)
            ticket_availability_pct = max(0, available_seats / zone_info['capacity'])
            
            records.append({
                'match_id': match['match_id'],
                'days_until_match': day,
                'seat_zone': zone_name,
                'opponent_tier': match['opponent_tier'],
                'ea_opponent_strength': match['ea_opponent_strength'],
                'is_weekday': match['is_weekday'], # RENAMED
                'is_international': match['is_international'],
                'top_player_injured': match['top_player_injured'],
                'league_winner_known': match['league_winner_known'],
                'weather_forecast': weather,
                'ticket_price': ticket_price,
                'web_visits': web_visits,
                'social_media_sentiment': round(np.clip(excitement_factor * random.uniform(0.5, 1.2) - 0.2, -1, 1), 2),
                'google_trends_index': int(np.clip(excitement_factor * time_urgency * random.uniform(0.8, 1.2) * 50, 20, 100)),
                'zone_seats_availability': available_seats,
                'ticket_availability_pct': round(ticket_availability_pct, 4),
                'zone_historical_sales': daily_sales
            })
            available_seats -= daily_sales
            
    return records


def main():
    """Main function to generate and save the dataset."""
    print("Generating synthetic dataset with final column names...")
    
    all_records = []
    matches = generate_match_details(NUM_MATCHES)
    
    for match in matches:
        excitement_factor = calculate_excitement_factor(match)
        daily_records = generate_daily_data(match, excitement_factor)
        all_records.extend(daily_records)
        
    df = pd.DataFrame(all_records)
    
    # Add other placeholder columns and rename to match schema
    df['team_position'] = df.apply(lambda row: random.randint(1, 5) if row['days_until_match'] > 30 else random.randint(1, 3), axis=1)
    df['is_holiday'] = df['days_until_match'].apply(lambda x: x in [0, 1, 6, 7, 45, 46]) # RENAMED
    df['popular_concert_in_city'] = df['days_until_match'].apply(lambda x: x in [20, 21, 22]) # RENAMED
    df['flights_to_barcelona_index'] = df.apply(lambda row: int(np.clip(row['google_trends_index'] * random.uniform(0.8, 1.2), 20, 100)), axis=1)
    df['internal_search_trends'] = df['web_visits'].apply(lambda x: int(x / 10 * random.uniform(0.8, 1.2)))
    df['web_conversion_rate'] = df.apply(lambda row: np.clip(row['zone_historical_sales'] / (row['web_visits'] + 1e-6), 0, 0.1), axis=1)
    df['competitor_avg_price'] = df['ticket_price'].apply(lambda x: x * random.uniform(0.8, 1.2))

    # --- ADJUSTMENT: Use the specified column order ---
    final_column_order = [
        'match_id', 'days_until_match', 'seat_zone', 'zone_historical_sales', 'ticket_price',
        'ea_opponent_strength', 'web_visits', 'weather_forecast', 'is_weekday', 'is_international',
        'opponent_tier', 'top_player_injured', 'league_winner_known', 'team_position', 'is_holiday',
        'popular_concert_in_city', 'flights_to_barcelona_index', 'google_trends_index',
        'internal_search_trends', 'social_media_sentiment', 'web_conversion_rate',
        'zone_seats_availability', 'ticket_availability_pct', 'competitor_avg_price'
    ]
    
    df = df[final_column_order]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Dataset successfully generated at: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("Column names have been updated to the final specified version.")

if __name__ == '__main__':
    main()