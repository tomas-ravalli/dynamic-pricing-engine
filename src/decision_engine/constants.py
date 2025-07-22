# A sample of base features for a single data point.
# This is used by the simulation and optimization scripts
# to generate predictions for a hypothetical scenario.
SAMPLE_BASE_FEATURES = {
    # Match characteristics
    'days_until_match': 15,
    'opponent_tier': 'A',
    'weather_forecast': 'Sunny',

    # Contextual factors
    'competing_city_events': False,
    'flights_to_barcelona_index': 110,
    'google_trends_index': 85,
    'internal_search_trends': 3500,
    'web_visits': 80000,
    'web_conversion_rate': 0.045,
    'social_media_sentiment': 1.8,
    'competitor_avg_price': 155.0,

    # Zone-specific information
    'zone': 'Lateral',
    'zone_seats_availability': 10852,
}