# Example usage
if __name__ == '__main__':
    # Create a sample feature set for a hypothetical match, based on the demand model's needs.
    # This dictionary contains all features EXCEPT 'ticket_price'.
    sample_base_features = {
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
        'ticket_availability_pct': 33.3
    }
    
    # Convert to DataFrame for the engines
    features_df = pd.DataFrame([sample_base_features])

    # --- Test Simulation Engine ---
    sim_engine = SimulationEngine()
    if sim_engine.model:
        # Simulate the outcome for a price of €195.00
        simulation_result = sim_engine.run_simulation(price=195.00, base_features=features_df)
        print("\n--- Simulation Result ---")
        print(simulation_result)

    # --- Test Optimization Engine ---
    opt_engine = OptimizationEngine()
    if opt_engine.model:
        # Define business constraints for the price search
        price_constraints = (75, 350) # Min price €75, Max price €350
        optimization_result = opt_engine.find_optimal_price(
            base_features=features_df,
            price_range=price_constraints,
            increment=5 # Check price in steps of €5
        )
        print("\n--- Optimization Result ---")
        print(optimization_result)
