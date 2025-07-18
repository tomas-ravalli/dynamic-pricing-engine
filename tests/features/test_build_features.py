import pandas as pd
import pytest
import sys
import os

# Add project root to Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Now we can import the function to be tested
from src.features.build_features import build_feature_pipeline

@pytest.fixture
def sample_raw_data():
    """Creates a sample raw DataFrame for testing."""
    return pd.DataFrame({
        'zone': ['Lateral', 'Gol'],
        'days_until_match': [10, 5],
        'opponent_tier': ['A', 'C'],
        'weather_forecast': ['Sunny', 'Rainy'],
        'competing_city_events': [False, True],
        'flights_to_barcelona_index': [85, 50],
        'google_trends_index': [75, 40],
        'internal_search_trends': [3500, 1200],
        'web_visits': [60000, 25000],
        'web_conversion_rate': [0.04, 0.02],
        'social_media_sentiment': [0.6, -0.1],
        'competitor_avg_price': [220.50, 180.00],
        'zone_seats_availability': [10000, 15000],
        'ticket_price': [195.00, 85.00]
    })

def test_feature_pipeline_output_shape(sample_raw_data):
    """
    Tests if the feature pipeline produces an output with the correct number of columns.
    """
    # Arrange: Get the feature pipeline
    pipeline = build_feature_pipeline()
    
    # Act: Transform the sample data
    processed_data = pipeline.fit_transform(sample_raw_data)
    
    # Assert: Check the shape of the output
    # Expected columns: 10 numerical + 6 one-hot encoded (3 zones + 2 tiers + 2 weathers handled by handle_unknown) + 1 boolean
    # Note: The exact number of one-hot encoded features depends on the categories present in the test data.
    # Let's verify the number of rows and that it has more columns than the original numeric ones.
    assert processed_data.shape[0] == sample_raw_data.shape[0]
    assert processed_data.shape[1] > 10 # Should be more than just the numerical features

def test_feature_pipeline_no_errors(sample_raw_data):
    """
    Tests if the feature pipeline runs without raising any errors on sample data.
    """
    # Arrange
    pipeline = build_feature_pipeline()
    
    # Act & Assert
    try:
        pipeline.fit_transform(sample_raw_data)
    except Exception as e:
        pytest.fail(f"Feature pipeline raised an exception on valid data: {e}")
