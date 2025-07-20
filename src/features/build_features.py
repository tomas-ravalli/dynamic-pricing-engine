import pandas as pd
import logging
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_feature_pipeline():
    """
    Defines the feature engineering pipeline for the dataset.
    This includes scaling for numerical features and one-hot encoding for categorical features.
    It returns the pipeline and the lists of features used.
    """
    # Define numerical features to be scaled
    numerical_features = [
        'days_until_match',
        'flights_to_barcelona_index',
        'google_trends_index',
        'internal_search_trends',
        'web_visits',
        'web_conversion_rate',
        'social_media_sentiment',
        'competitor_avg_price',
        'zone_seats_availability',
        'ticket_price' # Include ticket_price for scaling
    ]

    # Define categorical features to be one-hot encoded
    categorical_features = [
        'zone',
        'opponent_tier',
        'weather_forecast'
    ]
    
    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns (like boolean flags)
    )
    
    return preprocessor, numerical_features, categorical_features

def main():
    """
    Main function to run the feature engineering pipeline.
    It loads the synthetic data, processes it, and saves the result.
    """
    
    logging.info("Starting feature engineering process...")
    
    # Load the raw synthetic dataset
    try:
        df = pd.read_csv(config.SYNTHETIC_DATA_PATH)
        logging.info(f"Successfully loaded data from {config.SYNTHETIC_DATA_PATH}")
    except FileNotFoundError:
        logging.error(f"Error: Raw data file not found at {config.SYNTHETIC_DATA_PATH}. Please run src.data.make_dataset.py first.")
        return

    # Define the target variable
    target_column = 'zone_historical_sales'
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        return

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Build and fit the feature engineering pipeline
    feature_pipeline, numerical_features, categorical_features = build_feature_pipeline()
    X_processed = feature_pipeline.fit_transform(X)
    logging.info("Successfully applied feature engineering pipeline.")

    # Get feature names after transformation for the new DataFrame
    # This is important for model interpretability and debugging
    num_features = numerical_features
    cat_features = feature_pipeline.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    # Get remainder columns if any (passthrough)
    processed_cols = X.columns.drop(numerical_features + categorical_features)
    remainder_cols = [col for col in processed_cols if col in X.columns]
    
    processed_feature_names = list(num_features) + list(cat_features) + remainder_cols

    # Create a new DataFrame with the processed features
    processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
    
    # Add the target variable back to the processed DataFrame
    processed_df[target_column] = y.values

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    
    # Save the processed data
    processed_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logging.info(f"Feature engineering complete. Processed data saved to {config.PROCESSED_DATA_PATH}")

    import joblib
    joblib.dump(feature_pipeline, config.FEATURE_PIPELINE_PATH)
    logging.info(f"Feature engineering pipeline saved to {config.FEATURE_PIPELINE_PATH}")

if __name__ == '__main__':
    main()