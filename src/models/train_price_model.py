import pandas as pd
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import the updated, specific path variable
from config import SYNTHETIC_DATA_PATH, PRICE_ELASTICITY_MODEL_PATH

def train_price_model(df):
    logging.info("Starting PRICE ELASTICITY model and transformer training...")

    X = df.drop(columns=['ticket_price', 'match_id'])
    y = df['ticket_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    numerical_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', Ridge(alpha=1.0))])
    
    model_pipeline.fit(X_train, y_train)
    logging.info("Price model training complete.")

    r2 = model_pipeline.score(X_test, y_test)
    logging.info(f"Price model evaluation on test set: R-squared = {r2:.4f}")

    # Use the updated path variable to save the model
    output_path = Path(PRICE_ELASTICITY_MODEL_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, output_path)
    logging.info(f"Trained price model pipeline saved to {output_path}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    train_price_model(df)

if __name__ == '__main__':
    main()
