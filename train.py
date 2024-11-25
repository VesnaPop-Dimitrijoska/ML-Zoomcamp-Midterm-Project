import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import joblib  # To save the model

# Set random state
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    cat_cols = list(df.select_dtypes('object').columns)
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].values)
    return df

def preprocess_data(df):
    """Split dataset into features and target."""
    X = df.drop(columns=['Delay'])  # Features
    y = df['Delay']                # Target
    return X, y

def train_best_model(X, y):
    """Train the Random Forest model with the best hyperparameters."""
    # Define bins for stratified splits
    bins = [15, 60, 120, 180, 240, 300, 360, 2462]
    y_binned = np.digitize(y, bins=bins, right=True)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE).split(X, y_binned)

    # Preprocessing pipeline
    cat_cols = ['UniqueCarrier', 'TailNum', 'Origin', 'Dest']
    num_cols = list(set(X.columns) - set(cat_cols))

    col_trans = ColumnTransformer(
        [('mms', MinMaxScaler(), num_cols)],
        remainder='drop'
    )

    # Best model pipeline
    pipeline = Pipeline(
        [
            ('col_trans', col_trans),
            ('reg', RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_estimators=100,
                min_samples_split=5,
                min_samples_leaf=1,
                max_depth=None,
                criterion='squared_error'
            ))
        ]
    )

    # Train the model
    pipeline.fit(X, y)
    return pipeline

if __name__ == "__main__":
    data_path = "AirlineDelay_CleanDataset.csv"
    output_model_path = "best_model.pkl"

    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df)

    # Train the model
    best_model = train_best_model(X, y)

    # Save the model to a file
    joblib.dump(best_model, output_model_path)
    print(f"Model saved to {output_model_path}")
