import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import (DATA_PATH, PROCESSED_DATA_DIR, ARTIFACTS_DIR, 
                        CLASS_MAPPING, TEST_SIZE, RANDOM_STATE)
from src.utils import save_pickle, save_numpy

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df

def feature_engineering(df):
    print("⚙️ Engineering Features...")
    # 1. Seismic Energy
    df['energy_release'] = 10 ** (1.5 * df['magnitude'])
    
    # 2. Impact Factor (Mag / log(Depth))
    # Add epsilon to depth to avoid div by zero if depth=0
    df['impact_factor'] = df['magnitude'] / np.log1p(df['depth'])
    
    # 3. Shallow Flag
    df['is_shallow'] = (df['depth'] < 70).astype(int)
    
    return df

def run_preprocessing():
    print("--- Starting Preprocessing Pipeline ---")
    
    # 1. Load
    df = load_data()
    
    # 2. Feature Engineering
    df = feature_engineering(df)
    
    # 3. Encoding Target
    df['alert_encoded'] = df['alert'].map(CLASS_MAPPING)
    
    # 4. Split X/y
    X = df.drop(['alert', 'alert_encoded'], axis=1)
    y = df['alert_encoded']
    
    # Save feature names for later use (UI)
    feature_names = X.columns.tolist()
    save_pickle(feature_names, os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler (Crucial for the App!)
    save_pickle(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    
    # 7. Save Processed Data
    save_numpy(X_train_scaled, os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    save_numpy(X_test_scaled, os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    save_numpy(y_train.to_numpy(), os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    save_numpy(y_test.to_numpy(), os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    print("--- Preprocessing Complete ---")

if __name__ == "__main__":
    run_preprocessing()