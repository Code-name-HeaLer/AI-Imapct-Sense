import os
from sklearn.ensemble import RandomForestClassifier
from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR, IMAGES_DIR, RF_PARAMS
from src.utils import load_numpy, save_pickle
from src.evaluation import evaluate_model

def run_training():
    print("\n🌲 Starting Random Forest Training...")
    
    # 1. Load Data
    X_train = load_numpy(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    y_train = load_numpy(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    X_test = load_numpy(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = load_numpy(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    # 2. Initialize Model (Using params from config.py)
    model = RandomForestClassifier(**RF_PARAMS)
    
    # 3. Train
    model.fit(X_train, y_train)
    print("✅ Model trained.")
    
    # 4. Evaluate
    evaluate_model(model, X_test, y_test, "Random Forest", IMAGES_DIR)
    
    # 5. Save Model
    save_path = os.path.join(ARTIFACTS_DIR, 'random_forest_model.pkl')
    save_pickle(model, save_path)

if __name__ == "__main__":
    run_training()