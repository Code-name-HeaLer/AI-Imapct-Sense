import os

# --- PATHS ---
# Get the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "earthquake_data.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
IMAGES_DIR = os.path.join(ARTIFACTS_DIR, "images")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- DATA CONFIG ---
TARGET_COLUMN = "alert"
CLASS_MAPPING = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- HYPERPARAMETERS ---
# We store the "Best" params we found earlier here, 
# so the training scripts look clean.

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_leaf': 1,
    'bootstrap': True,
    'random_state': RANDOM_STATE
}

XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.2,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    'random_state': RANDOM_STATE
}