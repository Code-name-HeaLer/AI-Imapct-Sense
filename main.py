import sys
from src.preprocessing import run_preprocessing
from src.training.train_logistic import run_training as train_lr
from src.training.train_decision_tree import run_training as train_dt
from src.training.train_random_forest import run_training as train_rf
from src.training.train_xgboost import run_training as train_xgb

def main():
    print("========================================")
    print("🌍 IMPACT SENSE - FULL TRAINING PIPELINE")
    print("========================================")
    
    # 1. Run Data Pipeline
    run_preprocessing()
    
    # 2. Run Baseline Models
    train_lr()
    train_dt()
    
    # 3. Run Advanced Models
    train_rf()
    train_xgb()
    
    print("\n========================================")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("1. Models saved in 'artifacts/'")
    print("2. Plots saved in 'artifacts/images/'")
    print("========================================")

if __name__ == "__main__":
    main()