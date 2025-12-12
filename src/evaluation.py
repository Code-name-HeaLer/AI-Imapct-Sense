import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import save_plot

def evaluate_model(model, X_test, y_test, model_name, output_dir):
    """
    Calculates metrics and saves a confusion matrix plot.
    Returns the accuracy score.
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # 1. Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Green', 'Yellow', 'Orange', 'Red'])
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n")
    print(report)
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Green', 'Yellow', 'Orange', 'Red'],
                yticklabels=['Green', 'Yellow', 'Orange', 'Red'], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}\nAccuracy: {acc:.2%}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Save Plot
    save_plot(fig, f"cm_{model_name.replace(' ', '_')}.png", output_dir)
    
    return acc