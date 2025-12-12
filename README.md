# 🌍 ImpactSense: Earthquake Impact Prediction

ImpactSense is an end-to-end Machine Learning system capable of predicting earthquake alert levels (Green, Yellow, Orange, Red) based on seismic data. It leverages advanced ensemble models to analyze the relationship between magnitude, depth, and intensity.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-green)
![App](https://img.shields.io/badge/Framework-Streamlit-red)

## 📊 Project Results

| Model               | Accuracy   | F1-Score (Macro) | Status          |
| :------------------ | :--------- | :--------------- | :-------------- |
| **Random Forest**   | **90.38%** | **0.90**         | 🏆 **Champion** |
| XGBoost             | 89.62%     | 0.90             | Runner-up       |
| Decision Tree       | 84.23%     | 0.84             | Baseline 2      |
| Logistic Regression | 63.85%     | 0.64             | Baseline 1      |

## 🏗️ Project Structure

This project follows a professional modular architecture:

```text
ImpactSense-Portfolio/
├── artifacts/          # Saved models (.pkl) and scaler
├── data/               # Raw and processed datasets
├── src/                # Source code package
│   ├── training/       # Training scripts for specific models
│   ├── preprocessing.py# Feature engineering pipeline
│   ├── evaluation.py   # Metric calculation & plotting
│   └── config.py       # Central configuration
├── app/                # Streamlit UI code
└── main.py             # Master execution script
```

## 🚀 How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Code-name-HeaLer/AI-Imapct-Sense
   cd AI-Imapct-Sense
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (Full Pipeline):**

   ```bash
   python main.py
   ```

4. **Launch the User Interface:**
   ```bash
   streamlit run app/main_ui.py
   ```

## 🧠 Key Features

- **Physics-Based Features:** Engineered `impact_factor` and `energy_release` based on seismic formulas.
- **Ensemble Learning:** Utilizes Random Forest and XGBoost for robust classification.
- **Explainability:** Features ranked by importance (Significance Score & MMI were top drivers).

## 👨‍💻 Author

Built as a practice ML Project by **Swagat Prasad Nanda**.

[![Gmail](https://img.shields.io/badge/Gmail-Email%20Me-red?style=flat-square&logo=gmail)](mailto:swagatprasad3344@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/swagat-prasad-nanda/)

---

©️ Code Name HeaLer | All rights reserved
