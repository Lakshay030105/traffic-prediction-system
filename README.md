# 🚗 Metro Interstate Traffic Demand Prediction
**INNOVEXA CATALYST Machine Learning Internship Project**

## 📌 Project Overview
This project predicts hourly traffic volume on a metro interstate using a custom-weighted machine learning ensemble. By cleaning historical data and engineering time-cyclical features, the final model successfully captures complex traffic patterns driven by human routine and environmental factors.

---

## 📊 Dataset Documentation
The project utilizes the **Metro Interstate Traffic Volume** dataset (48,000+ records). 
* **Target Variable:** `traffic_volume` - Numeric count of hourly vehicles on the interstate.
* **Features:**
  * `temp`: Average temperature in Kelvin.
  * `clouds_all`: Percentage of cloud cover.
  * `weather_main`: Categorical description of the current weather (Clear, Clouds, Rain, Snow, etc.).
  * `holiday`: US National holidays.
  * `date_time`: Hour and date of the recorded data.

---

## 🚀 Project Pipeline & Methodology

### 1. Data Cleaning
* **Missing Values:** Identified and removed localized null values within the `holiday` column.
* **Deduplication:** Scanned and removed 17 exact duplicate records to prevent artificial data weighting.

### 2. Exploratory Data Analysis (EDA)
* **Distributions:** Histograms mapped the target variable, identifying normal flow versus extreme congestion events.
* **Correlations:** Heatmaps established initial bivariate relationships, highlighting the dominance of time over weather in predicting traffic volume.

### 3. Feature Engineering
Target leakage was strictly avoided (no direct traffic density scores were derived from the target).
* **Cyclic Time Encoding:** Transformed hours into `hour_sin` and `hour_cos` using trigonometric functions, allowing models to understand the 24-hour continuous loop.
* **Routine Indicators:** Created specific `weekend_flag` and `rush_hour` binary flags.

### 4. Feature Selection
Redundant features were aggressively pruned using Correlation Heatmaps and Random Forest Feature Importance testing.
* **Dropped:** `snow_1h`, `holiday_new`, and `rush_hour`.
* **Result:** Reduced structural complexity while maintaining an exceptionally high R2 score.

---

## 🧠 Model Architecture & Evaluation
The project evaluated multiple algorithms before finalizing a custom-weighted ensemble model as per the project blueprint. 

| Model | R2 Score | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **Random Forest (Baseline)** | 0.9544 | 426.89 | 239.37 |
| **LightGBM (Optimized)** | 0.9557 | 420.61 | 252.96 |
| **XGBoost (Optimized)** | 0.9576 | 411.53 | 246.68 |
| **Final Ensemble (55% LGBM + 45% XGB)**| **0.9576** | **411.43** | **245.59** |

**Conclusion:** The weighted ensemble achieved the lowest overall error rates. XGBoost captured the core temporal patterns, while the histogram-based LightGBM aggressively minimized residuals on environmental edge cases.

---

## 💻 How to Run the Streamlit Web App

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-folder>