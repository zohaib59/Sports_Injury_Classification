# Athlete Injury Risk Prediction Model

## 🎯 Objective
This project builds a real-world machine learning pipeline for predicting short-term injury risk in professional athletes using tabular data.
Classify whether an athlete is at risk of injury within the next 10 days using physiological metrics, training workload, recovery markers, and historical injury data.
Predicting Injuries Before They Happen — with Machine Learning
In elite sports, even a single injury can derail performance, cost sponsorships, or end careers.


## Dataset
- Format: CSV
- Records: 10,000 (balanced Yes/No)
- Target: `injury_next_10_days`
- Features: 22 columns including:
  - `heart_rate_resting`, `body_temperature`, `hydration_level`
  - `muscle_soreness_rating`, `fatigue_self_report_score`
  - `previous_injuries_last_90_days`, `injury_type_last_occurred`
  - `match_frequency_last_month`, `player_position`, `BMI`, `team_physio_alert_flag`

## ML Pipeline Includes:
- Outlier detection via boxplots
- Label encoding for categorical data
- StandardScaler for neural/logistic models
- Train/Test split (80/20)
- 12 ML classifiers:
  - Tree-based, linear, boosting, ensemble, neural
- Evaluation:
  - Accuracy, Precision, Recall, F1
  - Classification Report
  - Confusion Matrix

