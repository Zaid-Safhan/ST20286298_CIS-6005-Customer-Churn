# Customer Churn Prediction

The aim of the project is to predict whether a customer is likely to leave a service. The dataset was taken from the Kaggle competition:

**Predict Customer Churn – Playground Series, Season 6 Episode 3**

https://www.kaggle.com/competitions/playground-series-s6e3

The following models were tested:

- Logistic Regression
- Weighted Logistic Regression
- Random Forest
- XGBoost
- Multi-Layer Perceptron

XGBoost gave the best overall result.

Main results:

- Validation ROC-AUC: 0.9156
- Kaggle public score: 0.91240
- Kaggle private score: 0.91372
- Threshold used in the application: 0.35

## Folder structure

```text
ST20286298_CIS-6005_Customer_Churn/
├── app/
│   ├── app.py
│   └── model/
│       ├── final_xgboost_pipeline.joblib
│       └── model_information.json
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── Notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_preprocessing_setup.ipynb
│   ├── 04_logistic_regression_baseline.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_xgboost.ipynb
│   └── 07_mlp_neural_network.ipynb
├── report/
├── screenshots
├── README.md
└── requirements.txt
