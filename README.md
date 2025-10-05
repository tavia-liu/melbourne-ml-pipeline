# Melbourne Housing ML 

End-to-end supervised machine learning pipeline predicting Melbourne housing prices using Python, scikit-learn, and feature engineering.

---

## Overview
This project builds a **full ML regression pipeline** to predict housing prices in Melbourne using the Kaggle [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot).  
It covers the complete data science workflow — from data profiling and preprocessing to feature engineering, model comparison, and evaluation.

---

## Features
- **Data Preprocessing:** Imputation, one-hot encoding, and scaling using scikit-learn `Pipeline` and `ColumnTransformer`
- **Feature Engineering:** Derived features like `relative_size`, `accessibility`, and `binary_type`
- **Feature Selection:** Model-based selection using Random Forest importances (top 30%)
- **Models Implemented:**  
  - Linear Regression  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree Regressor  
  - Random Forest Regressor  
- **Model Evaluation:** R² and RMSE on both baseline and feature-engineered models
- **Cross-Validation:** 8-fold CV with `GridSearchCV` for hyperparameter tuning

---

## Key Results
| Model | R² (Baseline) | R² (Engineered) | RMSE (Baseline) | RMSE (Engineered) |
|-------|----------------|-----------------|-----------------|------------------|
| Linear Regression | 0.68 | 0.66 | 353,770 | 370,000 |
| KNN Regressor | 0.75 | 0.78 | 314,186 | 285,000 |
| Decision Tree | 0.73 | 0.76 | 325,000 | 295,000 |
| Random Forest | **0.81** | **0.80** | **270,000** | **275,000** |

**➡️ Best Model:** Random Forest Regressor — balanced accuracy, robustness, and resistance to overfitting.

---

## Tech Stack
- **Language:** Python 3  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Tools:** Jupyter Notebook, GitHub, AWS (for testing and compute)  

---
