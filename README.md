# 🚗 Used Car Price Prediction

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-v1.4.1-orange.svg)](https://scikit-learn.org/stable/)

## 🌟 Project Overview

This academic Machine Learning project focuses on building a regression model to predict **used car prices**. The project adopts **AI engineering best practices** by using a **Scikit-learn Pipeline** to ensure consistent preprocessing, model reliability, and deployability.

**Key Points:**

- **Objective:** Predict a car's selling price based on features like manufacturing year, kilometers driven, brand, fuel type, etc.  
- **Core Model:** Random Forest Regressor  
- **Technical Highlight:** Full **ML Pipeline** with robust handling of **unseen categorical features** (`handle_unknown='ignore'` in One-Hot Encoding).

---

## 🛠️ Technology Stack

| Area | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Data Science** | `Pandas`, `NumPy` | Data manipulation, cleaning, numerical operations |
| **ML Engineering** | `Scikit-learn` | Building **Pipeline**, `ColumnTransformer`, model training |
| **Deployment** | `Joblib` / `Pickle` | Model serialization (save/load pipeline) |
| **Visualization / EDA** | `Matplotlib`, `Seaborn` | Exploratory data analysis, plotting relationships |

---

## 💡 Methodology & Model Artifacts

### 1. Feature Engineering

- **Categorical Features:** One-Hot Encoding with `handle_unknown='ignore'` to avoid errors with unseen categories.  
- **Numerical Features:** Standardized using `StandardScaler` to prevent scale bias.  

All preprocessing is encapsulated in a `ColumnTransformer` inside the pipeline.

### 2. ML Pipeline Construction

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

model_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(...)),  # Preprocessing logic
    ('regressor', RandomForestRegressor())     # Random Forest Regressor
])
3. Model Evaluation
Model is trained on X_train and evaluated on X_test. Metrics:

Metric	Description	Result (Update Later)
R² Score	Coefficient of determination	[UPDATE]
MAE	Mean Absolute Error	[UPDATE]
RMSE	Root Mean Squared Error	[UPDATE]

Note: Update metrics after training the model.

4. Project Structure
car-price-prediction/
├── data/
│   └── car_data.csv          # Raw dataset
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb  # EDA & feature engineering
│   └── 02_Model_Training.ipynb     # Training, evaluation, export
├── artifacts/
│   └── model_pipeline.pkl    # Saved pipeline (preprocessor + model)
├── src/
│   └── train.py              # Training script
└── requirements.txt          # Dependencies
🚀 Getting Started
1. Environment Setup
# Clone repository
git clone https://github.com/phongilm20/Car-price-prediction.git
cd Car-price-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
2. Model Training
Run 02_Model_Training.ipynb to:

Load and preprocess the dataset

Train model_pipeline

Save trained pipeline to artifacts/

✅ Deployment & Inference
Predicting new data is simple: load the pipeline and pass raw data.

import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load('artifacts/model_pipeline.pkl')

# Prepare new raw data
new_data = pd.DataFrame({
    'year': [2023],
    'km_driven': [5000],
    'transmission': ['Manual'],
    'brand': ['Maruti'],
    'owner': ['First Owner'],
    # ... include all other required columns
})

# Predict
predicted_price = model.predict(new_data)
print(f"Predicted price: {predicted_price[0]}")
