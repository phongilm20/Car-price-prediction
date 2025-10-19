# 🚗 Regression Project: Used Car Price Prediction

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-v1.4.1-orange.svg)](https://scikit-learn.org/stable/)

## 🌟 1. Project Overview

This is an academic Machine Learning project focused on building and deploying a regression model to predict the selling price of used cars. The project emphasizes the adoption of **AI Engineering** standards by utilizing a **Scikit-learn Pipeline** to ensure the model's consistency, reliability, and deployability.

* **Objective:** Predict the car's selling price based on factors such as manufacturing year, kilometers driven, brand, fuel type, etc.
* **Core Model:** Random Forest Regressor.
* **Technical Highlight:** Implementation of a robust **ML Pipeline** and proper handling of **Unseen Features** (`handle_unknown` in One-Hot Encoding).

***

## 🛠️ 2. Technology Stack

| Area | Tool/Library | Technical Purpose |
| :--- | :--- | :--- |
| **Data Science Core** | `Pandas`, `NumPy` | Data manipulation, cleaning, and numerical operations. |
| **ML Engineering** | `Scikit-learn` | Building the **`Pipeline`**, `ColumnTransformer` (Preprocessor), and the model. |
| **Deployment Prep** | `Joblib` / `Pickle` | **Model Serialization** (Saving the entire workflow). |
| **EDA/Visualization** | `Matplotlib`, `Seaborn` | Exploratory data analysis and feature relationship plotting. |

***

## 💡 3. Academic Methodology & Model Artifacts

### 3.1. Feature Engineering

The entire preprocessing logic is encapsulated within a `ColumnTransformer` to apply different transformations to different column types:

* **Categorical Data:** Handled using `OneHotEncoder` with the critical parameter `handle_unknown='ignore'`. **This definitively resolves the `ValueError` concerning missing/unseen feature names in new data.**
* **Numerical Data:** Standardized using `StandardScaler` to bring numerical features (`year`, `km_driven`) to a common scale, preventing model bias.

### 3.2. ML Pipeline Construction

**Leveraging `sklearn.pipeline.Pipeline`**

```python
# Core Technical Structure:
model_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(...)), # Consistent preprocessing step
    ('regressor', RandomForestRegressor())    # Regression model
])
```

### 3.3. Model Evaluation & Project Structure

The model is trained on the **$X_{train}$** set and evaluated on the **$X_{test}$** set based on the following academic metrics:

| Metric | Formula/Significance | Result (To be Updated) |
| :--- | :--- | :--- |
| **$R^2$ Score** | Coefficient of Determination (Model fit) | **[UPDATE]** |
| **MAE** | Mean Absolute Error (Average monetary error) | **[UPDATE]** |
| **RMSE** | Root Mean Squared Error (Emphasizes larger errors) | **[UPDATE]** |

*(\*\*Note:\*\* Please update the $R^2$, MAE, and RMSE values here once you complete your model training.)*

```
car-price-prediction/
├── data/
│   └── car_data.csv          # Raw dataset
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb  # Exploratory Analysis and Feature Engineering
│   └── 02_Model_Training.ipynb     # Training, Evaluation, and .pkl Export
├── artifacts/
│   └── model_pipeline.pkl    # 🔑 Saved model file (includes Preprocessor)
├── src/
│   └── train.py              # Formal training script 
└── requirements.txt          # List of necessary libraries
```

-----

## 🚀 5. Getting Started

### 5.1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction

# 2. Create and activate a virtual environment (AI Engineer best practice)
python -m venv venv
source venv/bin/activate 

# 3. Install required libraries
pip install -r requirements.txt
```

### 5.2. Model Training

Open and run the `02_Model_Training.ipynb` notebook to:

1. Load and preprocess the data.  
2. Train the `model_pipeline`.  
3. Save the trained model to the `artifacts/` folder.

-----

## ✅ 6. Deployment and Inference

To make a prediction with new data, you only need to load the single `model_pipeline.pkl` file and pass the **RAW DATA** to the `.predict()` function.

```python
import joblib
import pandas as pd

# 1. Load the complete Pipeline (Preprocessor + Model)
model = joblib.load('artifacts/model_pipeline.pkl')

# 2. New Data (MUST BE RAW DATA, not yet OHE/Scaled)
new_data = pd.DataFrame({
    'year': [2023], 
    'km_driven': [5000],
    'transmission': ['Manual'],
    'brand': ['Maruti'], 
    'owner': ['First Owner'],
    # ... ensure all other raw columns are present
})

# 3. Predict: The Pipeline automatically preprocesses this raw data
predicted_price = model.predict(new_data) 
print(f"Predicted price: {predicted_price[0]}")
```

-----
