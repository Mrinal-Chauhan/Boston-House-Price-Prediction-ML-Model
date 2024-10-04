# House Price Prediction Model

## Overview
This project is a regression model built to predict house prices based on various features such as crime rate, property tax, accessibility to highways, and others. The model uses a dataset of 506 samples with 13 input features to predict the median value of owner-occupied homes.

## Features
The dataset includes the following features:
- **CRIM**: Crime rate by town
- **ZN**: Proportion of residential land zoned for large lots
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built before 1940
- **DIS**: Weighted distances to employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: Proportion of Black residents
- **LSTAT**: Percentage of lower status of the population
- **MEDV**: Median value of owner-occupied homes (target variable)

## Steps Involved

### 1. Data Preprocessing
We cleaned and preprocessed the data by:
- Handling missing values (if any)
- Identifying and treating outliers
- Creating new features to handle multicollinearity between highly correlated variables such as `RAD` and `TAX`

### 2. Feature Scaling
To ensure that all features contribute equally to the model, we scaled the data using normalization/standardization techniques. This step improves model convergence and accuracy by preventing features with larger values from dominating the model.

### 3. Model Building
We built several regression models, including:
- **Linear Regression**
- **Random Forest**
- **XGBoost**

Each model was trained and evaluated to predict house prices, with the **XGBoost** model providing the best results.

### 4. Evaluation
The model's performance was evaluated using the following metrics:
- **R-squared (R²)**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

Among the tested models, XGBoost performed the best, offering the highest accuracy and lowest error rates.

## Dependencies
To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Load the dataset in the Jupyter notebook or script.
3. Run the notebook to see the data preprocessing steps, model building, and evaluation results.
4. Modify and experiment with different algorithms and parameters to improve the model.

## Conclusion
This project demonstrates the full pipeline for building a machine learning regression model, from data cleaning and preprocessing to model evaluation. After testing multiple models, XGBoost provided the most accurate predictions, highlighting its robustness in handling complex datasets.
