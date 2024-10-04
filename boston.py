import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('bostondata.csv', header=None)

header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = header



# Plotting all the columns to look at their distributions
for i in df.columns:

    plt.figure(figsize = (7, 4))

    sns.histplot(data = df, x = i, kde = True)

    plt.show()


# Plotting the heatmap of correlation between features
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()



# Scatterplot to visulaize the relationship between RAD and TAX

plt.figure(figsize = (5, 5))
sns.scatterplot(x = 'RAD', y = 'TAX', data = df)
plt.show()



# Scatterplot to visulaize the relationship between above mentioned parameters

# 1) RM and MEDV
plt.figure(figsize = (5, 5))
sns.scatterplot(x = 'RM', y = 'MEDV', data = df)
plt.show()

# 2) INDUS and TAX
plt.figure(figsize = (5, 5))
sns.scatterplot(x = 'INDUS', y = 'TAX', data = df)
plt.show()

# 3) NOX and INDUS
plt.figure(figsize = (5, 5))
sns.scatterplot(x = 'NOX', y = 'INDUS', data = df)
plt.show()

# 4) AGE and NOX
plt.figure(figsize = (5, 5))
sns.scatterplot(x = 'AGE', y = 'NOX', data = df)
plt.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# *Regression Model


## *Missing Data

# First of all removing data with house price missing (value to be predicted) (y parameter)
df.dropna(subset='MEDV',inplace=True)

# ZN column is supposed to have 0 have

# Removing data with more than 7 missing values (has less than half information which a row should have) (self chosen number)
df.dropna(thresh=7) # Minimum 7 Non NA values, else it will remove that row

# With rows having less than 8 missing values, using data interpolation
df.interpolate(method ='linear', limit_direction ='forward')

# Because the method is forward, the first NA value will remain untouched, so finally removing that single row
df.dropna()



# *Data Outliers*

columns_to_check = ['CRIM', 'ZN', 'INDUS', 'AGE', 'DIS', 'TAX', 'B', 'LSTAT']

for column in columns_to_check:
    # Calculate Q1 and Q3 for each column
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for extreme and mild outliers
    lower_bound_extreme = Q1 - 3 * IQR
    upper_bound_extreme = Q3 + 3 * IQR
    lower_bound_mild = Q1 - 1.5 * IQR
    upper_bound_mild = Q3 + 1.5 * IQR

    # Remove extreme outliers
    df = df[(df[column] >= lower_bound_extreme) & (df[column] <= upper_bound_extreme)]

    # Transform Mild outliers
    # Create a boolean mask for mild outliers
    mild_outliers_mask = (df[column] < lower_bound_mild) | (df[column] > upper_bound_mild)

    # Transform mild outliers using square root transformation
    df.loc[mild_outliers_mask, column] = np.sqrt(np.abs(df.loc[mild_outliers_mask, column]))


print(df.describe().T)


# Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns = header)
df.describe()


# Split the Dataset

X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


model1 = LinearRegression()
model1.fit(X_train, y_train)

print(model1.intercept_)



# *Evaluation

y_prediction = model1.predict(X_train)

print('R^2:',r2_score(y_train, y_prediction))
print('Adjusted R^2:',1 - (1-r2_score(y_train, y_prediction))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',mean_absolute_error(y_train, y_prediction))
print('MSE:',mean_squared_error(y_train, y_prediction))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_prediction)))


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_prediction)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

# Checking residuals
plt.scatter(y_prediction,y_train-y_prediction)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()



# Now predicting testing data with the model

y_test_pred = model1.predict(X_test)

# Model Evaluation
acc_linreg = r2_score(y_test, y_test_pred)
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mean_absolute_error(y_test, y_test_pred))
print('MSE:',mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_test_pred)))


#  xxxxxxxxxxxxxxxxxxxxx

# * Random Forest


reg = RandomForestRegressor()

reg.fit(X_train, y_train)

y_prediction = reg.predict(X_train)


# Model Evaluation
print('R^2:',r2_score(y_train, y_prediction))
print('Adjusted R^2:',1 - (1-r2_score(y_train, y_prediction))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',mean_absolute_error(y_train, y_prediction))
print('MSE:',mean_squared_error(y_train, y_prediction))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_prediction)))


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_prediction)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()



# Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_rf = r2_score(y_test, y_test_pred)
print('R^2:', acc_rf)
print('Adjusted R^2:',1 - (1-r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mean_absolute_error(y_test, y_test_pred))
print('MSE:',mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_test_pred)))


# xxxxxxxxxxxxxxxxxx

# * XGBoost Regressor 

#Create a XGBoost Regressor
reg = XGBRegressor()

# Train the model using the training sets
reg.fit(X_train, y_train)


# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',mean_absolute_error(y_train, y_pred))
print('MSE:',mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred)))


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


#Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_xgb = r2_score(y_test, y_test_pred)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mean_absolute_error(y_test, y_test_pred))
print('MSE:',mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_test_pred)))