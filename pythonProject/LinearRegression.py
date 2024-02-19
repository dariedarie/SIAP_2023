import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Common import data_preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

xtrain,ytrain,xtest,ytest = data_preprocessing()

linear = LinearRegression()
linear.fit(xtrain, ytrain)

linear_pred = linear.predict(xtest)
r2 = r2_score(ytest, linear_pred)
print(f"R-squared (R2): {r2}")
n = xtest.shape[0]
k = xtest.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R-squared: {adjusted_r2}")

linear_pred = linear.predict(xtest)
mae = mean_absolute_error(ytest, linear_pred)
print(f"Mean Absolute Error: {mae}")

linear_pred = linear.predict(xtest)
mse = mean_squared_error(ytest, linear_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

pred_df = pd.DataFrame({
    'price from .csv': ytest,
    'linear prediction': linear_pred
})

print(pred_df.head(20))