import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from XGBRegression import data_preprocessing


xtrain,ytrain,xtest,ytest = data_preprocessing()
print(ytrain.head())

# print(xtrain.isna().sum())
# nan_rows = xtrain[xtrain.isna().any(axis=1)]
# print(nan_rows)

print(len(xtrain), len(ytrain))
rf = RandomForestRegressor(random_state=15)

rf.fit(xtrain, ytrain)

rf_pred = rf.predict(xtest)
r2 = r2_score(ytest, rf_pred)
print(f"R-squared (R2): {r2}")
n = xtest.shape[0]
k = xtest.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R-squared: {adjusted_r2}")

rf_pred = rf.predict(xtest)
mae = mean_absolute_error(ytest, rf_pred)
print(f"Mean Absolute Error: {mae}")

rf_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, rf_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

print(f"Mean of 'price(USD)': {train_data['price(USD)'].mean()}")

#For testing only!!!!
# Create a dataframe with feature impact
# df_impact = pd.DataFrame({'feature': X.columns, 'impact': rf.feature_importances_})
# df_impact.sort_values(by='impact', ascending=False, inplace=True)
# print(df_impact)

pred_df = pd.DataFrame({
    'actual price': ytest,
    'random forest': rf_pred
})

print(pred_df.head(20))