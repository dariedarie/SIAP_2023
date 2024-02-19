import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Common import data_preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

xtrain,ytrain,xtest,ytest = data_preprocessing()

rf = RandomForestRegressor(random_state=42)
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

df_impact = pd.DataFrame({'feature': xtrain.columns, 'impact': rf.feature_importances_})
df_impact.sort_values(by='impact', ascending=False, inplace=True)
print(df_impact)

pred_df = pd.DataFrame({
    'price from .csv': ytest,
    'random forest prediction': rf_pred
})

print(pred_df.head(20))