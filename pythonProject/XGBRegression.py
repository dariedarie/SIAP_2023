import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Common import data_preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

xtrain,ytrain,xtest,ytest = data_preprocessing()

param_grid = {'colsample_bytree': [1.0], 'gamma': [1], 'learning_rate': [0.1], 'max_depth': [3], 'n_estimators': [100], 'reg_alpha': [0], 'reg_lambda': [0.1], 'subsample': [0.5]}
#
# xgb = XGBRegressor(random_state=42)
# xgb.fit(xtrain, ytrain)
#
# xgb_pred = xgb.predict(xtest)
# r2 = r2_score(ytest, xgb_pred)
# print(f"R-squared (R2): {r2}")
n = xtest.shape[0]
k = xtest.shape[1]
# adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
# print(f"Adjusted R-squared: {adjusted_r2}")
#
# xgb_pred = xgb.predict(xtest)
# mae = mean_absolute_error(ytest, xgb_pred)
# print(f"Mean Absolute Error: {mae}")
#
# xgb_pred = xgb.predict(xtest)
# mse = mean_squared_error(ytest, xgb_pred)
# print(f"Mean Squared Error: {mse}")
#
# rmse = np.sqrt(mse)
# print(f"Root Mean Squared Error: {rmse}")
#
# df_impact = pd.DataFrame({'feature': xtrain.columns, 'impact': xgb.feature_importances_})
# df_impact.sort_values(by='impact', ascending=False, inplace=True)
# print(df_impact)
#
# pred_df = pd.DataFrame({
#     'price from .csv': ytest,
#     'xgb prediction': xgb_pred
# })
#
# print(pred_df.head(25))

xgb = XGBRegressor(random_state=42)

scorer = make_scorer(r2_score)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=scorer, cv=5)


grid_search.fit(xtrain, ytrain)

print("Best hyperparameters:", grid_search.best_params_)

best_xgb_model = grid_search.best_estimator_

xgb_pred = best_xgb_model.predict(xtest)

r2 = r2_score(ytest, xgb_pred)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
mae = mean_absolute_error(ytest, xgb_pred)
mse = mean_squared_error(ytest, xgb_pred)
rmse = np.sqrt(mse)

print(f"R-squared (R2): {r2}")
print(f"Adjusted R-squared: {adjusted_r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

df_impact = pd.DataFrame({'feature': xtrain.columns, 'impact': best_xgb_model.feature_importances_})
df_impact.sort_values(by='impact', ascending=False, inplace=True)
print(df_impact)

pred_df = pd.DataFrame({
    'price from .csv': ytest,
    'xgb prediction': xgb_pred
})
print(pred_df.head(25))

import matplotlib.pyplot as plt
import seaborn as sns

# Residual Analysis
residuals = ytest - xgb_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=ytest, y=residuals)
plt.title('Analiza ostataka')
plt.xlabel('Cene')
plt.ylabel('Ostaci')
plt.show()

# Feature Importance Analysis
plt.figure(figsize=(12, 8))
sns.barplot(x='impact', y='feature', data=df_impact)
plt.title('Značaj osobine u XGBoost Modelu')
plt.xlabel('Uticaj')
plt.ylabel('Osobina')
plt.show()
