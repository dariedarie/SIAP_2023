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
import seaborn as sns
import matplotlib.pyplot as plt

xtrain,ytrain,xtest,ytest = data_preprocessing()

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

scorer = make_scorer(r2_score)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, cv=5)

grid_search.fit(xtrain, ytrain)

print("Best hyperparameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

#rf.fit(xtrain, ytrain)

rf_pred = best_rf_model.predict(xtest)
r2 = r2_score(ytest, rf_pred)
print(f"R-squared (R2): {r2}")
n = xtest.shape[0]
k = xtest.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R-squared: {adjusted_r2}")

rf_pred = best_rf_model.predict(xtest)
mae = mean_absolute_error(ytest, rf_pred)
print(f"Mean Absolute Error: {mae}")

rf_pred = best_rf_model.predict(xtest)
mse = mean_squared_error(ytest, rf_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

df_impact = pd.DataFrame({'feature': xtrain.columns, 'impact': best_rf_model.feature_importances_})
df_impact.sort_values(by='impact', ascending=False, inplace=True)
print(df_impact)

pred_df = pd.DataFrame({
    'price from .csv': ytest,
    'random forest prediction': rf_pred
})

print(pred_df.head(20))

# # Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price from .csv', y='random forest prediction', data=pred_df)
plt.plot(pred_df['price from .csv'], pred_df['price from .csv'], color='red', linestyle='--', label='Dijagonala')
plt.title('Cena vs Random Forest Predikcija')
plt.xlabel('Cena')
plt.ylabel('Random Forest Predikcija')
plt.show()

# Bar plot for feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='impact', y='feature', data=df_impact)
plt.title('Značaj osobine u Random Forest Modelu')
plt.xlabel('Uticaj')
plt.ylabel('Osobina')
plt.show()

#Residual Analysis
residuals = ytest - rf_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Analiza ostataka')
plt.xlabel('Ostatak')
plt.ylabel('Učestalost')
plt.show()

# Scatter plot of residuals against predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rf_pred, y=residuals)
plt.title('Ostaci vs Predviđene cene')
plt.xlabel('Predviđene cene')
plt.ylabel('Ostaci')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()