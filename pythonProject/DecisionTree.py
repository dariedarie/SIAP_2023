import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Common import data_preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree

xtrain,ytrain,xtest,ytest = data_preprocessing()

param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(xtrain, ytrain)

decision_tree_pred = decision_tree.predict(xtest)
r2 = r2_score(ytest, decision_tree_pred)
print(f"R-squared (R2): {r2}")
n = xtest.shape[0]
k = xtest.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R-squared: {adjusted_r2}")

decision_tree_pred = decision_tree.predict(xtest)
mae = mean_absolute_error(ytest, decision_tree_pred)
print(f"Mean Absolute Error: {mae}")

decision_tree_pred = decision_tree.predict(xtest)
mse = mean_squared_error(ytest, decision_tree_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

df_impact = pd.DataFrame({'feature': xtrain.columns, 'impact': decision_tree.feature_importances_})
df_impact.sort_values(by='impact', ascending=False, inplace=True)
print(df_impact)

pred_df = pd.DataFrame({
    'price from .csv': ytest,
    'decision_tree prediction': decision_tree_pred
})

print(pred_df.head(20))

feature_importances = decision_tree.feature_importances_
feature_names = xtrain.columns

df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

residuals = ytest - decision_tree_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(ytest, residuals, color='blue')
plt.title('Decision Tree Residuals')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # Add a horizontal line at y=0
plt.show()

# Plot the feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=df_importance)
plt.title('Decision Tree Feature Importance')
plt.show()