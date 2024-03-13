import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from Common import data_preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

xtrain, ytrain, xtest, ytest = data_preprocessing()

threshold = ytrain.mean()
ytrain_binary = (ytrain > threshold).astype(int)
ytest_binary = (ytest > threshold).astype(int)

logreg = LogisticRegression(random_state=42)
logreg.fit(xtrain, ytrain_binary)

logreg_pred = logreg.predict(xtest)
r2 = r2_score(ytest_binary, logreg_pred)
print(f"R-squared (R2): {r2}")
n = xtest.shape[0]
k = xtest.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R-squared: {adjusted_r2}")

logreg_pred = logreg.predict(xtest)
mae = mean_absolute_error(ytest_binary, logreg_pred)
print(f"Mean Absolute Error: {mae}")

logreg_pred = logreg.predict(xtest)
mse = mean_squared_error(ytest_binary, logreg_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

df_impact = pd.DataFrame({'feature': xtrain.columns, 'coefficient': logreg.coef_[0]})
df_impact['abs_coefficient'] = np.abs(df_impact['coefficient'])
df_impact.sort_values(by='abs_coefficient', ascending=False, inplace=True)
print(df_impact)

pred_df = pd.DataFrame({
    'price from .csv': ytest,
    'logistic prediction': logreg_pred
})

print(pred_df.head(20))

# # # Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price from .csv', y='logistic prediction', data=pred_df)
plt.plot(pred_df['price from .csv'], pred_df['price from .csv'], color='red', linestyle='--', label='Dijagonala')
plt.title('Cena vs Logistic Predikcija')
plt.xlabel('Cena')
plt.ylabel('Logistic Predikcija')
plt.show()

# Bar plot for feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='coefficient', y='feature', data=df_impact)
plt.title('Značaj osobine u Logistic Modelu')
plt.xlabel('Uticaj')
plt.ylabel('Osobina')
plt.show()

#Residual Analysis
residuals = ytest - logreg_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Analiza ostataka')
plt.xlabel('Ostatak')
plt.ylabel('Učestalost')
plt.show()

# Scatter plot of residuals against predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=logreg_pred, y=residuals)
plt.title('Ostaci vs Predviđene cene')
plt.xlabel('Predviđene cene')
plt.ylabel('Ostaci')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()