import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('datasets/cleaned_all_phones.csv')

# Create a new column 'price_range' based on price ranges
p_list = []
for i in df['price(USD)']:
    if i <=400:
        p_list.append(1)                 #low range price
    elif i>400 and i<=700:
        p_list.append(2)                 #medium range price
    elif i>700 and i<=1000:
        p_list.append(3)                 #high range price
    else:
        p_list.append(4)                 #premium range price
df['price_range'] = p_list

df['width'] = [int(i.split('x')[0]) for i in df['resolution']]
df['height'] = [int(i.split('x')[1]) for i in df['resolution']]

le = LabelEncoder()
df['brand'] = le.fit_transform(df['brand'])
df['battery_type'] = le.fit_transform(df['battery_type'])
df['os'] = le.fit_transform(df['os'])

bool_col = [col for col in df.columns if df[col].dtype == 'bool']
df[bool_col] = df[bool_col].astype(int)

df['announcement_date'] = pd.to_datetime(df['announcement_date'])
df['year'] = df['announcement_date'].dt.year

camera = [x for x in df.columns if 'video' in x]
df['camera_score'] = df[camera].sum(axis=1)

df.drop(bool_col, axis=1, inplace=True)

df = df.drop(['phone_name', 'announcement_date', 'resolution'], axis=1)

# Separate features and target variable
X = df.drop(['price(USD)'], axis=1)
y = df['price(USD)']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=15)

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

print(f"Mean of 'price(USD)': {df['price(USD)'].mean()}")

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