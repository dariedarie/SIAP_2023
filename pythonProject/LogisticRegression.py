import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

# Read training data
data = pd.read_csv('datasets/train.csv')

# Splitting into X and y as per the required for Scikit learn packages
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# Splitting the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
clf.fit(X_train, Y_train)

# Predictions on the test set
preds = clf.predict(X_test)

# Calculate accuracy
accuracy = float(np.sum(preds == Y_test)) / Y_test.shape[0]
print("Accuracy: %f" % (accuracy))

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, preds)
print("Mean Squared Error (MSE): %f" % mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE): %f" % rmse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, preds)
print("Mean Absolute Error (MAE): %f" % mae)

# Read test data
testdata = pd.read_csv('datasets/test.csv')

# Exclude 'id' column from test data
testdata = testdata.drop(columns=['id'])

# Predictions on the test data
test_prediction = clf.predict(testdata)

# Write predictions to a CSV file
file = open("Prediction.csv", "w", newline="")
new_file = csv.writer(file)
for i in range(len(test_prediction)):
    new_file.writerow(list([i + 1, test_prediction[i]]))

file.close()
