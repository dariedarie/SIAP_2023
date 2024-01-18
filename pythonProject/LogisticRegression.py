import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import csv

# Read training data
data = pd.read_csv('datasets/train.csv')

# Splitting into X and y as per the required for Scikit learn packages
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# Splitting the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
clf.fit(X_train, Y_train)

preds = clf.predict(X_test)

accuracy = float(np.sum(preds == Y_test)) / Y_test.shape[0]
print("accuracy: %f" % (accuracy))

# Read test data
testdata = pd.read_csv('datasets/test.csv')

# Exclude 'id' column from test data
testdata = testdata.drop(columns=['id'])

test_prediction = clf.predict(testdata)

file = open("Prediction.csv", "w", newline="")
new_file = csv.writer(file)
for i in range(len(test_prediction)):
    new_file.writerow(list([i + 1, test_prediction[i]]))

file.close()
