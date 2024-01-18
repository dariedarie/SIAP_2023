import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import csv


data = pd.read_csv('mobileCSVs/train.csv')
#Splitting into X and y as per the required for Scikit learn packages
X, y = data.iloc[:,:-1], data.iloc[:,-1]

#Splitting the dataset into training and testing
X_train, X_test, Y_train, Y_test= train_test_split(X, y, test_size=0.2, random_state=123)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# Y_train_minmax = min_max_scaler.fit_transform(Y_train)
# X_test_minmax = min_max_scaler.transform(X_test)
# Y_test_minmax = min_max_scaler.transform(Y_test)

clf = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
# clf = LogisticRegression(multi_class='multinomial',class_weight='balanced', solver='saga')
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, Y_train)
clf.fit(X_train,Y_train)

preds = clf.predict(X_test)

accuracy = float(np.sum(preds==Y_test))/Y_test.shape[0]
print("accuracy: %f" % (accuracy))

testdata =pd.read_csv('mobileCSVs/test.csv')
test_prediction=clf.predict(testdata)

file = open("Prediction.csv","w",newline="")
new_file = csv.writer(file)
#new_file.writerow(['id','price_range'])
for i in range(0,1000):
    new_file.writerow(list([i+1,test_prediction[i]]))

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
