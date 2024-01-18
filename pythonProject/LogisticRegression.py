# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from tqdm.notebook import tqdm

# Load the dataset
data = pd.read_csv("datasets/cleaned_all_phones.csv")

# Explore the data
print(data.head())

# Data preprocessing
# Add your data preprocessing steps here

# Assuming `df` is your DataFrame
df = data.copy()  # Not sure why you're copying twice, but I retained it
X = df.drop(columns='price(USD)')
y = df['price(USD)']

cat_col = X.select_dtypes(include=['object']).columns
num_col = ['inches', 'battery', 'ram(GB)', 'weight(g)', 'storage(GB)', 'has_pro_or_max', 'number_of_pixels']

def preprocessing(X, y, num_col, cat_col):
    # Using StandardScaler for numerical columns
    num_scaler = StandardScaler()
    X[num_col] = num_scaler.fit_transform(X[num_col])

    # Using OrdinalEncoder for categorical columns
    cat_encoder = OrdinalEncoder()
    X[cat_col] = cat_encoder.fit_transform(X[cat_col])

    # Assuming 'y' is a 1D array, using StandardScaler for target variable 'y'
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(np.array(y).reshape(-1, 1))

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocessing(X, y, num_col, cat_col)

print("Number of transactions in x_train dataset:", x_train.shape)
print("Number of transactions in y_train dataset:", y_train.shape)
print("Number of transactions in x_test dataset:", x_test.shape)
print("Number of transactions in y_test dataset:", y_test.shape)


# Split the data into features and target variable
X = data.drop("target_variable", axis=1)
y = data["target_variable"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)

# Train Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
svm_predictions = svm_model.predict(X_test_scaled)
gb_predictions = gb_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate the models
svm_accuracy = accuracy_score(y_test, svm_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"SVM Accuracy: {svm_accuracy}")
print(f"Gradient Boosting Accuracy: {gb_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

# Display additional evaluation metrics, visualizations, or further analysis
# Add your additional analysis code here

# Define models
models = []
models.append(['KNeighbors', KNeighborsRegressor()])
models.append(['Decision Tree', DecisionTreeRegressor()])
models.append(['Random Forest', RandomForestRegressor()])
models.append(['SVM', SVR()])
models.append(['SGD', SGDRegressor()])
models.append(['AdaBoost Regressor ', AdaBoostRegressor()])
models.append(['GradientBoosting', GradientBoostingRegressor()])
models.append(['XGBoost ', XGBRegressor()])

# ... (previous code)

# Train models and evaluate
def train_models(models, x_train, y_train, x_test, y_test):
    lst_1 = []
    for m in tqdm(range(len(models))):
        lst_2 = []
        print(models[m][0])
        model = models[m][1]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)   # K-Fold Validation

        MSE = mean_squared_error(y_test, y_pred)
        MAE = mean_absolute_error(y_test, y_pred)
        R2 = r2_score(y_test, y_pred)

        lst_2.append(models[m][0])
        lst_2.append((model.score(x_test, y_test)) * 100)
        lst_2.append(accuracies.mean() * 100)
        lst_2.append(accuracies.std() * 100)
        lst_2.append(MSE)
        lst_2.append(MAE)
        lst_2.append(R2)

        lst_1.append(lst_2)

    df = pd.DataFrame(lst_1, columns=['Model', 'Accuracy', 'K-Fold Mean Accuracy', 'Std. Deviation', 'MSE', 'MAE', 'R2'])
    df.sort_values(by=['Accuracy', 'K-Fold Mean Accuracy'], inplace=True, ascending=False)
    return df

train_models(models, x_train, y_train, x_test, y_test)

# ... (previous code)

# x_train['target'] = y_train
cor = x_train.corr()
cor_target = abs(cor["target"])

# Selecting highly correlated features
relevant_features = list(cor_target[cor_target > 0.1].index)
relevant_features.remove('target')

# Extract relevant features from the original DataFrame 'df'
X = df[relevant_features]
y = df['price(USD)']

# Categorize columns
cat_col = X.select_dtypes(include=['object']).columns
num_col = X.select_dtypes(exclude=['object']).columns

# Perform preprocessing on the relevant features
x_train, x_test, y_train, y_test = preprocessing(X, y, num_col, cat_col)

print("Number of transactions in x_train dataset:", x_train.shape)
print("Number of transactions in y_train dataset:", y_train.shape)
print("Number of transactions in x_test dataset:", x_test.shape)
print("Number of transactions in y_test dataset:", y_test.shape)

# Train models and evaluate
train_models(models, x_train, y_train, x_test, y_test)

