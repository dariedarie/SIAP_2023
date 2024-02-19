import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def calculate_pixels(resolution):
    try:
        h, w = map(int, resolution.split('x'))
        return h * w
    except ValueError:
        return None

def data_preprocessing():
    glavni = pd.read_csv('datasets/cleaned_all_phones.csv')

    data = [glavni]

    def calculate_pixels(resolution):
        try:
            h, w = map(int, resolution.split('x'))
            return h * w
        except ValueError:
            return None

    for df in data:
        recnik = {
            'Android 8.1': 'Android 8.0',
            'Android 12L': 'Android 12',
            'Android 9.0 Pie': 'Android 9.0',
            'Android 7.1.1': 'Android 7.0',
            'Android 7.1': 'Android 7.0',
            'Android 7.1.2': 'Android 7.0',
            'Android 7.0.1': 'Android 7.0',
            'Android 8.0 Oreo': 'Android 8.0',
            'Android 8.1 Oreo': 'Android 8.0',
            'Android 6': 'Android 6.0',
            'Android 6.0.1': 'Android 6.0',
            'Android 12 or 13': 'Android 12',
            'Android 10/ Android 11': 'Android 10',
            'iOS 15.4': 'iOS 15',
            'iOS 14.1': 'iOS 14',
            'iOS 11.1.1': 'iOS 11'
        }
        df['os'] = df['os'].replace(recnik)
        df[['os_name', 'os_version']] = df['os'].str.split(n=1, expand=True)
        df['number_of_pixels'] = df['resolution'].apply(calculate_pixels)
        df['announcement_year'] = df['announcement_date'].apply(lambda x: x.split('-')[0]).astype('int32')
        df.drop(columns=['os'], inplace=True)
        bool_col = [col for col in df.columns if df[col].dtype == 'bool']
        df[bool_col] = df[bool_col].astype(int)
        outliers = df[(df['weight(g)'] > 450.0) | (df['price(USD)'] > 2000.0)]
        le = LabelEncoder()
        df['brand'] = le.fit_transform(df['brand'])
        df['battery_type'] = le.fit_transform(df['battery_type'])
        df['os_name'] = le.fit_transform(df['os_name'])
        df['os_version'] = le.fit_transform(df['os_version'])
        camera = [x for x in df.columns if 'video' in x]
        df['better_cam'] = df[camera].sum(axis=1)
        df.drop(outliers.index, inplace=True)
        df.drop(columns=['resolution'], inplace=True)
        df.drop(columns=['announcement_date'], inplace=True)
        df.drop(columns=['phone_name'], inplace=True)
        df.drop(bool_col, axis=1, inplace=True)

    X = glavni.drop(['price(USD)'], axis=1)
    y = glavni['price(USD)']


    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

    return xtrain,ytrain,xtest,ytest