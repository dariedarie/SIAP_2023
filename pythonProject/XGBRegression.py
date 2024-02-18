import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def calculate_pixels(resolution):
    try:
        h, w = map(int, resolution.split('x'))
        return h * w
    except ValueError:
        return None

def data_preprocessing():
    # test_data = pd.read_csv('datasets/TestSet.csv')
    # train_data = pd.read_csv('datasets/TrainingSet.csv')

    glavni = pd.read_csv('datasets/cleaned_all_phones.csv')

    # data = [test_data, train_data]
    data = [glavni]


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
        #df['announcement_year'] = df.announcement_date.apply(lambda x: x.split('-')[0] if '-' in x else x.split('/')[0]).astype('int64')
        df['announcement_year'] = df['announcement_date'].apply(lambda x: x.split('-')[0]).astype('int32')
        df.drop(columns=['os'], inplace=True)
        df.drop(columns=['resolution'], inplace=True)
        df.drop(columns=['announcement_date'], inplace=True)
        df.drop(columns=['phone_name'], inplace=True)

    bool_cols = list(glavni.select_dtypes(include='bool').columns)
    glavni[bool_cols] = glavni[bool_cols].replace({True: 1, False: 0})

    print(glavni)


    text_colums = ['brand', 'battery_type', 'os_name']
    for col in text_colums:
        encoder = LabelEncoder()
        glavni[col] = encoder.fit_transform(glavni[col])

    X = glavni.drop('price(USD)', axis=1)
    y = glavni['price(USD)']


    #
    # num_scaler = StandardScaler()
    # y_scaler = StandardScaler()
    # X = num_scaler.fit_transform(X)
    # y = y_scaler.fit_transform(np.array(y).reshape(-1, 1))




    xtrain, ytrain, xtest, ytest = train_test_split(X, y, test_size=0.3, random_state=0)


    # xtrain = train_data.drop('price(USD)', axis=1)
    # ytrain = train_data['price(USD)']
    # xtest = test_data.drop('price(USD)', axis=1)
    # ytest = test_data['price(USD)']

    #print(xtrain)

    # data2 = [xtrain, xtest]
    # text_colums = ['brand', 'battery_type', 'os_name']
    # for df in data2:
    #     for col in text_colums:
    #         encoder = LabelEncoder()
    #         df[col] = encoder.fit_transform(df[col])


    # camera = [x for x in train_data.columns if 'video' in x]
    # train_data['camera_score'] = train_data[camera].sum(axis=1)
    # xtest = xtest.dropna()
    # ytest = ytest.dropna()
    # ytest = ytest[xtest.index.isin(ytest.index)]
    # idx = ytest.index.difference(xtest.index)
    # ytest = ytest.loc[~ytest.index.isin(idx)]
    # print(len(ytest), len(xtest))
    #ytrain = ytrain[xtrain.index.isin(ytrain.index)]

    print(len(xtrain))
    print(len(ytrain))

    xtrain = xtrain.dropna()
    #### ne slazu se duzine xtrain-a i ytrain-a, kada dropna iz xtraina ne znam sta da uradim

    print(len(xtrain))
    print(len(ytrain))

    return xtrain,ytrain,xtest,ytest