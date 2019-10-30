import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
# from sklearn.datasets import load_breast_cancer
import pandas as pd
import time
import matplotlib.pyplot as plt
# from scikitplot.metrics import plot_cumulative_gain
import scikitplot.metrics as skplt


from pylearn.logisticregression import SGDClassification
from pylearn.linearmodel import Regression
from pylearn.metrics import *
from pylearn.neuralnetwork import NeuralNetwork
from pylearn.resampling import *



def load_CC_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


    outlier_gender1 = np.where(X[:,1] < 1)[0]
    outlier_gender2 = np.where(X[:,1] > 2)[0]

    outlier_education1 = np.where(X[:,2] < 1)[0]
    outlier_education2 = np.where(X[:,2] > 4)[0]

    outlier_marital1 = np.where(X[:,3] < 1)[0]
    outlier_marital2 = np.where(X[:,3] > 3)[0]

    inds = np.concatenate((outlier_gender1,
                           outlier_gender2,
                           outlier_education1,
                           outlier_education2,
                           outlier_marital1,
                           outlier_marital2))


    outlier_rows = np.unique(inds)

    X = np.delete(X, outlier_rows, axis=0)
    y = np.delete(y, outlier_rows, axis=0)


    onehotencoder = OneHotEncoder(categories="auto")
    preprocessor = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('onehot', onehotencoder, [1, 2, 3])])


    X = preprocessor.fit_transform(X)

    # y = onehotencoder.fit_transform(y)


    return X, np.ravel(y)



def cumulative_gain(X_test, y_test, model):
    # X_train_val, X_test, y_train_val, y_test = train_test_spilt(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scale_columns = list(range(9, X.shape[1]))

    X_train, X_test = standardize_specific(X_train, X_test, scale_columns)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test, probability=True)
    # print(model.beta)

    y_pred = y_pred.reshape((len(y_pred), 1))
    y_test = y_test.reshape((len(y_test), 1))

    y_probas = np.concatenate((1-y_pred, y_pred), axis=1)



    area_ratio = cumulative_gain_area_ratio(y_test, y_probas)



def analyze_logistic(X, y, model):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)

    scale_columns = list(range(9, X.shape[1]))
    minmaxscaler = MinMaxScaler()
    scaler = ColumnTransformer(
                        remainder='passthrough',
                        transformers=[('minmaxscaler'), minmaxscaler, columns])

    X_train_val = scaler.fit(X_train_val)
    X_test = scaler.transform(X_train_val)

    n_etas = 4
    eta_vals = np.logspace(-1, -5, n_etas)

    for eta in eta_vals:
        model.set_eta(eta)
        




def main():
    np.random.seed(2019)
    filename = 'data/default_of_credit_card_clients.xls'
    X, y = load_CC_data(filename)

    scale_columns = list(range(9, X.shape[1]))



    model = SGDClassification()

    cumulative_gain(X, y, model)

    # t0 = time.time()
    # mse, r2, accuracy = CV(X_reduced, y, model, scale_columns=None)
    # t1 = time.time()
    # print('elapsed time CV:', t1 - t0)
    #
    # print('mse:', mse)
    # print('r2:', r2)
    # print('accuracy:', accuracy)







if __name__ == '__main__':
    main()
