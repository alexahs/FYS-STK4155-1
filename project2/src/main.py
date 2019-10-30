import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
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



def cumulative_gain(X, y, model):
    # X_train_val, X_test, y_train_val, y_test = train_test_spilt(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #
    scale_columns = list(range(9, X.shape[1]))
    #
    X_train, X_test = standardize_specific(X_train, X_test, scale_columns)
    #
    model.fit(X_train, y_train)
    #
    y_pred = model.predict(X_test, probability=True)
    # print(model.beta)

    y_pred = y_pred.reshape((len(y_pred), 1))
    y_test = y_test.reshape((len(y_test), 1))

    y_probas = np.concatenate((1-y_pred, y_pred), axis=1)




    area_ratio = cumulative_gain_area_ratio(y_test, y_probas)






def analyze_logistic(X, y, model, analyze_eta=False):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)


    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    columns = list(range(9, X.shape[1]))
    minmaxscaler = MinMaxScaler()
    scaler = ColumnTransformer(
                        remainder='passthrough',
                        transformers=[('minmaxscaler', minmaxscaler, columns)])



    scaler.fit(X_train_val)
    X_test = scaler.transform(X_test)


    # print(y_train_val.shape)
    # print(y_test.shape)


    if analyze_eta:
        n_etas = 5
        eta_vals = np.logspace(-1, -5, n_etas)

        #0: mse, 1: r2, 2:accuracy
        error_models = np.zeros((3, n_etas))
        i = 0
        for eta in eta_vals:
            print('eta=', eta)
            model.set_eta(eta)

            error_models[:, i] = CV(X_train_val, y_train_val, model)

            i += 1
    #end if

    X_train_val = scaler.transform(X_train_val)

    model.fit(X_train_val, y_train_val)

    pred_train = model.predict(X_train_val)
    pred_test = model.predict(X_test)

    # print("X_test:")
    # print(X_test.shape)
    #
    #
    # print("train1:")
    # print(y_train_val.shape)
    # print(pred_train.shape)
    #
    # print("test1:")
    # print(y_test.shape)
    # print(pred_test.shape)


    pred_train = pred_train.reshape((len(pred_train), 1))
    y_train = y_train_val.reshape((len(y_train_val), 1))
    pred_train = np.concatenate((1-pred_train, pred_train), axis=1)


    pred_test = pred_test.reshape((len(pred_test), 1))
    y_test = y_test.reshape((len(y_test), 1))
    pred_test = np.concatenate((1 - pred_test, pred_test), axis=1)



    # print(y_prob_test.shape)
    # print(y_prob_train.shape)
    # print(y_test.shape)
    # print(y_train_val.shape)

    # print("train2:")
    # print(y_train_val.shape)
    # print(pred_train.shape)
    #
    #
    # print("test2:")
    # print(y_test.shape)
    # print(pred_test.shape)


    area_ratio_train = cumulative_gain_area_ratio(y_train, pred_train, title='training results')
    area_ratio_test = cumulative_gain_area_ratio(y_test, pred_test, title='test results')

    print('area ratio train:', area_ratio_train)
    print('area ratio test:', area_ratio_test)








def main():
    np.random.seed(2019)
    filename = 'data/default_of_credit_card_clients.xls'
    X, y = load_CC_data(filename)
    # dataset = load_breast_cancer()

    # X, y = dataset.data, dataset.target

    scale_columns = list(range(9, X.shape[1]))



    model = SGDClassification()

    # cumulative_gain(X, y, model)

    analyze_logistic(X, y, model)

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
