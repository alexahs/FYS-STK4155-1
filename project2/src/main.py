import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
import pandas as pd
import time
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix
import scikitplot.metrics as skplt
import seaborn as sns


from pylearn.logisticregression import SGDClassification
from pylearn.linearmodel import Regression
from pylearn.metrics import *
from pylearn.neuralnetwork import NeuralNetwork
from pylearn.resampling import *



def preprocess_CC_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)


    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


    #find and remove outliers in the data
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


    #split data into categorical and continuous features
    categorical_inds = (1, 2, 3, 5, 6, 7, 8, 9, 10)
    continuous_inds = (0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    X_cat = X[:,categorical_inds]
    X_cont = X[:, continuous_inds]


    #onehot encode categorical data
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)
    preprocessor = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('onehot', onehotencoder, list(range(X_cat.shape[1])))])

    X_cat = preprocessor.fit_transform(X_cat)

    #join categorical and continuous features
    X = np.concatenate((X_cont, X_cat), axis=1)

    cont_feat_inds = list(range(X_cont.shape[1]))

    return X, np.ravel(y), cont_feat_inds



def analyze_logistic(X, y, model, scale_columns, analyze_eta=False):

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)

    minmaxscaler = MinMaxScaler()
    scaler = ColumnTransformer(
                        remainder='passthrough',
                        transformers=[('minmaxscaler', minmaxscaler, scale_columns)])



    scaler.fit(X_train_val)
    X_test = scaler.transform(X_test)


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
            print(error_models[:, i])

            i += 1
        max_accuracy_ind = np.argmax(error_models[2, :])
        eta_opt = error_models[2, max_accuracy_ind]
        model.set_eta(eta_opt)
    #end if

    X_train_val = scaler.transform(X_train_val)



    #model
    model.fit(X_train_val, y_train_val)
    pred_train = model.predict(X_train_val)
    pred_test = model.predict(X_test)


    #sklearn
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X_train_val, y_train_val)
    # pred_skl = clf.predict(X_test)


    # pred_train = model.predict(X_train_val, probability=True)
    # pred_test = model.predict(X_test, probability=True)
    # area_ratio_train = cumulative_gain_area_ratio(y_train_val, pred_train, title='training results')
    # area_ratio_test = cumulative_gain_area_ratio(y_test, pred_test, title='test results')
    # print('area ratio train:', area_ratio_train)
    # print('area ratio test:', area_ratio_test)



    ax1 = plot_confusion_matrix(y_test, pred_test, normalize=True, cmap='Blues')
    # ax2 = plot_confusion_matrix(y_test, pred_skl, normalize=True, cmap='Reds')


    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    # ax2.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()








def main():
    # np.random.seed(2019)
    filename = 'data/default_of_credit_card_clients.xls'
    X, y, scale_columns = preprocess_CC_data(filename)


    # print(X.shape)


    # dataset = load_breast_cancer()
    # X, y = dataset.data, dataset.target


    # scale_columns = list(range(9, X.shape[1]))


    model = SGDClassification()
    # clf = linear_model.LogisticRegressionCV()

    # cumulative_gain(X, y, model)

    analyze_logistic(X, y, model, scale_columns, analyze_eta=False)


    # corr = pd.DataFrame(X)
    # c = corr.corr().round(2)
    # # print(c)
    #
    #
    #
    # sns.heatmap(c)
    # plt.show()
    #
    #
    # U, S, VT = np.linalg.svd(c)
    #
    # plt.semilogy(S)
    # plt.show()


    # U, S, VT = np.linalg.svd(X)

    # plt.semilogy(S)
    # plt.show()




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
