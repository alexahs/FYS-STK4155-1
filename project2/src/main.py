import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import time


from pylearn.logisticregression import SGDClassification
from pylearn.linearmodel import Regression
from pylearn.metrics import *
from pylearn.neuralnetwork import NeuralNetwork
from pylearn.resampling import CV



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


def main():
    np.random.seed(2019)
    filename = 'data/default_of_credit_card_clients.xls'
    X, y = load_CC_data(filename)

    scale_columns = list(range(9, X.shape[1]))

    model = SGDClassification()

    t0 = time.time()
    mse, r2, accuracy = CV(X, y, model, scale_columns=scale_columns)
    t1 = time.time()
    print('elapsed time CV:', t1 - t0)

    print('mse:', mse)
    print('r2:', r2)
    print('accuracy:', accuracy)







if __name__ == '__main__':
    main()
