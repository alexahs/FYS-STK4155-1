import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pylearn.logisticregression import *


class Test():

    def __init__(self):
        pass


    def testLogisticRegression(self):
        dataset = datasets.load_breast_cancer()
        data = dataset.data
        target = dataset.target

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)



        clf = linear_model.LogisticRegressionCV()
        clf.fit(X_train_scaled, y_train)
        pred_skl = clf.predict(X_test_scaled)

        model = SGDClassification()
        model.fit(X_train_scaled, y_train)
        pred_model = model.predict(X_test_scaled)

        accuracy_test_skl = accuracy_score(pred_skl, y_test)
        accuracy_test_model = accuracy_score(pred_model, y_test)
        accuracy_model_skl = accuracy_score(pred_skl, pred_model)

        print('scikit accuracy:', accuracy_test_skl)
        print('model accuracy:', accuracy_test_model)





if __name__ == '__main__':
    tests = Test()
    tests.testLogisticRegression()
