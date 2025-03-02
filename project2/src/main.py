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
from matplotlib.patches import Rectangle


from pylearn.logisticregression import SGDClassification
from pylearn.linearmodel import Regression
from pylearn.metrics import *
from pylearn.neuralnetwork import NeuralNetwork
from pylearn.resampling import *



def preprocess_CC_data(filename, which_onehot = 1):
    """
    Function for preprocessing the credit card data. Removes outliers in the
    data set, and uses one hot encoding on categorical features.

    Inputs:
    - filename of the data set
    - which_onehot:
        - "1" specifies to one hot encode the features sex, education and marriage
        - "2" specifies to one hot encode the features sex, education, marriage and payment history

    Returns:
    - The design matrix, shape (n_observations, n_features).
    - Target vector, shape (n_observations,).
    - Indices of continuous features in the data set
    """

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
    if which_onehot==1:
        #only marriage, sex and education onehot encoded
        categorical_inds = (1, 2, 3)
        continuous_inds = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


    elif which_onehot==2:
        #all categories onehot encoded
        categorical_inds = (1, 2, 3, 5, 6, 7, 8, 9, 10)
        continuous_inds = (0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

    else:
        print('which_onehot must be specified as either 1 or 2')
        exit(0)

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

    continuous_feature_inds = list(range(X_cont.shape[1]))

    return X, np.ravel(y), continuous_feature_inds



def analyze_logistic(X, y, model, scale_columns, analyze_params=False, balance_outcomes=False):
    """
    Function for doing analysis of logistic regression. Plots cumulative gain, confusion matrix
    and grid search of optimal learning rate/epochs in SGD with k-fold CV (optional).
    Performs scaling of all continuous features in the data set.

    Inputs:
    - X: design matrix, shape (n, p)
    - y: targets, shape (n,)
    - scale_columns: list of indices of which columns to MinMax scale
    - analyze_params: boolean, option to perform grid search of learning rate and n_epochs in SGD
    - balance_outcomes: boolean, option to balance training data in case of skewed classes
    """

    #split data in train/validate and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1)


    #balance training set such that outcomes are 50/50 in training data
    if balance_outcomes:
        non_default_inds = np.where(y_train_val==0)[0]
        default_inds = np.where(y_train_val==1)[0]

        remove_size = len(non_default_inds) - len(default_inds)
        remove_inds = np.random.choice(non_default_inds, size=remove_size, replace=False)

        X_train_val = np.delete(X, remove_inds, axis=0)
        y_train_val = np.delete(y, remove_inds, axis=0)
    #end if



    #scale continuous features
    minmaxscaler = MinMaxScaler(feature_range=(-1,1))
    scaler = ColumnTransformer(
                        remainder='passthrough',
                        transformers=[('minmaxscaler', minmaxscaler, scale_columns)])


    #scale only test data at this point (CV scales training/validation)
    scaler.fit(X_train_val)
    X_test = scaler.transform(X_test)


    if analyze_params:

        #initialize vectors for saving results
        error_scores = pd.DataFrame(columns=['log eta', 'n_epochs', 'mse', 'r2', 'accuracy'])
        n_etas = 4
        eta_vals = np.linspace(-1, -4, n_etas)
        n_epoch_vals = np.array([10, 100, 500, 1000])
        n_epochs = len(n_epoch_vals)
        accuracy_scores = np.zeros((n_etas, n_epochs))

        max_accuracy = 0
        best_eta = 0
        best_n_epochs = 0

        #perform grid search of best learning rate
        #and number of epochs with k-fold cross-validation
        i = 0
        for eta in eta_vals:
            model.set_eta(10**eta)

            j = 0
            for epoch in n_epoch_vals:
                model.set_n_epochs(epoch)

                #perform cross validation
                mse, r2, accuracy = CV(X_train_val, y_train_val, model)
                accuracy_scores[i, j] = accuracy

                error_scores = error_scores.append({'log eta': eta,
                                                    'n_epochs': epoch,
                                                    'mse': mse,
                                                    'r2': r2,
                                                    'accuracy': accuracy}, ignore_index=True)

                #check if current configuration is better
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_eta = eta
                    best_n_epochs = epoch


                j += 1
                #end for epoch
            i += 1
            #end for eta

        #set optimal model parameters
        model.set_eta(10**best_eta)
        model.set_n_epochs(best_n_epochs)

        #plot heatmap of grid search
        acc_table = pd.pivot_table(error_scores, values='accuracy', index=['log eta'], columns='n_epochs')
        idx_i = np.where(acc_table == max_accuracy)[0]
        idx_j = np.where(acc_table == max_accuracy)[1]

        fig = plt.figure()
        ax = sns.heatmap(acc_table, annot=True, fmt='.2g', cbar=True, linewidths=1, linecolor='white',
                            cbar_kws={'label': 'Accuracy'})

        ax.add_patch(Rectangle((idx_j, idx_i), 1, 1, fill=False, edgecolor='red', lw=2))
        ax.set_xlabel('Number of epochs')
        ax.set_ylabel(r'log$_{10}$ of Learning rate')

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()
    #end if

    #scale training data
    X_train_val = scaler.transform(X_train_val)

    #pylearn model
    model.fit(X_train_val, y_train_val)
    pred_train = model.predict(X_train_val)
    pred_test = model.predict(X_test)

    #sklearn model
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X_train_val, y_train_val)
    pred_skl = clf.predict(X_test)

    #get accuracy scores
    accuracy_on_test = accuracy_score(y_test, pred_test)
    accuracy_on_train = accuracy_score(y_train_val, pred_train)
    accuracy_skl =accuracy_score(y_test, pred_skl)


    #predict
    pred_train_prob = model.predict(X_train_val, probability=True)
    pred_test_prob = model.predict(X_test, probability=True)


    #get area ratio and plot cumulaive gain
    area_ratio_train = cumulative_gain_area_ratio(y_train_val, pred_train_prob, title='Training results')
    area_ratio_test = cumulative_gain_area_ratio(y_test, pred_test_prob, title=None)
    plt.show()


    #plot confusion matrix
    ax1 = plot_confusion_matrix(y_test, pred_test, normalize=True, cmap='Blues', title=' ')
    ax2 = plot_confusion_matrix(y_train_val, pred_train, normalize=True, cmap='Blues', title='Training data')

    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    ax2.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()

    #print some stats
    print('===accuracy and area ratio stats===')
    print('accuracy on test:', accuracy_on_test)
    print('accuracy on train:', accuracy_on_train)
    print('accuracy skl:', accuracy_skl)
    print('area ratio train:', area_ratio_train)
    print('area ratio test:', area_ratio_test)


    if analyze_params:
        print('===grid search stats===')
        print('max accuracy:', max_accuracy)
        print('eta:', best_eta)
        print('n_epochs:', best_n_epochs)



def main():
    np.random.seed(2010)
    filename = 'data/default_of_credit_card_clients.xls'
    # X, y, scale_columns = preprocess_CC_data(filename, which_onehot = 2)

    # dim_red(X)

    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    scale_columns = list(range(X.shape[1]))




    model = SGDClassification(batch_size=100, eta0=0.001)
    # clf = linear_model.LogisticRegressionCV()

    # cumulative_gain(X, y, model)

    analyze_logistic(X, y, model, scale_columns, analyze_params=True, balance_outcomes=False)





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
