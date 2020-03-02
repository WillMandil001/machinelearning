import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def readData():
    data = pd.read_csv('prostate_dataset.txt',delimiter="\t")
    is_train = data['train']=="T"
    is_test = data['train']=="F"

    data_train = data[is_train]
    data_test = data[is_test]

    train_X = data_train[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
    train_y = data_train[['lpsa']]
    test_X = data_test[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
    test_y = data_test[['lpsa']]
    print("data_train="+str(data_train.shape))
    print("data_test="+str(data_test.shape))
    print("train_X="+str(train_X.shape))
    print("train_y="+str(train_y.shape))
    print("test_X="+str(test_X.shape))
    print("test_y="+str(test_y.shape))

    return train_X, train_y, test_X, test_y

def trainModels(X, y, alpha_ridge, alpha_lasso):
    lse = linear_model.LinearRegression().fit(X, y)
    ridge = linear_model.Ridge(alpha=alpha_ridge)
    ridge = ridge.fit(X, y)
    lasso = linear_model.Lasso(alpha=alpha_lasso)
    lasso = lasso.fit(X, y)

    return lse, ridge, lasso

def testModels(lse, ridge, lasso):
    true_outputs = np.array(test_y['lpsa'])
    predictions_lse = lse.predict(test_X)
    predictions_ridge = ridge.predict(test_X)
    predictions_lasso = lasso.predict(test_X)

    for i in range(0, len(true_outputs)):
        print("y*["+str(i)+"]="+str(true_outputs[i])+" y_lse["+str(i)+"]="+str(predictions_lse[i][0])+" y_lse["+str(i)+"]="+str(predictions_ridge[i][0])+"...")

    print("lse.coef_="+str(lse.coef_))
    print("ridge.coef_="+str(ridge.coef_))
    print("lasso.coef_="+str(lasso.coef_))
    print("")
    print("mse_lse="+str(mean_squared_error(true_outputs, predictions_lse)))
    print("mse_ridge="+str(mean_squared_error(true_outputs, predictions_ridge)))
    print("mse_lasso="+str(mean_squared_error(true_outputs, predictions_lasso)))
    print("")
    print("mae_lse="+str(mean_absolute_error(true_outputs, predictions_lse)))
    print("mae_ridge="+str(mean_absolute_error(true_outputs, predictions_ridge)))
    print("mae_lasso="+str(mean_absolute_error(true_outputs, predictions_lasso)))

alpha_ridge = 0.4 # need to be optimised, e.g. using cross validation
alpha_lasso = 0.4 # need to be optimised, e.g. using cross validation
train_X, train_y, test_X, test_y = readData()
lse, ridge, lasso = trainModels(train_X, train_y, alpha_ridge, alpha_lasso)
testModels(lse, ridge, lasso)
