# %%
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from methods import *

##### Loading of the data

data_path ="/Users/Bench/Documents/Desktop/EPFL/Master/MA1/Machine Learning/ml_project_1/train.csv" # Fill in the data path for the train values
data_path_test ="/Users/Bench/Documents/Desktop/EPFL/Master/MA1/Machine Learning/ml_project_1/test.csv" # Fill in the data path for the test values
ytrain, xtrain, ids = load_csv_data(data_path)
xtest, indices = load_csv_submit(data_path_test)


##### Initialization of the parameters of the model
max_iter = 700
gamma = 0.085
lambda_ = [0.001291549665014884, 0.000774263682681127, 0.002154434690031882] ##Parameter of the ridge \ Was optimized through grid search
threshold = 0.5
degree = 3 ## Degree max of polynomial of the feature expansion \ Was optimized through grid search
jeti_train = get_index_jet(xtrain)  ## Separation of the data using the jet number
jeti_test = get_index_jet(xtest)

weights = []
xte = []
for i in range(3):
    ## Processing of the data
    xtr = xtrain[jeti_train[i]]
    ytr = ytrain[jeti_train[i]]
    xte.append(xtest[jeti_test[i]])
    xtr, xte[i] = data_process(xtr, xte[i], degree)
    initial_w = np.ones(xtr.shape[1])

    ## Computation of the model
    loss, w = gd_log_ridge(ytr, xtr, initial_w, max_iter, gamma, lambda_[i])
    weights.append(w)

### Gluing back the data into one y_pred  conserving the order of the indices
y_pred = []
indi = []
for i in range(3):
    print("i=", i)
    yp = predict_labels(weights[i], xte[i])
    indi.append(indices[jeti_test[i]])
    y_pred.append(yp)

y_predfin = np.concatenate((y_pred[0],y_pred[1], y_pred[2]))
indifin = np.concatenate((indi[0],indi[1],indi[2]))

## Create the csv file for the submission on Ai Crowd

create_csv_submission(indifin, y_predfin, name='submission.csv')

# %%