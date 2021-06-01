'''

Grid search script over polynomial degree and lambda (ridge parameter)
The following parameters are chosen by manual inspection before
building the grid search :
    - max_iter = 700 
    - gamma = 0.085 
Note that the result of the grid search depends on these parameters

The grid search loop returns a np.array of shape (3,4,10) : score.

In order to get the optimal combination (degree, lambda_) for each jet number i:
    - (d,l) = np.unravel_index(np.argmax(score[i]), score[i].shape)
    - degree_opt_i = degrees[d]
    - lambda_opt_i = lambdas[l]
    
    
Running this script gave us 
jetnum 0:
    degree = 3
    lambda = 0.0007278953843983146
jetnum 1:
    degree = 3
    lambda = 0.0007880462815669912
jetnum 2:
    degree = 3
    lambda = 0.00045203536563602405
'''

# %% USEFUL IMPORTS
##################################################

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from methods import *
from matplotlib import cm

# %%
##################################################
### Loading csv data ###

data_path = "C:\\Users\\Eliott\\Desktop\\train.csv"
ys, datas, ids = load_csv_data(data_path)

# %%  GRID SEARCH LOOP
##################################################


ratio = 0.5
max_iter = 700
gamma = 0.085

method = 'log'
threshold = 0.5

xtrain, ytrain, xtest, ytest = split_data(datas, ys, ratio)

jet_index_train = get_index_jet(xtrain)
jet_index_test = get_index_jet(xtest)

lambdas = np.logspace(-4, -3, 30)
degrees = np.array([2, 3])

score = np.ones((3, len(degrees), len(lambdas)))

for d, degree in enumerate(degrees):
    print('degree =', degree)
    for l, lambda_ in enumerate(lambdas):
        print('lambda_index =', l)
        for i in range(3):
            # i = jet number, learning processes are different between i's
            train_index = jet_index_train[i]
            test_index = jet_index_test[i]

            x_tr, y_tr = xtrain[train_index], ytrain[train_index]
            x_te, y_te = xtest[test_index], ytest[test_index]

            x_tr, x_te = data_process(x_tr, x_te, degree)
            initial_w = np.random.rand(x_tr.shape[1])
            weight, loss, yhat, ypred = learning(y_tr, x_tr, x_te, initial_w, max_iter, gamma, lambda_, threshold,
                                                 method)

            score[i, d, l] = compute_score(ypred, y_te)

# %% GRID SEARCH RESULTS
##################################################

for i in range(3):
    (d, l) = np.unravel_index(np.argmax(score[i]), score[i].shape)
    print('jet_number =', i)
    print('==============')
    print('degree_opt', '=', degrees[d])
    print('lambda_opt', '=', lambdas[l])
    print('score =', score[i, d, l], '\n')
