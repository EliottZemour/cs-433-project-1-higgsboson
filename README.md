
# Project 1 : The Higgs Boson detection

The goal of this project is to determine if the detected particule is a Higgs boson, based on a list of features computed at the CERN

Team members : Thomas Benchetrit, Romain Palazzo, Eliott Zemour


## How to use
In order to get the same results as what was submitted on the [AiCrowd plateform](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs):
1. Install python 3.7 & numpy 1.14
2. Go the script folder and run `python run.py`. Once the program has stopped running, the submission will be available in the script folder under the name `submission.csv`

## Description of the files 
Five `.py` files are available in the script folder : 

- `run.py` which is used to get the submission 
- `methods.py` which contains the functions used to compute the model through different ML methods, cross-validate the results and to do feature expansion
- `proj1_helpers.py` which contains the functions used to import, process and export the data
- `implementations.py` which contains six basics ML functions to do linear, ridge and logistic regression, using both gradient descent and stochastic gradient descent
- `grid_search.py` which performs a grid search script over polynomial degree and lambda to determine the best hyper-parameters.
