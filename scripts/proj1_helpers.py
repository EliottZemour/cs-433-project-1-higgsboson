# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from methods import *


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def get_index_jet(x):
    """Get the index of the groups depending on the jet number (Described in the rapport)."""
    jeti_0 = np.where(x[:, 22] == 0)[0]
    jeti_1 = np.where(x[:, 22] == 1)[0]
    jeti_2 = np.where(x[:, 22] >= 2)[0]
    return [jeti_0, jeti_1, jeti_2]


def load_csv_submit(data_path):
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    return input_data, ids


def load_jetnum_submit(data_path):
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    no_jet_ind = np.where(input_data[:, 22] == 0)  # in which case we neglect a lot of columns
    drop_columns = np.array([4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29])
    data_0 = input_data[no_jet_ind[0], :]
    data_0 = np.delete(data_0, drop_columns, axis=1)
    data_0 = standardize(data_0)
    data_0[np.where(data_0 > 3)] = 0
    data_0[np.where(data_0 < -3)] = 0
    id_0 = ids[no_jet_ind]
    data_0 = np.insert(data_0, 0, 1, axis=1)

    print("ok for 0")

    one_jet_ind = np.where(input_data[:, 22] == 1)  # in which case we neglect some columns
    drop_columns_1 = np.array([4, 5, 6, 12, 22, 26, 27, 28])
    data_1 = input_data[one_jet_ind[0], :]
    data_1 = np.delete(data_1, drop_columns_1, axis=1)
    data_1 = standardize(data_1)
    data_1[np.where(data_1 > 3)] = 0
    data_1[np.where(data_1 < -3)] = 0
    id_1 = ids[one_jet_ind]
    data_1 = np.insert(data_1, 0, 1, axis=1)

    print("ok for 1")

    more_jet_ind = np.where(input_data[:, 22] > 1)  # in which case we neglect no columns
    data_2 = input_data[more_jet_ind[0], :]
    data_2 = np.delete(data_2, 22, axis=1)
    data_2 = standardize(data_2)
    data_2[np.where(data_2 > 3)] = 0
    data_2[np.where(data_2 < -3)] = 0
    id_2 = ids[more_jet_ind]
    data_2 = np.insert(data_2, 0, 1, axis=1)

    print("ok for 2")

    datas = [data_0, data_1, data_2]
    indexes = [id_0, id_1, id_2]

    return datas, indexes


def load_jetnum(data_path):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0

    no_jet_ind = np.where(input_data[:, 22] == 0)  # in which case we neglect a lot of columns
    drop_columns = np.array([4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29])
    data_0 = input_data[no_jet_ind[0], :]
    data_0 = np.delete(data_0, drop_columns, axis=1)
    data_0 = standardize(data_0)
    data_0[np.where(data_0 > 3)] = 0
    data_0[np.where(data_0 < -3)] = 0
    y_0 = yb[no_jet_ind]
    id_0 = ids[no_jet_ind]
    data_0 = np.insert(data_0, 0, 1, axis=1)

    print("ok for 0")

    one_jet_ind = np.where(input_data[:, 22] == 1)  # in which case we neglect some columns
    drop_columns_1 = np.array([4, 5, 6, 12, 22, 26, 27, 28])
    data_1 = input_data[one_jet_ind[0], :]
    data_1 = np.delete(data_1, drop_columns_1, axis=1)
    data_1 = standardize(data_1)
    data_1[np.where(data_1 > 3)] = 0
    data_1[np.where(data_1 < -3)] = 0
    y_1 = yb[one_jet_ind]
    id_1 = ids[one_jet_ind]
    data_1 = np.insert(data_1, 0, 1, axis=1)

    print("ok for 1")

    more_jet_ind = np.where(input_data[:, 22] > 1)  # in which case we neglect no columns
    data_2 = input_data[more_jet_ind[0], :]
    data_2 = np.delete(data_2, 22, axis=1)
    data_2 = standardize(data_2)
    data_2[np.where(data_2 > 3)] = 0
    data_2[np.where(data_2 < -3)] = 0
    y_2 = yb[more_jet_ind]
    id_2 = ids[more_jet_ind]
    data_2 = np.insert(data_2, 0, 1, axis=1)

    print("ok for 2")

    ys = [y_0, y_1, y_2]
    datas = [data_0, data_1, data_2]
    indexes = [id_0, id_1, id_2]

    return ys, datas, indexes


def standardize(x, mean=np.nan, std=np.nan):
    """Standardize data set."""
    if np.any(np.isnan(mean)) or np.any(np.isnan(std)):
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
    x = (x-mean) / std
    return x, mean, std


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(data @ weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def change999(tx):
    """Change the -999 to Nan"""
    tx[np.where(tx == -999)] = np.nan
    return tx


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def log_normalize(x):
    """Log-normalize the skewed features """
    skewed_indices = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29] ## The indices were obtain by looking at the data
    x[:, skewed_indices] = np.log1p(x[:, skewed_indices])
    return x


def data_process(x_train, x_test, degree):
    """Processing the data."""

    # Put -999 to nan
    x_train = change999(x_train)
    x_test = change999(x_test)

    # Log normalization
    x_train = log_normalize(x_train)
    x_test = log_normalize(x_test)

    # standardization
    x_train, mean, std = standardize(x_train)
    x_test, _, _ = standardize(x_test, mean, std)

    # Put nan to 0
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    # Feature expand to the polynomial of degree_max = degree
    x_train = buildpoly(x_train, degree)
    x_test = buildpoly(x_test, degree)

    return x_train, x_test
