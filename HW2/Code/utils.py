import os
import numpy as np
import pandas as pd

from optim.Optimizer import *


def accuracy(h, y):
    """
    h : (N, ), predicted label
    y : (N, ), correct label
    """

    total = h.shape[0]
    correct = len(np.where(h==y)[0])

    acc = correct / total

    return acc


def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def z_score (x) :
    x_ = pd.DataFrame()
    for i in range(x.shape[1]) :
        x_col = x.iloc[:,i]
        # ========================= EDIT HERE =========================
        '''
        Edit here to do z_score normalization.
        '''
        
        x_col = (x_col - x_col.mean()) / x_col.std()

        # ========================= EDIT HERE =========================
        x_[i] = x_col

    return x_.values[:, :].astype(np.float32)


def min_max (x) :
    x_ = pd.DataFrame()
    for i in range(x.shape[1]) :
        x_col = x.iloc[:,i]
        # ========================= EDIT HERE =========================
        '''
        Edit here to do min_max normalization.
        '''
        
        x_col = (x_col - x_col.min()) / (x_col.max() - x_col.min())

        # ========================= EDIT HERE =========================
        x_[i] = x_col

    return x_.values[:, :].astype(np.float32)


def BanknoteData(path, filename):
    df = pd.read_csv(os.path.join(path, filename))

    x = z_score(df.iloc[:, :-1])

    # ========================= EDIT HERE =========================
    '''
    z_score normalization / min_max normalization
    '''
    x = z_score(df.iloc[:, :-1])
    # x = min_max(df.iloc[:, :-1])

    # ========================= EDIT HERE =========================
    y = df.values[:, -1].astype(np.int32)


    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((bias, x), axis=1)
    
    return x, y


def IrisData(path, filename):
    df = pd.read_csv(os.path.join(path, filename))

    # ========================= EDIT HERE =========================
    '''
    z_score normalization / min_max normalization
    '''
    x = z_score(df.iloc[:, :-1])
    # x = min_max(df.iloc[:, :-1])

    # ========================= EDIT HERE =========================

    y = df.values[:, -1].astype(np.int32)
    
    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((bias, x), axis=1)
    
    return x, y


data_dir = {
    'Banknote': 'banknote',
    'Iris': 'iris',
}


def load_data(data_name):
    dir_name = data_dir[data_name]
    path = os.path.join('./data', dir_name)

    if data_name == 'Banknote':
        train_x, train_y = BanknoteData(path, 'train.csv')
        test_x, test_y = BanknoteData(path, 'test.csv')
    elif data_name == 'Iris':
        train_x, train_y = IrisData(path, 'train.csv')
        test_x, test_y = IrisData(path, 'test.csv')
    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y)
