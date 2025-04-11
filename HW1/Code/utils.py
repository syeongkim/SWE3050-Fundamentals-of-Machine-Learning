import numpy as np
import os
from optim.Optimizer import *
import pandas as pd

def RMSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse

def optimizer(optim_name, gamma, threshold=10):
    if optim_name == 'SGD':
        optim = SGD(threshold)
    elif optim_name == 'Momentum':
        optim = Momentum(gamma, threshold)
    # 5번 문제(Mini Kaggle challenge)를 위해 추가함
    elif optim_name == 'Adam':
        optim = Adam(threshold=threshold)
    elif optim_name == 'Adagrad':
        optim = Adagrad()
    elif optim_name == 'RMSprop':
        optim = RMSProp()
    else:
        raise NotImplementedError
    return optim


def load_data(data_name, normalize):
    path = os.path.join('./data', data_name)

    if data_name == 'CCPP':
        train_x, train_y = CCPPData(path, 'train.csv', normalize)
        test_x, test_y = CCPPData(path, 'test.csv', normalize)

    elif data_name == 'Airbnb':
        train_x, train_y = AirbnbData(path, 'train.csv', normalize)
        test_x, test_y = AirbnbData(path, 'test.csv', normalize)
    elif data_name == 'Wine':
        train_x, train_y = WineData(path, 'train.csv', normalize)
        test_x, test_y = WineData(path, 'test.csv', normalize)

    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y)

def load_data_with_feature_select(data_name, normalize, selected_feature=None):
    path = os.path.join('./data', data_name)

    if data_name == 'Boston':
        train_x, train_y = BostonData(path, 'train.csv', normalize, selected_feature)
        test_x, test_y = BostonData(path, 'test.csv', normalize, selected_feature)
    elif data_name == 'Traffic':
        train_x, train_y = TrafficData(path, 'train.csv', normalize, selected_feature)
        test_x, test_y = TrafficData(path, 'test.csv', normalize, selected_feature)
    elif data_name == 'Wine':
        train_x, train_y = WineData(path, 'train.csv', normalize)
        test_x, test_y = WineData(path, 'test.csv', normalize)

    else:
        raise NotImplementedError
    
    return (train_x, train_y), (test_x, test_y)


def AirbnbData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)

def RealEstateData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)

def CCPPData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)

def WineData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)


def TrafficData(path, filename, normalize, selected_feature):
    if filename == "train.csv":
        return load_reg_data(path, filename, target_at_front=False, normalize=normalize, selected_feature=selected_feature)
    if filename == "test.csv":
        return load_reg_data(path, filename, target_at_front=False, normalize=normalize, selected_feature=selected_feature)


def BostonData(path, filename, normalize, selected_feature):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize, selected_feature=selected_feature)


def load_reg_data(path, filename, target_at_front, normalize=None, shuffle=False, selected_feature=None):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    if selected_feature is not None:
        # 헤더에서 컬럼명과 인덱스 매핑
        existing_columns = {col: idx for idx, col in enumerate(lines[0])}
        column_indices = []
        for col in selected_feature:
            index = existing_columns.get(col, 0)
            if index == 0 and col not in existing_columns:
                print(f"⚠️ Warning: Column '{col}' not found in the {fullpath}.")
            column_indices.append(index)
        lines = [[row[i] for i in column_indices] for row in lines]

    data = lines[1:]

    data = np.array([[float(f) for f in d] for d in data], dtype=np.float64)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data[:, :-1], data[:, -1]

    num_data = x.shape[0]
    if normalize == 'MinMax':
        # ====== EDIT HERE ======

        x_max = np.max(x, axis = 0)
        x_min = np.min(x, axis = 0)

        x = (x - x_min) / (x_max - x_min)
        
        # ========================
    elif normalize == 'ZScore':
        # ====== EDIT HERE ======
        
        x_mean = np.mean(x, axis = 0)
        x_std = np.std(x, axis = 0)
        x = (x - x_mean) / x_std
        
        # =======================
    else:
        pass

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float64)
    x = np.concatenate((x, bias), axis=1)

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y

def Inference2SolutionCsv(inference, output_file_name):
    inference_df = pd.DataFrame(inference, columns=["traffic_volume"])
    inference_df.insert(0, 'ID', range(0, len(inference_df)))
    inference_df.to_csv(output_file_name, index=False, float_format="%.6f")
    return

