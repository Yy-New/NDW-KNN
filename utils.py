import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def get_csv_data(data_url):
    data = pd.read_csv(data_url, header=None)
    class_label = data.iloc[0:2]
    metabolites = data.iloc[2:, 0]
    rest_of_data = data.iloc[2:, 1:]
    return class_label, metabolites, rest_of_data


def get_excel_data(data_url):
    data = pd.read_excel(data_url, header=None)
    class_label = data.iloc[0:2]
    metabolites = data.iloc[2:, 0]
    rest_of_data = data.iloc[2:, 1:]
    return class_label, metabolites, rest_of_data


def is_folder_exists(save_path_dir):
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)


def merge_heading(class_label, metabolites, data):
    tmp_data = np.insert(data, 0, metabolites.tolist(), axis=1)
    df = pd.DataFrame(tmp_data)
    df = pd.concat([class_label, df], axis=0).reset_index(drop=True)
    return df


def get_NRMSE(x_true, x_pred):
    x_true = x_true.astype('float')
    x_pred = x_pred.astype('float')
    mse = mean_squared_error(x_true, x_pred)
    rmse = np.sqrt(mse)
    mean_std = np.mean(x_true.std())
    return rmse / mean_std


def get_MAPE(x_true, x_pred):
    x_true = x_true.astype('float')
    x_pred = x_pred.astype('float')
    absolute_percentage_errors = np.abs((x_pred - x_true) / x_true)
    return np.mean(absolute_percentage_errors)


def nan_permutation(x):
    new_data = []
    max_num = 0
    for item in x:
        nan_data = []
        tmp_data = []
        nan_count = 0
        for val in item:
            if np.isnan(val):
                nan_data.append(val)
                nan_count += 1
            else:
                tmp_data.append(val)
        new_data.append(tmp_data + nan_data)
        max_num = max(max_num, nan_count)
    return new_data, len(x[0]) - max_num


def get_complete_subset_data(data):
    data = data.values.astype("float")
    data, max_num = nan_permutation(data)
    data = pd.DataFrame(data).iloc[:, :max_num]
    return data


def nan_distance(x, y, p):
    n, m = x.shape
    ans = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            nan_mask = ~np.isnan(x[i]) & ~np.isnan(y[j])
            num_cnt = np.sum(nan_mask)
            if num_cnt > 0:
                diff = np.abs(x[i, nan_mask] - y[j, nan_mask]) ** p
                sum_diff = np.sum(diff)
                ans[i, j] = (m / num_cnt * sum_diff) ** (1 / p)
                ans[j, i] = ans[i, j]
    return ans

