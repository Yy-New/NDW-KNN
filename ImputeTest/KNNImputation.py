import copy
import numpy as np
import pandas as pd
from utils import nan_distance


def replace_min(x):
    x = np.array(x)
    min_values = np.nanmin(x, axis=0)
    nan_mask = np.isnan(x)
    x[nan_mask] = np.tile(min_values, (x.shape[0], 1))[nan_mask]
    return x


def _KNN(x, dist, k):
    temp_x = copy.deepcopy(x)
    sorted_indices = np.argsort(dist, axis=1)
    for i in range(len(x)):
        nan_j = np.isnan(x[i])
        if not nan_j.any():
            continue
        for j in np.where(nan_j)[0]:
            temp_sorted_indices = sorted_indices[i]
            top_k_indices = temp_sorted_indices[~np.isnan(x[temp_sorted_indices, j])][:k]
            temp_x[i, j] = np.average(x[top_k_indices, j])
    return temp_x


def NS_KNN(x, dist, k):
    temp_x = copy.deepcopy(x)
    temp_x = replace_min(temp_x)
    miss_x = copy.deepcopy(x)
    for i in range(len(x)):
        nan_j = np.isnan(x[i])
        if not nan_j.any():
            continue
        sorted_indices = np.argsort(dist[i])
        top_k_indices = sorted_indices[1: k+1]
        for j in np.where(nan_j)[0]:
            miss_x[i][j] = np.average(temp_x[top_k_indices, j])
    return miss_x


def knn_imputation(miss_data, k, dist_choice):
    miss_data = miss_data.values.astype(float)
    nan_euclidean_distance = nan_distance(miss_data, miss_data, dist_choice)
    imputed_data = _KNN(miss_data, nan_euclidean_distance, k)
    imputed_data = pd.DataFrame(imputed_data)
    return imputed_data


def ns_knn_imputation(miss_data, k, dist_choice):
    miss_data = miss_data.values.astype(float)
    nan_euclidean_distance = nan_distance(miss_data, miss_data, dist_choice)
    imputed_data = NS_KNN(miss_data, nan_euclidean_distance, k)
    imputed_data = pd.DataFrame(imputed_data)
    return imputed_data
