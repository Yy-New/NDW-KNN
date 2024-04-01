import copy
import random
import numpy as np
from ImputeTest.IDWUtils import get_dynamic_k, dynamic_weight_KNN


def classify_impute_data(miss_data, sample_indices):
    min_non_nan = np.nanmin(miss_data)
    data_filled = np.where(np.isnan(miss_data), np.nanmin(miss_data), miss_data)
    nan_indexes = np.where(np.isnan(miss_data))[0]
    tmp_k = []
    for _ in nan_indexes:
        tmp_k.append(sample_indices[_])
    flattened_array = np.concatenate(tmp_k)
    result_array = np.unique(flattened_array)
    mcar_index, mnar_index = [], []
    for _ in nan_indexes:
        if _ in result_array:
            mnar_index.append(_)
    for i in range(len(tmp_k)):
        for j in range(len(tmp_k[i])):
            if tmp_k[i][j] in nan_indexes and nan_indexes[i] not in mnar_index:
                mnar_index.append(nan_indexes[i])
    for _ in nan_indexes:
        if _ not in mnar_index:
            mcar_index.append(_)
    num = 1000
    mean_estimation = np.nanmean(data_filled)
    std_estimation = np.nanstd(data_filled)
    norm_dist = np.random.normal(mean_estimation, std_estimation, num)
    min_filtered_list = norm_dist[(norm_dist < min_non_nan) & (0 < norm_dist)].tolist()
    if min_non_nan == 0:
        min_filtered_list = [min_non_nan] * len(mnar_index)
    else:
        while len(min_filtered_list) < len(mnar_index):
            num *= 10
            norm_dist = np.random.normal(mean_estimation, std_estimation, num)
            min_filtered_list = norm_dist[(norm_dist < min_non_nan) & (0 < norm_dist)].tolist()
    selected_numbers = random.sample(min_filtered_list, len(mnar_index))
    for i, idx in enumerate(mnar_index):
        data_filled[idx] = selected_numbers[i]
    norm_dist = np.random.normal(mean_estimation, std_estimation, 1000)
    max_filtered_list = norm_dist[(norm_dist > min_non_nan)].tolist()
    selected_numbers = random.sample(max_filtered_list, len(mcar_index))
    for i, idx in enumerate(mcar_index):
        data_filled[idx] = selected_numbers[i]
    return data_filled


def classify_normal_preprocessing(miss_data, nan_euclidean_distance):
    fill_data = []
    dynamic_k = get_dynamic_k(nan_euclidean_distance)
    sorted_indices = np.argsort(nan_euclidean_distance, axis=1)
    sample_indices = []
    for _ in range(len(dynamic_k)):
        tmp = sorted_indices[_, 1:dynamic_k[_]+1]
        sample_indices.append(tmp)
    for i in range(len(miss_data)):
        nan_indexes = np.where(np.isnan(miss_data[i]))[0]
        if len(nan_indexes)==0:
            fill_data.append(miss_data[i])
            continue
        tmp_item = copy.deepcopy(miss_data[i])
        item = classify_impute_data(tmp_item, sample_indices)
        fill_data.append(item)
    return np.array(fill_data)


def dynamic_weight_knn_imputation(miss_data, origin_miss_data, nan_euclidean_distance, p):
    imputed_data = dynamic_weight_KNN(miss_data, nan_euclidean_distance, origin_miss_data, p)
    return imputed_data.T

