import copy
import os

import numpy as np

from ImputeTest.IDWKNNImputation import classify_normal_preprocessing
from ImputeTest.IDWUtils import dynamic_weight_KNN
from utils import get_csv_data, is_folder_exists, merge_heading, nan_distance

MM_path = "../MMData"
distance_list = ["ManhattanDistance", "EuclideanDistance", "ChebyshevDistance"]
file_list = os.listdir(MM_path)

for dist in range(3):
    for file in file_list:
        PercentMissing_list_path = f"{MM_path}/{file}"
        PercentMissing_list = os.listdir(PercentMissing_list_path)
        for PercentMissing in PercentMissing_list:
            num_file_list_path = f"{PercentMissing_list_path}/{PercentMissing}"
            num_file_list = os.listdir(num_file_list_path)
            for num_file in num_file_list:
                miss_file_list_path = f"{num_file_list_path}/{num_file}"
                miss_file_list = os.listdir(miss_file_list_path)
                for miss_file in miss_file_list:
                    if "target" in miss_file:
                        continue
                    miss_file_path = f"{miss_file_list_path}/{miss_file}"
                    class_label, metabolites, miss_data = get_csv_data(miss_file_path)
                    tmp_miss_file_path = miss_file_path.replace("NormalizationMMData", "MMData")
                    tmp_miss_data = get_csv_data(tmp_miss_file_path)[2].values.astype(float)
                    means = np.nanmean(tmp_miss_data, axis=1)
                    stds = np.nanstd(tmp_miss_data, axis=1)
                    lower_limit = -(means/stds)
                    fill_dir_path = f"../IDWKNNFillData/{file}/{distance_list[dist]}/{PercentMissing}/{num_file}"
                    is_folder_exists(fill_dir_path)
                    miss_data = miss_data.T
                    miss_data = miss_data.values.astype(float)
                    origin_miss_data = copy.deepcopy(miss_data)

                    nan_euclidean_distance = nan_distance(miss_data, miss_data, dist + 1)
                    miss_data = classify_normal_preprocessing(miss_data.T, nan_euclidean_distance, lower_limit)

                    knn_data = dynamic_weight_KNN(miss_data.T, nan_euclidean_distance, origin_miss_data, 1).T
                    tmp_knn_data = merge_heading(class_label, metabolites, knn_data)
                    knn_save_data_path = fill_dir_path + f'/dynamic_weight_knn_' + miss_file
                    tmp_knn_data.to_csv(knn_save_data_path, header=None, index=False)
                    print(f"{file}: {num_file} {PercentMissing} {distance_list[dist]} {miss_file} Filling is complete.")


