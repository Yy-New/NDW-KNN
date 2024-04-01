import os

from ImputeTest.KNNImputation import knn_imputation, ns_knn_imputation
from utils import get_csv_data, is_folder_exists, merge_heading

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
                    class_label, metabolites, missing_data = get_csv_data(miss_file_path)
                    fill_dir_path = f"../KNNFillData/{file}/{distance_list[dist]}/{PercentMissing}/{num_file}"
                    is_folder_exists(fill_dir_path)

                    knn_data = knn_imputation(missing_data.T, 6, dist + 1).T
                    tmp_knn_data = merge_heading(class_label, metabolites, knn_data)
                    knn_save_data_path = fill_dir_path + '/knn_k=6_' + miss_file
                    tmp_knn_data.to_csv(knn_save_data_path, header=None, index=False)
                    print(f"{distance_list[dist]}: {num_file} {PercentMissing} {file} {miss_file} KNN_k=6 Filling is complete.")

                    knn_data = ns_knn_imputation(missing_data.T, 6, dist + 1).T
                    tmp_knn_data = merge_heading(class_label, metabolites, knn_data)
                    knn_save_data_path = fill_dir_path + '/ns_knn_k=6_' + miss_file
                    tmp_knn_data.to_csv(knn_save_data_path, header=None, index=False)
                    print(f"{distance_list[dist]}: {num_file} {PercentMissing} {file} {miss_file} NS_KNN_k=6 Filling is complete.")



