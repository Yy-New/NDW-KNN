import os

import numpy as np
import pandas as pd

from utils import get_csv_data, get_NRMSE, get_excel_data, is_folder_exists

distance_list = ["ManhattanDistance", "EuclideanDistance", "Third-order Minkowski Distance"]

FillData_path = "../IDWKNNFillData"
RealData_path = "../LogPublicData"
file_list = os.listdir(FillData_path)
for file in file_list:
    real_data = get_excel_data(RealData_path + '/' + file + '.xlsx')[2]
    nan_mask = real_data.isnull()
    dist_fill_data_path = f"{FillData_path}/{file}"
    dist_fill_data_list = os.listdir(dist_fill_data_path)
    for dist_fill_data in dist_fill_data_list:
        PercentMissing_list_path = f"{dist_fill_data_path}/{dist_fill_data}"
        PercentMissing_list = os.listdir(PercentMissing_list_path)
        for PercentMissing in PercentMissing_list:
            num_file_list_path = f"{PercentMissing_list_path}/{PercentMissing}"
            num_file_list = os.listdir(num_file_list_path)
            for num_file in num_file_list:
                fill_file_list_path = f"{num_file_list_path}/{num_file}"
                fill_file_list = os.listdir(fill_file_list_path)
                fill_file_list_path = f"{num_file_list_path}/{num_file}"
                fill_file_list = os.listdir(fill_file_list_path)
                NRMSE_prop_list = []
                NRMSE_list = []
                methods = []
                index = -1
                temp_NRMSE_list = []
                for fill_file in fill_file_list:
                    method_name = fill_file.split(f"_{file}_")[0]
                    miss_prop = float(fill_file.split(f"_MNAR_")[1].split('%')[0])
                    if miss_prop not in NRMSE_prop_list:
                        NRMSE_prop_list.append(miss_prop)
                    fill_file_path = fill_file_list_path + '/' + fill_file
                    fill_data = get_csv_data(fill_file_path)[2]
                    tmp_real_data = real_data.where(~nan_mask, fill_data)
                    NRMSE_result = get_NRMSE(tmp_real_data, fill_data)
                    if method_name not in methods:
                        if len(temp_NRMSE_list) != 0:
                            NRMSE_list.append(temp_NRMSE_list)
                        methods.append(method_name)
                        temp_NRMSE_list = []
                    temp_NRMSE_list.append(NRMSE_result)
                NRMSE_list.append(temp_NRMSE_list)
                NRMSE_list = pd.DataFrame(NRMSE_list, index=methods, columns=NRMSE_prop_list).sort_index(axis=1)
                save_dir_path = f"{FillData_path.replace('..', '../Results/NRMSE')}/{file}/{dist_fill_data}/{PercentMissing}"
                is_folder_exists(save_dir_path)
                save_path = f"{save_dir_path}/{file}_NRMSE_{num_file}.xlsx"
                NRMSE_list.to_excel(save_path)
                print(f"{file}: {dist_fill_data} {PercentMissing} {num_file} Processing completed.")
