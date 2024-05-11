import os
import pandas as pd

from utils import get_csv_data, get_excel_data, is_folder_exists, get_MAPE


FillData_path = "../KNNFillData"
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
                MAPE_prop_list = []
                MAPE_list = []
                methods = []
                index = -1
                temp_MAPE_list = []
                for fill_file in fill_file_list:
                    method_name = fill_file.split(f"_{file}_")[0]
                    miss_prop = float(fill_file.split(f"_MNAR_")[1].split('%')[0])
                    if miss_prop not in MAPE_prop_list:
                        MAPE_prop_list.append(miss_prop)
                    fill_file_path = fill_file_list_path + '/' + fill_file
                    fill_data = get_csv_data(fill_file_path)[2]
                    tmp_real_data = real_data.where(~nan_mask, fill_data)
                    MAPE_result = get_MAPE(real_data, fill_data)
                    if method_name not in methods:
                        if len(temp_MAPE_list) != 0:
                            MAPE_list.append(temp_MAPE_list)
                        methods.append(method_name)
                        temp_MAPE_list = []
                    temp_MAPE_list.append(MAPE_result)
                MAPE_list.append(temp_MAPE_list)
                MAPE_list = pd.DataFrame(MAPE_list, index=methods, columns=MAPE_prop_list).sort_index(axis=1)
                Save_Evaluation_path = ""
                save_dir_path = f"{FillData_path.replace('..', '../Results/MAPE')}/{file}/{dist_fill_data}/{PercentMissing}"
                is_folder_exists(save_dir_path)
                save_path = f"{save_dir_path}/{file}_MAPE_{num_file}.xlsx"
                MAPE_list.to_excel(save_path)
                print(f"{file}: {dist_fill_data} {PercentMissing} {num_file} Processing completed.")
