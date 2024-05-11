import os
import numpy as np
import pandas as pd

from utils import is_folder_exists


method_path = "../Results/MAPE"
method_list = os.listdir(method_path)
for method in method_list:
    if "DW" in method:
        continue
    file_path = f"{method_path}/{method}"
    file_list = os.listdir(file_path)
    for file in file_list:
        dist_list_path = f"{file_path}/{file}"
        dist_list = os.listdir(dist_list_path)
        for dist in dist_list:
            PercentMissing_list_path = f"{dist_list_path}/{dist}"
            PercentMissing_list = os.listdir(PercentMissing_list_path)
            for PercentMissing in PercentMissing_list:
                nrmse_list_path = f"{PercentMissing_list_path}/{PercentMissing}"
                nrmse_list = os.listdir(nrmse_list_path)
                result_list = []
                for nrmse in nrmse_list:
                    nrmse_file_path = f"{nrmse_list_path}/{nrmse}"
                    NRMSE_result = pd.read_excel(nrmse_file_path, index_col=0)
                    result_list.append(NRMSE_result)
                average = np.zeros(result_list[0].shape)
                max_NRMSE = np.zeros(result_list[0].shape)
                min_NRMSE = np.full_like(result_list[0], np.inf)
                for result in result_list:
                    for i in range(result.shape[0]):
                        for j in range(result.shape[1]):
                            average[i][j] += result.values[i][j]
                            max_NRMSE[i][j] = np.maximum(result.values[i][j], max_NRMSE[i][j])
                            min_NRMSE[i][j] = np.minimum(result.values[i][j], min_NRMSE[i][j])
                average = average / len(result_list)
                max_NRMSE -= average
                min_NRMSE = average - min_NRMSE

                average = pd.DataFrame(average)
                average.index = result_list[0].index
                average.columns = result_list[0].columns
                save_path = f"{PercentMissing_list_path.replace('Result', 'AverageResult')}/"
                is_folder_exists(save_path)
                file_save_path = f"{save_path}/{file}_{PercentMissing}.xlsx"
                average.to_excel(file_save_path)

                max_NRMSE = pd.DataFrame(max_NRMSE)
                max_NRMSE.index = result_list[0].index
                max_NRMSE.columns = result_list[0].columns
                save_path = save_path.replace("MAPE", "MAPEError")
                is_folder_exists(save_path)
                file_save_path = f"{save_path}/max_{file}_{PercentMissing}.xlsx"
                max_NRMSE.to_excel(file_save_path)

                min_NRMSE = pd.DataFrame(min_NRMSE)
                min_NRMSE.index = result_list[0].index
                min_NRMSE.columns = result_list[0].columns
                file_save_path = f"{save_path}/min_{file}_{PercentMissing}.xlsx"
                min_NRMSE.to_excel(file_save_path)
                print(f"{file}: {dist} {PercentMissing} Processing completed.")

