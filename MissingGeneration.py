import os
import random
import numpy as np
import pandas as pd
from utils import is_folder_exists, get_excel_data


def MM_generate(data, mis_prop, alpha, beta, gamma, k, class_label, metabolites, file):
    data = data.values.astype(float)
    mean_concentrations = np.mean(data, axis=1)
    metabolites = metabolites.tolist()
    sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

    mv_num = round(data.size * mis_prop)
    low_abundance_num = round(beta * data.shape[0])
    mid_abundance_num = round(alpha * data.shape[0])
    low_abundance_metabolites = sorted_metabolites[:low_abundance_num]
    mid_abundance_metabolites = sorted_metabolites[low_abundance_num:mid_abundance_num]

    data_target = np.full(data.shape, "O", dtype='object')

    low_missing_num = round(data.shape[1] * gamma * 0.8)
    metabolite_random_low_missing_num = round(data.shape[1] * gamma)
    low_abundance_metabolites_index = []
    for metabolite in low_abundance_metabolites:
        if mv_num > low_missing_num:
            mv_num -= low_missing_num
        else:
            low_missing_num = mv_num
            mv_num = 0
        sort_index = np.argsort(data[metabolites.index(metabolite)])
        indices_missing = sort_index[:low_missing_num]
        kk = metabolites.index(metabolite)
        data[kk][indices_missing] = np.nan
        data_target[kk][indices_missing] = "MNAR"
        tmp = sort_index[low_missing_num:metabolite_random_low_missing_num]
        low_abundance_metabolites_index.append(tmp + kk * data[kk].shape[0])

    low_abundance_metabolites_index = [item for sublist in low_abundance_metabolites_index for item in sublist]
    random_low_missing_num = round(len(low_abundance_metabolites_index) * 0.5)
    if mv_num > random_low_missing_num:
        mv_num -= random_low_missing_num
    else:
        random_low_missing_num = mv_num
        mv_num = 0
    indices_missing = random.sample(low_abundance_metabolites_index, random_low_missing_num)
    data.flat[indices_missing] = np.nan
    data_target.flat[indices_missing] = "MNAR"

    gamma *= 0.5
    mid_missing_num = round(data.shape[1] * gamma * 0.8)
    metabolite_random_mid_missing_num = round(data.shape[1] * gamma)
    mid_abundance_metabolites_index = []
    for metabolite in mid_abundance_metabolites:
        if mv_num > mid_missing_num:
            mv_num -= mid_missing_num
        else:
            mid_missing_num = mv_num
            mv_num = 0
        sort_index = np.argsort(data[metabolites.index(metabolite)])
        indices_missing = sort_index[:mid_missing_num]
        kk = metabolites.index(metabolite)
        data[kk][indices_missing] = np.nan
        data_target[kk][indices_missing] = "MNAR"
        tmp = sort_index[mid_missing_num:metabolite_random_mid_missing_num]
        mid_abundance_metabolites_index.append(tmp + kk * data[kk].shape[0])

    mid_abundance_metabolites_index = [item for sublist in mid_abundance_metabolites_index for item in sublist]
    random_mid_missing_num = round(len(mid_abundance_metabolites_index) * 0.5)
    if mv_num > random_mid_missing_num:
        mv_num -= random_mid_missing_num
    else:
        random_mid_missing_num = mv_num
        mv_num = 0
    indices_missing = random.sample(mid_abundance_metabolites_index, random_mid_missing_num)
    data.flat[indices_missing] = np.nan
    data_target.flat[indices_missing] = "MNAR"

    non_nan_coordinates = np.column_stack(np.where(~np.isnan(data)))
    non_nan_flat = [item[0]*data.shape[1]+item[1] for item in non_nan_coordinates]
    mcar_missing = random.sample(non_nan_flat, mv_num)
    data.flat[mcar_missing] = np.nan
    data_target.flat[mcar_missing] = "MCAR"

    percent_of_MNAR = np.where(data_target=="MNAR")[0].size / round(data.size * mis_prop)
    percent_of_MNAR = round(percent_of_MNAR, 4)

    data = np.insert(data, 0, metabolites, axis=1)
    df = pd.DataFrame(data)
    df = pd.concat([class_label, df], axis=0).reset_index(drop=True)

    save_path_dir = f"./MMData/{file}/PercentMissing={mis_prop*100}%/{k}/"
    is_folder_exists(save_path_dir)
    df_path = save_path_dir + f"{file}_MNAR_{round(percent_of_MNAR*100, 2)}%.csv"
    df.to_csv(df_path, index=False, header=None)
    print(f"{df_path} 处理完成.")

    data_target = np.insert(data_target, 0, metabolites, axis=1)
    target_df = pd.DataFrame(data_target)
    target_df = pd.concat([class_label, target_df], axis=0).reset_index(drop=True)
    target_df.to_csv(save_path_dir + f"{file}_MNAR_{round(percent_of_MNAR*100, 2)}%_target.csv", index=False, header=None)


if __name__ == "__main__":
    dir_path = "./LogPublicData"
    files = os.listdir(dir_path)
    alpha = 0.7
    mis_prop_list = [0.1, 0.15, 0.2, 0.25, 0.3]
    gamma_list = [0.15, 0.225, 0.3, 0.375, 0.45]
    for item in files:
        file_name = item.split('.')[0]
        class_label, metabolites, rest_of_data = get_excel_data(dir_path + '/' + item)
        for i in range(len(mis_prop_list)):
            for k in range(1,11,1):
                for mnar_prop in range(50, 101, 5):
                    beta = ((mis_prop_list[i]*(mnar_prop/100))-0.315*gamma_list[i])/(0.45*gamma_list[i])
                    MM_generate(rest_of_data, mis_prop_list[i], alpha, beta, gamma_list[i], k, class_label, metabolites, file_name)
