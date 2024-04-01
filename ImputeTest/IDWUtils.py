import copy
import numpy as np

def weight_function(dist, p):
    sum = 0
    for item in dist:
        sum += 1 / (item ** p)
    dist_weight = []
    for item in dist:
        dist_weight.append((1/(item ** p)) / sum)
    return dist_weight


def get_dynamic_k(dist):
    n = dist.shape[0]
    temp_dist = np.sort(dist, axis=1)
    dynamic_k = []
    for i in range(n):
        sum = 0
        for j in range(2, n):
            sum += temp_dist[i][j] - temp_dist[i][1]
        sum /= ((n-2)*np.sqrt(len(dist)))
        cnt = 1
        for j in range(2, n):
            if temp_dist[i][j] - temp_dist[i][1] < sum:
                cnt += 1
        dynamic_k.append(cnt)
    return dynamic_k


def dynamic_weight_KNN(x, dist, origin_miss_data, p):
    dynamic_k = get_dynamic_k(dist)
    temp_x = copy.deepcopy(x)
    sorted_indices = np.argsort(dist, axis=1)
    for i in range(len(origin_miss_data)):
        nan_j = np.isnan(origin_miss_data[i])
        if not nan_j.any():
            continue
        for j in np.where(nan_j)[0]:
            temp_sorted_indices = sorted_indices[i, 1:]
            top_k_indices = temp_sorted_indices[~np.isnan(x[temp_sorted_indices, j])][:dynamic_k[i]]
            temp_dist = dist[i, top_k_indices]
            temp_weight = weight_function(temp_dist, p)
            temp_x[i, j] = np.sum(x[top_k_indices, j] * temp_weight)
    return temp_x
