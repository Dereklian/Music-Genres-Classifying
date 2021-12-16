from copy import deepcopy
import numpy as np


def calc_euclidean_distance(a, b, d):
    result = 0
    for i in range(d):
        result = result + ((a[i] - b[i]) ** 2)
    result = result ** 0.5
    return result


def find_stray(centroids, input_data, dimension):
    result = np.zeros(dimension, dtype=np.int)
    farthest = 0
    for point in input_data:
        temp_distance = 0
        for i in range(len(centroids)):
            temp_distance = temp_distance + calc_euclidean_distance(centroids[i], point, dimension)
        if temp_distance > farthest:
            farthest = temp_distance
            result = point
    return result


def run_kmeans(data, group_sum, dimension):
    iter_rounds_total = 30

    # initialize the first random center
    random_cent = np.random.randint(len(data) - 1)
    centroids_list = np.array([data[random_cent]])
    points_list = [[]]
    print("Finding the origin centroids...")
    for i in range(group_sum - 1):      # -1 as the find centroid is already found
        k = find_stray(centroids_list, data, dimension)     # find the farthest point under the current situation
        centroids_list = np.concatenate([centroids_list, np.array([k])])
        points_list.append([])
        print("Centroids found:", i + 1, ",",group_sum - i - 1, "to go.")

    print("\n\nIterative optimization begins:")
    # iteration
    candidate_list = deepcopy(points_list)
    progress = " | "
    for i in range(iter_rounds_total):
        progress = progress + "□"
        print(str(round(((i / iter_rounds_total) * 100), 1)).zfill(4) + "%" + progress)

        for point in data:
            closest_index = 0
            closest = calc_euclidean_distance(centroids_list[closest_index], point, dimension)
            for j in range(1, len(centroids_list)):
                if calc_euclidean_distance(centroids_list[j], point, dimension) < closest:
                    closest = calc_euclidean_distance(centroids_list[j], point, dimension)
                    closest_index = j
            candidate_list[closest_index].append(point)
        for k in range(0, len(centroids_list)):
            if iter_rounds_total - 1 == i:
                break
            if len(candidate_list[k]) != 0:
                centroids_list[k] = np.mean(candidate_list[k], axis=0)
                candidate_list[k] = []

    progress = progress + "□"
    print("100% " + progress)
    print("Read a " + str(dimension) + "-dimension data")
    print("After clustering, obtained a set with shape: " + str(centroids_list.shape))

    return centroids_list
