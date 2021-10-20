import json
import numpy as np
import os
import random
import dependence.utils as utils

from _04_combine import output_file_path as input_file_path

output_path = 'outputs\\06_k_means\\'
output_file = 'class.txt'
output_file_path = output_path + output_file

def k_means(data, class_count) -> list:
    data_count = data.shape[0]
    dims = data.shape[1]
    if data_count < class_count:
        raise ValueError()

    # initialize class center randomly
    class_center = np.empty((class_count, dims))
    order = [i for i in range(data_count)]
    random.shuffle(order)
    for i in range(class_count):
        class_center[i] = data[order[i]]

    class_results = np.zeros((data_count), dtype=np.uint8)

    while True:
        old_class_results = np.copy(class_results)
        class_results = np.zeros((data_count), dtype=np.uint8)

        # reclassification for each sample
        for i in range(data_count):
            dist = np.linalg.norm(data[i] - class_center, axis=1)
            class_results[i] = dist.argmin()
        
        # stops iteration if nothing changed
        if np.array_equal(old_class_results, class_results):
            break
        
        # update class centers
        class_center = np.zeros((class_count, dims))
        class_counts = np.zeros((class_count), dtype=np.uint32)
        for i in range(data_count):
            class_center[class_results[i]] += data[i]
            class_counts[class_results[i]] += 1
        class_counts = class_counts.reshape((class_count, 1))
        class_center /= class_counts
    
    return class_results.tolist()

def main():
    utils.check_04()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    data = np.load(input_file_path)['data']
    results = k_means(data, 2)

    with open(output_file_path, 'w') as f:
        json.dump(results, f)
    print(output_file_path)

if __name__ == '__main__':
    main()