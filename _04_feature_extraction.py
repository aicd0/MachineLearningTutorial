# feature extraction
# last update: 2021/10/5

import json
import math
import numpy as np
import os
import dependence.utils as utils

from collections import Counter
from scipy import stats
from _01_data2npz import output_path as input_path

output_path = 'outputs\\04_feature_extraction\\'

def renyi_entropy(data, alpha):
    return math.log(np.sum(data ** alpha)) / (1 - alpha)

def approximate_entropy(data, m, r=0.15):
    f = r * np.std(data, ddof=1)
    
    def phi(m):
        nonlocal data, f
        k = len(data) - m + 1
        x = np.empty((k, m))
        for i in range(k):
            x[i] = data[i : i + m]
        
        d = np.empty((k, k))
        for i in range(k):
            d[i] = np.abs(x[i] - x).max(axis=1)
                
        t = np.sum(d < f, axis=1) / k
        return np.average(np.log(t))
        
    return phi(m) - phi(m + 1)

def sample_entropy(data, m, r=0.15):
    f = r * np.std(data, ddof=1)

    def phi(m):
        nonlocal data, f
        k = len(data) - m + 1
        x = np.empty((k, m))
        for i in range(k):
            x[i] = data[i : i + m]
        
        d = np.empty((k, k))
        for i in range(k):
            d[i] = np.abs(x[i] - x).max(axis=1)
                
        t = np.sum(d < f, axis=1) / (k - 1)
        return np.average(t)
        
    return math.log(phi(m) / phi(m + 1))

def permutation_entropy(data, m, step=1):
    if step > m:
        step = m
    if step == 1:
        k = int(len(data) - m + 1)
    else:
        k = int((len(data) - m + 1) / step) + 1

    x = np.empty((k, m))
    for i in range(k):
        x[i] = data[i * step : i * step + m]

    index = np.argsort(x, axis=1)
    index_str = [str(i)[1 : -1] for i in index]
    count = Counter(index_str)
    pe = 0
    
    for val in count.values():
        p = val / k
        pe -= p * math.log(p)

    return pe / math.log(math.factorial(m))

def main():
    utils.check_npz_files()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in os.listdir(input_path):
        input_file_path = input_path + file_name
        output_file_path = output_path + utils.get_display_name(file_name) + '.txt'
        data = np.load(input_file_path)['data']

        size = 3000
        all_features = []

        for i in range(0, len(data), size):
            data_trimmed = data[i : i + size]
            if len(data_trimmed) != size:
                continue

            features = {}
            features['start'] = i
            # Skewness
            features['skew'] = stats.skew(data_trimmed)
            # Kurtosis
            features['kurtosis'] = stats.kurtosis(data_trimmed)
            # Renyi Entropy
            features['re_en'] = renyi_entropy(data_trimmed, alpha=2)
            # Approximate Entropy
            features['ap_en'] = approximate_entropy(data_trimmed, m=6)
            # Sample Entropy
            features['sa_en'] = sample_entropy(data_trimmed, m=6)
            # Permutation Entropy
            features['pe_en'] = permutation_entropy(data_trimmed, m=6, step=1)
            all_features.append(features)
            print('start=%d' % i)

        with open(output_file_path, 'w') as f:
            json.dump(all_features, f)
        print(output_file_path)

if __name__ == '__main__':
    main()