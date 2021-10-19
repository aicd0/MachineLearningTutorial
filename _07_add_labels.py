import json
import math
import numpy as np
import os
import dependence.utils as utils

from _04_combine import output_file_path as input_file_path
from _06_k_means import k_means

output_path = 'outputs\\07_add_labels\\'
output_file = 'labels.txt'
output_file_path = output_path + output_file

def data_descretization(data, ratio) -> list:
    if len(data.shape) != 1:
        raise ValueError()

    sample_count = data.shape[0]

    if not isinstance(ratio, list):
        raise TypeError()
    if len(ratio) == 0 or sum(ratio) >= 1:
        raise ValueError()

    for i in range(1, len(ratio)):
        ratio[i] += ratio[i - 1]
    labels = np.full((sample_count), len(ratio), dtype=np.uint32)

    for i in range(len(ratio)):
        target = ratio[i] * sample_count
        rmin = 0
        r2 = 1

        while (data <= r2).sum() < target:
            rmin = r2
            r2 *= 2
        rmax = r2

        while True:
            r2 = (rmin + rmax) / 2
            cres = data <= r2
            scnt = cres.sum()

            if rmax - rmin >= 1e-5:
                if scnt > target:
                    rmax = r2
                    continue
                elif scnt < target:
                    rmin = r2
                    continue

            labels -= cres
            break

    return labels.tolist()

def hypersphere_labels(data, center=None, ratio: list=[0.5], w=None) -> list:
    sample_size = data.shape[1]

    if utils.is_none(center):
        c_average = data.mean(axis=0)
        c_std = data.std(axis=0, ddof=1)
        center = np.random.normal(c_average, c_std, (sample_size))
    if center.shape != (sample_size,):
        raise ValueError()

    if utils.is_none(w):
        w = 1 / data.var(axis=0)
    if w.shape != (sample_size,):
        raise ValueError()

    dif = data - center
    data_r2 = (dif * dif * w).sum(axis=1)
    
    return data_descretization(data_r2, ratio)

def custom_labels(data, ratio: list=[0.5]) -> list:
    dmin = data.min(axis=1, keepdims=True)
    dmax = data.max(axis=1, keepdims=True)
    val = np.sin(2 * math.pi * (data - dmin) / (dmax - dmin)).sum(axis=1)
    return data_descretization(val, ratio)

def main():
    utils.check_04()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = np.load(input_file_path)['data']
    size = data.shape[0]
    step = data.shape[1]
    
    k_means_labels = k_means(data, 2)
    hsphere_labels = hypersphere_labels(data)
    c_labels = custom_labels(data)

    labels = ['start', 'km', 'hs', 'cs']
    all_labels = []
    for i in range(size):
        current_labels = []
        current_labels.append(i * step)
        current_labels.append(k_means_labels[i])
        current_labels.append(hsphere_labels[i])
        current_labels.append(c_labels[i])
        all_labels.append(current_labels)

    with open(output_file_path, 'w') as f:
        json.dump([labels, all_labels], f)
    print(output_file_path)

if __name__ == '__main__':
    main()