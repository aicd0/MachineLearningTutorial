import json
import matplotlib.pylab as plt
import numpy as np
import os
import random
import dependence.utils as utils

from _04_combine import output_file_path as input_file_path
from _06_add_labels import output_file_path as labels_file

output_path = 'outputs\\07_k_means\\'

def k_means(data, k) -> list:
    data_count = data.shape[0]
    dims = data.shape[1]
    if data_count < k:
        raise ValueError()

    # initialize class centers randomly
    centers = np.empty((k, dims))
    order = [i for i in range(data_count)]
    random.shuffle(order)
    for i in range(k):
        centers[i] = data[order[i]]

    results = np.zeros((data_count), dtype=np.uint8)

    while True:
        old_results = np.copy(results)
        results = np.zeros((data_count), dtype=np.uint8)

        # reclassification for each sample
        for i in range(data_count):
            dist = np.linalg.norm(data[i] - centers, axis=1)
            results[i] = dist.argmin()
        
        # stops iteration if nothing changed
        if np.array_equal(old_results, results):
            break
        
        # update class centers
        centers = np.zeros((k, dims))
        sizes = np.zeros((k), dtype=np.uint32)
        for i in range(data_count):
            centers[results[i]] += data[i]
            sizes[results[i]] += 1
        sizes = sizes.reshape((k, 1))
        centers /= sizes
    
    return results.tolist()

def v_measure(new_labels, old_labels, beta=1):
    new_labels = np.array(new_labels, np.uint32)
    old_labels = np.array(old_labels, np.uint32)

    if new_labels.shape != old_labels.shape:
        raise ValueError()
    if len(new_labels.shape) != 1:
        raise ValueError()

    data_count = len(new_labels)
    new_categories = np.max(new_labels) + 1
    old_categories = np.max(old_labels) + 1
    new_to_old = np.empty((new_categories), np.uint32)

    # calculate precision
    precision = np.empty((new_categories))
    
    for new_label in range(new_categories):
        votes = np.zeros((old_categories), np.uint32)

        for i in range(data_count):
            if new_labels[i] == new_label:
                votes[old_labels[i]] += 1

        new_to_old[new_label] = old_label = np.argmax(votes)
        precision[new_label] = votes[old_label] / votes.sum()
    
    # calculate recall
    correct = np.zeros((old_categories), np.uint32)
    total = np.zeros((old_categories), np.uint32)

    for i in range(data_count):
        predict = new_to_old[new_labels[i]]
        expect = old_labels[i]
        total[expect] += 1

        if predict == expect:
            correct[expect] += 1

    recall = correct / total

    # calculate v
    p = precision.mean()
    r = recall.mean()
    v = (1 + beta * beta) * p * r / (beta * beta * p + r)
    return v

def main():
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    utils.check_04()
    data_all = np.load(input_file_path)['data']
    
    utils.check_06()
    with open(labels_file, 'r') as f:
        labels_src = json.load(f)

    label_titles = labels_src[0][1:]
    label_count = len(label_titles)
    labels_all = [i[1:] for i in labels_src[1]]
    data_all, labels_all = utils.shuffle(zip(data_all, labels_all))
    data_all = np.array(data_all)

    k_start = 2
    k_end = 30
    l_k_v = np.empty((label_count, k_end - k_start + 1))

    for k in range(k_start, k_end + 1):
        new_labels = k_means(data_all, k)
        output_file = 'k=%d.txt' % k
        output_file_path = output_path + output_file

        with open(output_file_path, 'w') as f:
            json.dump(new_labels, f)
        print(output_file_path)

        for l in range(label_count):
            old_labels = np.array([v[l] for v in labels_all], dtype=np.uint32)
            l_k_v[l][k - k_start] = v_measure(new_labels, old_labels)
    
    # plot
    fig = plt.figure()

    for l, label_title in enumerate(label_titles):
        ax = fig.add_subplot(label_count, 1, l + 1)
        ax.set_title(label_title)
        plt.xlabel('k')
        plt.ylabel('v')
        plt.plot(l_k_v[l], color='r')

    fig.tight_layout()
    output_file_path = output_path + 'results.png'
    plt.savefig(output_file_path)
    print(output_file_path)
    plt.close('all')

if __name__ == '__main__':
    main()