import json
import numpy as np
import os
import dependence.utils as utils

from _05_feature_extraction import output_file_path as features_file
from _07_add_labels import output_file_path as labels_file

output_path = 'outputs\\08_decision_tree\\'
output_file = 'results.txt'
output_file_path = output_path + output_file

def main():
    utils.check_05()
    utils.check_07()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(features_file, 'r') as f:
        features_src = json.load(f)
    with open(labels_file, 'r') as f:
        labels_src = json.load(f)

    for i, label in enumerate(labels_src[0]):
        if label == 'start':
            continue
        labels = np.array([v[i] for v in labels_src[1]], dtype=np.uint32)

        for j, question in enumerate(features_src[0]):
            if question == 'start':
                continue
            answers = np.array([v[j] for v in features_src[1]])


    results = {}
    results['km'] = [['skew', 5.0], ['re_en', 3.5]]

    with open(output_file_path, 'w') as f:
        json.dump(results, f)
    print(output_file_path)

if __name__ == '__main__':
    main()