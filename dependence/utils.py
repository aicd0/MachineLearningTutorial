import os
import random
import threading
import time
import dependence.global_config as global_config

from typing import Any, Tuple

# random
def shuffle(x: zip) -> Tuple:
    x = list(x)
    random.shuffle(x)
    return zip(*x)

# type
def is_none(obj) -> bool:
    return type(obj) == type(None)

# string
def get_file_name(path: str) -> str:
    idx = max([path.rfind(s) for s in ['/', '\\']])
    return path[idx + 1 :]

def get_display_name(path: str) -> str:
    path = get_file_name(path)
    idx = path.rfind('.')
    if idx == -1:
        return path
    return path[0 : idx]

# inputs and outputs
__atomic_print_and_log_mutex = threading.Lock()

def __log(text: str, time_stamp: bool, file: str):
    if not os.path.exists(file):
        open(file, 'w').close()
    with open(file, 'a+') as f:
        if time_stamp:
            text = time.strftime("%Y/%m/%d %H:%M:%S: ", time.localtime()) + text
        f.writelines(text)

def atomic_print(content: Any, end: str='\n') -> None:
    global __atomic_print_and_log_mutex
    __atomic_print_and_log_mutex.acquire()
    print(content, end=end)
    __atomic_print_and_log_mutex.release()

def atomic_log(content: Any, end: str='\n', time_stamp: bool=False, file: str=global_config.log_file) -> None:
    global __atomic_print_and_log_mutex
    __atomic_print_and_log_mutex.acquire()
    text = str(content) + end
    __log(text, time_stamp, file)
    __atomic_print_and_log_mutex.release()

def atomic_print_and_log(content: Any, end: str='\n', time_stamp: bool=False, file: str=global_config.log_file):
    global __atomic_print_and_log_mutex
    __atomic_print_and_log_mutex.acquire()
    text = str(content) + end
    __log(text, time_stamp, file)
    print(text, end='')
    __atomic_print_and_log_mutex.release()

# others
def check_01() -> None:
    import _01_data2npz
    input_path = _01_data2npz.output_path
    if not os.path.exists(input_path) or len(os.listdir(input_path)) == 0:
        _01_data2npz.main()

def check_04() -> None:
    import _04_combine
    input_file_path = _04_combine.output_file_path
    if not os.path.exists(input_file_path):
        _04_combine.main()

def check_05() -> None:
    import _05_feature_extraction
    input_file_path = _05_feature_extraction.output_file_path
    if not os.path.exists(input_file_path):
        _05_feature_extraction.main()

def check_07() -> None:
    import _07_add_labels
    input_file_path = _07_add_labels.output_file_path
    if not os.path.exists(input_file_path):
        _07_add_labels.main()

def confusion_matrix(predict, expect, positive_label) -> Tuple:
    if predict.shape != expect.shape:
        raise ValueError()
    if len(predict.shape) != 1:
        raise ValueError()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(predict)):
        if predict[i] == expect[i]:
            if predict[i] == positive_label:
                tp += 1
            else:
                tn += 1
        else:
            if predict[i] == positive_label:
                fp += 1
            else:
                fn += 1

    return tp, fn, fp, tn

def analyse_confusion_matrix(tp_fn_fp_tn: Tuple):
    tp, fn, fp, tn = tp_fn_fp_tn

    print('tp=%d, fn=%d, fp=%d, tn=%d, accuracy=%.1f%%, precision=%.1f%%, recall=%.1f%%' % (
        tp, fn, fp, tn,
        (tp + tn) / (tp + fn + fp + tn) * 100,
        tp / (tp + fp) * 100,
        tp / (tp + fn) * 100))