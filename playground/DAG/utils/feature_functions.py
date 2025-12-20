import numpy as np
from sklearn import preprocessing as pp

# 机器
def features_extract_func(task):
    return [
            task.task_config.duration, 
            task.task_config.datasize,
            task.feature['first_layer_task'],
            task.feature['layers_task'], 
            task.feature['child_task_numbers']
            ]  


def features_extract_func_ac(task):
    return features_extract_func(task)
    #return features_extract_func(task) + [task.task_config.instances_number, len(task.running_task_instances),
    #                                      len(task.finished_task_instances)]

# 这些是什么的权重
def features_normalize_func(x):
    x = np.array(x)
    scaled = pp.scale(x)
    return scaled
    #y = (np.array(x) - np.array([0, 0, 0, 0, 0, 0, 1.167, 1.167, 1.5, 1.833, 1.833])) / np.array(
    #    [2, 1, 1, 1, 100, 1, 0.897, 0.897, 0.957, 1.572, 1.572])
    #return y


def features_normalize_func_ac(x):
    x = np.array(x)
    scaled = pp.scale(x)
    return scaled
    y = (np.array(x) - np.array([0, 0, 0, 0, 0, 0, 1.167, 1.167, 1.5, 1.833, 1.833, 0, 0, 0])) / np.array(
        [2, 1, 1, 1, 100, 1, 0.897, 0.897, 0.957, 1.572, 1.572, 1, 1, 1])
    return y
