import itertools
from copy import deepcopy
from glob import glob

import numpy as np
import torch
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver


def merge_dict(left, right):
    if not isinstance(right, dict):
        return right

    merged = dict(left)
    for key, value in right.items():
        if isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged

def train_test_split_using_knapsack(items, key, ratio=0.8):
    items_by_key = {
        key: list(items) 
        for key, items in itertools.groupby(sorted(items, key=key), key=key)
    }
    
    keys = list(items_by_key.keys())
    number_of_items_by_key = [len(items_by_key[key]) for key in keys]
    capacity = int(sum(number_of_items_by_key) * ratio)
    
    solver = KnapsackSolver(
        KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 
        'TrainTestSplitKnapsack'
    )
    solver.Init(profits=number_of_items_by_key, weights=[number_of_items_by_key], capacities=[capacity])
    solver.Solve()
    
    train_keys = set([key for index, key in enumerate(keys) if solver.BestSolutionContains(index)])
    test_keys = items_by_key.keys() - train_keys
    
    return (
        list(itertools.chain.from_iterable(items_by_key[key] for key in train_keys)),
        list(itertools.chain.from_iterable(items_by_key[key] for key in test_keys)),
    )

def to_step_interval(lr_scheduler):
    def inner(optimizer):
        return {
            'scheduler': lr_scheduler(optimizer),
            'interval': 'step',
        }
    return inner
     
def balanced_sampler(labels, number_of_samples_per_label_function, replacement):
    class_counts = np.bincount(labels)
    class_weights = [
        (np.float64(1.0) / class_counts[label]) * np.sum(class_counts) / 2
        for label in np.unique(labels)
    ]

    number_of_samples_per_label = number_of_samples_per_label_function(class_counts)
    number_of_samples = int(number_of_samples_per_label * len(class_counts))

    sample_weights = [class_weights[label] for label in labels]
    return torch.utils.data.WeightedRandomSampler(sample_weights, number_of_samples, replacement)

def oversampler(labels):
    return balanced_sampler(labels, number_of_samples_per_label_function=np.max, replacement=True)

def undersampler(labels, replacement=False):
    return balanced_sampler(labels, number_of_samples_per_label_function=np.min, replacement=replacement)

def list_files(*args, recursive=True):
    return list(itertools.chain.from_iterable(glob(pattern, recursive=recursive) for pattern in args))

def enter(message, options):
    assert(sum(1 if option.isupper() else 0 for option in options) <= 1)
    
    default = None
    for option in options:
        if option.isupper():
            default = option.lower()
            break
    
    prompt = f'{message} ({"/".join(options)}) '
    options = tuple(option.lower() for option in options)

    while True:
        choice = input(prompt).strip().lower()

        if choice in options:
            return choice

        if (not choice) and (default is not None):
            return default