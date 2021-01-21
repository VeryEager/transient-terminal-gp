"""
Contains shared functions for test scripts

Written by Asher Stout, 300432820
"""
import numpy as np
import pandas as pd


def average_results(logs, method):
    """
    Averages the results for every generation/seed in log

    :param log: self-explanatory
    :param method: The metric to average, should be obtainable in the log
    :return:
    """
    averaged = []
    for log in logs:
        averaged.append([ind[method] for ind in log])
    averaged = pd.DataFrame(averaged)
    acc = [np.mean([entry[0] for entry in averaged[col]]) for col in averaged]
    com = [np.mean([entry[1] for entry in averaged[col]]) for col in averaged]
    d = {'gen': [entry['gen'] for entry in logs[0]], method: zip(acc, com)}
    averaged = pd.DataFrame(d)  # This collects the average over the runs at each generation
    averaged = [{'gen': entry[0], method: entry[1][1]} for entry in averaged.iterrows()]
    return averaged
