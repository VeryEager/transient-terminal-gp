"""
Computes statistical significance tests from raw data obtained during evolution

Written by Asher Stout.
"""

import os
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path


def retrieve_final_gen(data):
    """
    Retrieves the testing fitness of the final generation of a method's results

    :param data: the data to retrieve the final generation info from
    :return: an array of the final generation info
    """
    finals = []
    for seed in data:
        finals.append(seed[len(seed)-1]['best'])
    return finals


if __name__ == "__main__":
    # Load correct files,
    path = Path.cwd() / '..' / '..' / 'docs' / 'Data'
    data = os.listdir(path)
    data = [d for d in data if d.__contains__('raw')]

    # Sort results into which dataset they use
    redwine = [a for a in data if a.__contains__('red')]
    whitewine = [a for a in data if a.__contains__('white')]
    concrete = [a for a in data if a.__contains__('concrete')]
    house = [a for a in data if a.__contains__('house')]

    for dataset in [redwine, whitewine, concrete, house]:
        # Preprocess dataset & MOGP
        mogp = dataset[1]
        print('\n', mogp)
        del dataset[1]
        mogp = np.load(Path.cwd() / '..' / '..' / 'docs' / 'Data' / mogp, allow_pickle=True)
        mogp = retrieve_final_gen(mogp)
        mogp_rmse = [a[0] for a in mogp]
        mogp_size = [a[1] for a in mogp]

        # Iterate through each other method to compare results to MOGP
        for data in dataset:
            print(data)
            method = np.load(Path.cwd() / '..' / '..' / 'docs' / 'Data' / data, allow_pickle=True)
            method = retrieve_final_gen(method)
            method_rmse = [a[0] for a in method]
            method_size = [a[1] for a in method]
            _stat, pval = mannwhitneyu(mogp_rmse, method_rmse, alternative='greater')
            print(pval)
            _stat, pval = mannwhitneyu(mogp_size, method_size, alternative='greater')
            print(pval)
