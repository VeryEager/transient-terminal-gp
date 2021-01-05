"""
Test script for gauging the performance of TTSGP versus standard GP

Written by Asher Stout, 300432820
"""

import sys
import ttgp
import shared
import pandas as pd
import random as rand
import standardgp as sgp
import numpy as np
import sklearn.model_selection as skms
from pathlib import Path    # supports inter-OS relative path

if __name__ == "__main__":
    """
    README: accepts two command line arguments
    arg1: name of dataset (in .csv format) to evaluate
    arg2: name of the dataset's target variable 
    
    """
    # Load red wine data
    path = Path.cwd() / '..' / 'data' / str(sys.argv[1] + '.csv')
    dataset = pd.read_csv(path.resolve(), sep=";")
    target = sys.argv[2]

    tts_log = []
    tts_best = []
    # Perform experiments over seeds
    for i, seed in enumerate(shared.seeds):
        rand.seed(seed)

        # Split into training & test sets
        train, test = skms.train_test_split(dataset, test_size=0.3, train_size=0.7, random_state=seed)
        train_data = train.drop([target], axis=1).values
        train_target = train[target].values
        test_data = test.drop([target], axis=1).values
        test_target = test[target].values

        # Perform Evolution using Seed
        _best, _log = sgp.evolve(train_data, train_target, dataset.columns.drop(['quality']), test_data, test_target)
        tts_log.append(_log)
        tts_best.append(_best)
        print("FINISHED EVOLUTION OF GENERATION: ", i)

    # Average the results & report descent & best individual.
    # TODO: There should be a cleaner way to achieve this.
    averaged = []
    for log in tts_log:
        averaged.append([ind['best'] for ind in log])
    averaged = pd.DataFrame(averaged)
    acc = [np.mean([entry[0] for entry in averaged[col]]) for col in averaged]
    com = [np.mean([entry[1] for entry in averaged[col]]) for col in averaged]
    d = {'gen': [entry['gen'] for entry in tts_log[0]], 'best': zip(acc, com)}
    averaged = pd.DataFrame(d)  # This collects the average over the runs at each generation
    averaged = [{'gen': entry[0], 'best':entry[1][1]} for entry in averaged.iterrows()]

    shared.draw_descent(averaged, measure='best', method="MOGP", fname=sys.argv[1]+'-evo')
    shared.draw_solution(tts_best[0], fname=sys.argv[1]+'-MOGP-ex')  # TODO: Use the best overall solution, not a random

