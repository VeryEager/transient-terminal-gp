"""
Test script for gauging the performance of TTSGP versus standard GP

Written by Asher Stout, 300432820
"""

import sys
import ttgp
import shared
import numpy as np
import pandas as pd
import random as rand
import mogp as mogp
import sgp
import sklearn.model_selection as skms
import test_shared as ts
from pathlib import Path    # supports inter-OS relative path

if __name__ == "__main__":
    """
    README: accepts three command line arguments
    arg1: name of dataset (in .csv format) to evaluate
    arg2: name of the dataset's target variable 
    arg3: separator character (usually ';' or ',' - CHECK DATASET PRIOR
    arg4: string representing the method used. ONE OF: (sgp, mogp, ttsgp)
    arg5: int GENERATIONS
    arg6: int POPSIZE
    
    """
    # Configure evolutionary method from command-line argument
    method = sys.argv[4]
    if method == "sgp":
        method = sgp
    elif method == "mogp":
        method = mogp
    elif method == "ttgp":
        method = ttgp
    else:
        print("UNKNOWN METHOD: MUST BE ONE OF (sgp, mogp, ttgp)")

    # Load red wine data
    path = Path.cwd() / '..' / 'data' / str(sys.argv[1] + '.csv')
    dataset = pd.read_csv(path.resolve(), sep=sys.argv[3])
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
        _best, _log = method.evolve(train_data, train_target, dataset.columns.drop([target]), test_data, test_target,
                                    generations=sys.argv[5], pop_size=sys.argv[6])
        tts_log.append(_log)
        tts_best.append(_best)
        print("FINISHED EVOLUTION OF GENERATION: ", i)

    # Average and save results in data file for best, besttrain, and balanced (on condition)
    best = ts.average_results(tts_log, 'best')
    path = Path.cwd() / '..' / 'docs' / 'Data' / str(sys.argv[1]+"-"+sys.argv[4]+"-best-"+sys.argv[5]+'-'+sys.argv[6])
    np.save(path, best)  # Save the results for later visualization

    besttrain = ts.average_results(tts_log, 'besttrain')
    path = Path.cwd() / '..' / 'docs' / 'Data' / str(sys.argv[1]+"-"+sys.argv[4]+"-besttrain"+sys.argv[5]+'-'+sys.argv[6])
    np.save(path, besttrain)  # Save the results for later visualization

    if method != 'sgp':  # Single-objective GP does not record balanced individual.
        balance = ts.average_results(tts_log, 'balanced')
        path = Path.cwd() / '..' / 'docs' / 'Data' / str(sys.argv[1]+"-"+sys.argv[4]+"-balance"+sys.argv[5]+'-'+sys.argv[6])
        np.save(path, balance)  # Save the results for later visualization
