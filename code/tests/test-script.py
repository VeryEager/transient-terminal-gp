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
    arg5: string which individual to plot. ONE OF: (best, balanced)
    
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
        _best, _log = method.evolve(train_data, train_target, dataset.columns.drop([target]), test_data, test_target)
        tts_log.append(_log)
        tts_best.append(_best)
        print("FINISHED EVOLUTION OF GENERATION: ", i)

    # Average the results and report descent & best individual.
    _type = sys.argv[5]
    averaged = ts.average_results(tts_log, _type)
    path = Path.cwd() / '..' / 'docs' / 'Data' / str(sys.argv[4]+"-data")
    np.save(path, averaged)  # Save the results for later visualization
    ts.draw_solutions_from_data(sys.argv[1], _type, 'complexity', 'mogp-data.npy', 'ttgp-data.npy')
    ts.draw_solutions_from_data(sys.argv[1], _type, 'accuracy', 'mogp-data.npy', 'ttgp-data.npy')
    # shared.draw_descent(averaged, measure=_type, method=sys.argv[4], fname=sys.argv[1]+'-evo-'+_type)
    # shared.draw_solution(tts_best[0], fname=sys.argv[1]+'-ex-'+sys.argv[4])

