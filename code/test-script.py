"""
Test script for gauging the performance of TTSGP versus standard GP

Written by Asher Stout, 300432820
"""
import random as rand
import os.path
import shared
import ttgp
import pandas as pd
import standardgp as sgp
import numpy as np
import sklearn.model_selection as skms

if __name__ == "__main__":
    # Load red wine data
    path = os.path.relpath('..\\data\\winequality-red.csv', os.path.dirname(__file__))
    dataset = pd.read_csv(path, sep=";")

    tts_log = []
    tts_best = []
    # Perform experiments over seeds
    for i, seed in enumerate(shared.seeds):
        rand.seed(seed)

        # Split into training & test sets
        train, test = skms.train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=seed)
        train_data = train.drop(['quality'], axis=1).values
        train_target = train['quality'].values
        test_data = test.drop(['quality'], axis=1).values
        test_target = test['quality'].values

        # Perform Evolution using Seed
        _best, _log = ttgp.evolve(train_data, train_target, dataset.columns.drop(['quality']), test_data, test_target)
        tts_log.append(_log)
        tts_best.append(_best)
        print("FINISHED EVOLUTION OF GENERATION: ", i)
        break

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

    shared.draw_descent(averaged, measure='best', method="Standard MOGP, 50 runs")
    shared.draw_solution(tts_best[0])
