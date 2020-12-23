"""
Test script for gauging the performance of TTSGP versus standard GP

Written by Asher Stout, 300432820
"""
import random as rand
import os.path
import pandas
import shared
import ttgp
import standardgp as sgp
import numpy as np

if __name__ == "__main__":
    # Load wine data
    path = os.path.relpath('..\\data\\winequality-red.csv', os.path.dirname(__file__))
    winered = pandas.read_csv(path, sep=";")
    winered_data = winered.drop(['quality'], axis=1).values
    winered_target = winered['quality'].values

    tts_log = []
    tts_best = []
    # Perform experiments over seeds
    for seed in shared.seeds:
        rand.seed(seed)
        _best, logs = ttgp.main(winered_data, winered_target, winered_data.shape[1], winered.columns.drop(['quality']))
        tts_log.append(logs)
        tts_best.append(_best)
        break

    # Average the results & report descent & best individual
    print([ind[i]['best'] for i,ind in enumerate(tts_log)])

    shared.draw_descent(tts_log, measure='min', method="TTS GP")
    shared.draw_solution(tts_best[10])
