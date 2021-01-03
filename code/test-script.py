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

if __name__ == "__main__":
    # Load wine data
    path = os.path.relpath('..\\data\\winequality-red.csv', os.path.dirname(__file__))
    winered = pd.read_csv(path, sep=";")
    winered_data = winered.drop(['quality'], axis=1).values
    winered_target = winered['quality'].values

    tts_log = []
    tts_best = []
    # Perform experiments over seeds
    for seed in shared.seeds:
        rand.seed(seed)
        _best, logs = sgp.main(winered_data, winered_target, winered_data.shape[1], winered.columns.drop(['quality']))
        tts_log.append(logs)
        tts_best.append(_best)

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

    shared.draw_descent(averaged, measure='best', method="TTS GP, 30 runs")
    shared.draw_solution(tts_best[10])  # for seed 718695369195
