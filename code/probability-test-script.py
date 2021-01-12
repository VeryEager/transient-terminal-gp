"""
Test script for evaluating the relative change in results on a dataset when using
different transient mutation probabilities

Written by Asher Stout, 300432820
"""

import sys
import time
import ttgp
import shared
import pandas as pd
import random as rand
import numpy as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plot
from pathlib import Path    # supports inter-OS relative path

colors = ['#1a2a6c', '#4d2652', '#73243f', '#a52025', '#c54622', '#de7a27']


def draw_mutation_descents(logs, measure, method, show=False, fname='descent'):
    """
    Plots the change in accuracy/complexity in solutions over multiple transient mutation probabilities

    :param logs: a list of each run's average
    :param measure: the measure to plot from the logbook
    :param method: method used, in string format
    :param show: whether to display the figure after saving
    :param fname: file name of the figure
    :return:
    """
    # Create plot, add titles & initialize the axes
    fig, ax1 = plot.subplots(figsize=(10, 6))
    fig.suptitle("Complexity of best TTSGP solutions (50 runs): tmutpb=[0.00, 0.25]")
    fig.tight_layout()
    ax1.set_xlabel('generation')
    ax1.set_ylabel('complexity (tree size)')
    ax1.tick_params(axis='y')

    # Draw all y axis COMPLEXITY
    for mut, color, prob in zip(logs, colors, np.arange(0.0, 0.3, 0.05)):
        xax = list(log['gen'] for log in mut)
        ax1.plot(xax, list(log[measure][1] for log in mut), color=color, alpha=0.6, label=str("prob = " +
                                                                                              str(round(prob, 2))))

    ax1.legend(loc='center left', bbox_to_anchor=(0.0, 0.85), shadow=False, ncol=1)
    fig.tight_layout()

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Figures' / str(fname + '-' + method)
    plot.savefig(fname=path)
    if show:
        plot.show()
    plot.clf()


def draw_time_ascent(logs, probabilities):
    """
    Draws the time of calculation for/e probability run of TTSGP

    :param logs: the time logs, a collection of floats
    :param probabilities:
    :return:
    """
    fig, ax1 = plot.subplots()
    fig.suptitle("Execution time for TTSGP (50 runs): tmutpb=[0.00, 0.25]")
    fig.tight_layout()
    ax1.set_xlabel('transient mutation probability')
    ax1.set_ylabel('execution time (seconds)')
    ax1.tick_params(axis='y')
    ax1.plot(probabilities, logs, color="#B90E0A", alpha=0.6)

    fig.tight_layout()
    # Save the figure
    path = Path.cwd() / '..' / 'docs' / 'Figures' / 'mutationdescent-times'
    plot.savefig(fname=path)
    plot.clf()


if __name__ == "__main__":
    """
    README: accepts three command line arguments
    arg1: name of dataset (in .csv format) to evaluate
    arg2: name of the dataset's target variable 
    arg3: separator character (usually ';' or ',' - CHECK DATASET PRIOR
    
    """
    # Load red wine data
    path = Path.cwd() / '..' / 'data' / str(sys.argv[1] + '.csv')
    dataset = pd.read_csv(path.resolve(), sep=sys.argv[3])
    target = sys.argv[2]

    prob_logs = []
    time_logs = []
    for prob in np.arange(0.00, 0.3, 0.05):
        tts_log = []
        tts_best = []
        start_time = time.time()
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
            _best, _log = ttgp.evolve(train_data, train_target, dataset.columns.drop([target]), test_data, test_target,
                                      tmutpb=prob)
            tts_log.append(_log)
            tts_best.append(_best)
            print("FINISHED EVOLUTION OF POPULATION: ", i)
            if i == 9:
                break
        time_logs.append(time.time()-start_time)

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
        print("FINISHED EVALUATION OF tmutpb: ", prob)
        prob_logs.append(averaged)
    draw_mutation_descents(prob_logs, measure='best', method='TTSGP', fname='mutationdescent')
    draw_time_ascent(time_logs, np.arange(0.0, 0.3, 0.05))
