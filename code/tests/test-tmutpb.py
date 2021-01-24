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
import test_shared as ts
from pathlib import Path    # supports inter-OS relative path

colors = ['#1a2a6c', '#272966', '#33285f', '#402759', '#4d2652', '#662545', '#73243f', '#7f2339', '#8c2232', '#99212c',
          '#a52025', '#b21f1f', '#b82c20', '#be3921', '#cb5324', '#d16025', '#d76d26', '#de7a27']
_range = np.arange(0.0, 0.95, 0.05)


def draw_mutation_descents(logs, measure, method, metric, show=False, fname='descent'):
    """
    Plots the change in accuracy/complexity in solutions over multiple transient mutation probabilities

    :param logs: a list of each run's average
    :param measure: the measure to plot from the logbook
    :param method: method used, in string format
    :param metric: the metric to be graphed, ONE OF: (complexity, accuracy)
    :param show: whether to display the figure after saving
    :param fname: file name of the figure
    :return:
    """
    fit = 0
    if metric == "complexity":
        fit = 0
    else:
        fit = 1

    # Create plot, add titles & initialize the axes
    fig, ax1 = plot.subplots(figsize=(10, 6))
    fig.suptitle(metric + " of best TTSGP solutions (50 runs): tmutpb="+_range)
    fig.tight_layout()
    ax1.set_xlabel('generation')
    ax1.set_ylabel(metric)
    ax1.tick_params(axis='y')

    # Draw all y axis COMPLEXITY
    for mut, color, prob in zip(logs, colors, _range):
        xax = list(log['gen'] for log in mut)
        ax1.plot(xax, list(log[measure][fit] for log in mut), color=color, alpha=0.6, label=str("prob = "+str(round(prob, 2))))
        plot.text(49.5, mut[49][measure][fit], str(prob), horizontalalignment='left', size='small', color=color)
    fig.tight_layout()

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Tests' / str(fname + '-' + metric + '-' + method)
    plot.savefig(fname=path)
    if show:
        plot.show()
    plot.clf()


def draw_time_ascent(logs, probabilities):
    """
    Draws the time of calculation for/e probability run of TTSGP

    :param logs: the time logs, a collection of floats
    :param probabilities: associated probabilities for each log
    :return:
    """
    fig, ax1 = plot.subplots()
    fig.suptitle("Execution time for TTSGP (50 runs): tmutpb="+_range)
    fig.tight_layout()
    ax1.set_xlabel('transient mutation probability')
    ax1.set_ylabel('execution time (seconds)')
    ax1.tick_params(axis='y')
    ax1.plot(probabilities, logs, color="#B90E0A", alpha=0.6)
    fig.tight_layout()

    # Save the figure
    path = Path.cwd() / '..' / 'docs' / 'Tests' / 'mutationdescent-times'
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
    for prob in _range:
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
                                      cxpb=(0.8-(prob-0.1)), tmutpb=prob)
            tts_log.append(_log)
            tts_best.append(_best)
            print("FINISHED EVOLUTION OF POPULATION: ", i)
        time_logs.append(time.time()-start_time)
        # Average the results & report descent & best individual.
        averaged = ts.average_results(tts_log, 'best')
        print("FINISHED EVALUATION OF tmutpb: ", prob)
        prob_logs.append(averaged)
    draw_mutation_descents(prob_logs, measure='best', method='TTSGP', metric='complexity', fname='mutationdescent')
    draw_mutation_descents(prob_logs, measure='best', method='TTSGP', metric='accuracy', fname='mutationdescent')
    draw_time_ascent(time_logs, _range)
