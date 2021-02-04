"""
Test script for evaluating the relative change in results on a dataset when using
different accuracy change thresholds for adding subtrees to the TTS

Written by Asher Stout, 300432820
"""

import sys
import ttgp
import shared
import pandas as pd
import random as rand
import numpy as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plot
import test_shared as ts
from pathlib import Path    # supports inter-OS relative path

colors = ['#1a2a6c', '#272966', '#33285f', '#402759', '#4d2652', '#59254c', '#662545', '#73243f', '#7f2339', '#8c2232',
          '#99212c', '#a52025', '#b21f1f', '#b82c20', '#c54622', '#cb5324', '#d16025', '#d76d26', '#de7a27']
_range = np.arange(5, 105, 5.0)


def draw_threshold_descents(logs, measure, method, metric, show=False, fname='descent'):
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
    fig.suptitle(metric+" of "+measure+" TTSGP solutions (50 runs): percentile=["+str(np.min(_range))+', '+str(np.max(_range))+']')
    fig.tight_layout()
    ax1.set_xlabel('generation')
    ax1.set_ylabel(metric)
    ax1.tick_params(axis='y')

    # Draw all y axis COMPLEXITY
    for mut, color, thresh in zip(logs, colors, _range):
        xax = list(log['gen'] for log in mut)
        ax1.plot(xax, list(log[measure][fit] for log in mut), color=color, alpha=0.6, label=str("thresh = " + str(round(thresh, 2))))
        plot.text(49.5, mut[49][measure][fit], str(thresh), horizontalalignment='left', size='small', color=color)

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Tests' / str(fname+'-'+method+"-"+metric+'-'+measure)
    plot.savefig(fname=path)
    if show:
        plot.show()
    plot.clf()


def draw_threshold_tts_effect(logs, measure):
    """
    Draws the threshold effect on a value from the TTS.

    :param logs: The logs to draw from
    :param measure: The measure to draw. ONE OF: (tsAvg, tsMed, tsMax, tsLen)
    :return:
    """
    # Create plot, add titles & initialize the axes
    fig, ax1 = plot.subplots(figsize=(10, 6))
    fig.suptitle("mean " + measure+" for each generation, 50 runs")
    fig.tight_layout()
    ax1.set_xlabel('generation')
    ax1.set_ylabel(measure)
    ax1.tick_params(axis='y')

    # Draw all y axis COMPLEXITY
    for mut, color, thresh in zip(logs, colors, _range):
        xax = list(log['gen'] for log in mut)
        ax1.plot(xax, list(log[measure] for log in mut), color=color, alpha=1.0, label=str("thresh = " + str(round(thresh, 2))), linewidth=1.2)
        plot.text(49.5, mut[49][measure], str(thresh), horizontalalignment='left', size='small', color=color)

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Tests' / str("thresh-"+measure+"-evo")
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

    thresh_logs = []
    bal_logs = []
    avg_logs = []
    med_logs = []
    max_logs = []
    len_logs = []   # Logs for each parameter we want to observe
    for thresh in _range:
        tts_log = []
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
            ttgp.__set_transient_threshold(thresh)
            _log, _best, _time = ttgp.evolve(train_data, train_target, dataset.columns.drop([target]), test_data, test_target)
            tts_log.append(_log)
            print("FINISHED EVOLUTION OF POPULATION: ", i)
        # Average the results & report descent & best individual.
        best = ts.average_results(tts_log, 'best')
        thresh_logs.append(best)
        bal = ts.average_results(tts_log, 'balanced')
        bal_logs.append(bal)
        avg = ts.average_singular_results(tts_log, 'tsAvg')
        avg_logs.append(avg)
        med = ts.average_singular_results(tts_log, 'tsMed')
        med_logs.append(med)
        max = ts.average_singular_results(tts_log, 'tsMax')
        max_logs.append(max)
        len = ts.average_singular_results(tts_log, 'tsLen')
        len_logs.append(len)
        print("FINISHED EVOLUTION OF THRESHOLD: ", thresh)
    # Save related results for later evaluation
    path = Path.cwd() / '..' / 'docs' / 'Data' / 'thresh-eval-best'
    np.save(path, thresh_logs)
    path = Path.cwd() / '..' / 'docs' / 'Data' / 'thresh-eval-balance'
    np.save(path, bal_logs)
    path = Path.cwd() / '..' / 'docs' / 'Data' / 'thresh-eval-tsAvg'
    np.save(path, avg_logs)
    path = Path.cwd() / '..' / 'docs' / 'Data' / 'thresh-eval-tsMed'
    np.save(path, med_logs)
    path = Path.cwd() / '..' / 'docs' / 'Data' / 'thresh-eval-tsMax'
    np.save(path, max_logs)
    path = Path.cwd() / '..' / 'docs' / 'Data' / 'thresh-eval-tsLen'
    np.save(path, len_logs)

    # draw_threshold_descents(thresh_logs, measure='best', method='TTSGP', metric='complexity', fname='thresholddescent')
    # draw_threshold_descents(thresh_logs, measure='best', method='TTSGP', metric='accuracy', fname='thresholddescent')
    # draw_threshold_descents(bal_logs, measure='balanced', method='TTSGP', metric='complexity', fname='thresholddescent')
    # draw_threshold_descents(bal_logs, measure='balanced', method='TTSGP', metric='accuracy', fname='thresholddescent')
    # draw_threshold_tts_effect(avg_logs, "tsAvg"), draw_threshold_tts_effect(med_logs, "tsMed")
    # draw_threshold_tts_effect(max_logs, "tsMax"), draw_threshold_tts_effect(len_logs, "tsLen")
