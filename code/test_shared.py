"""
Contains shared functions for test scripts

Written by Asher Stout, 300432820
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from pathlib import Path

log_colors = ['#DC143C', '#1471DC', '#7FDC14']


def average_results(logs, method):
    """
    Averages the results for every generation/seed in log

    :param logs: self-explanatory
    :param method: The metric to average, should be obtainable in the log
    :return:
    """
    averaged = []
    for log in logs:
        averaged.append([ind[method] for ind in log])
    averaged = pd.DataFrame(averaged)
    acc = [np.mean([entry[0] for entry in averaged[col]]) for col in averaged]
    com = [np.mean([entry[1] for entry in averaged[col]]) for col in averaged]
    d = {'gen': [entry['gen'] for entry in logs[0]], method: zip(acc, com)}
    averaged = pd.DataFrame(d)  # This collects the average over the runs at each generation
    averaged = [{'gen': entry[0], method: entry[1][1]} for entry in averaged.iterrows()]
    return averaged


def average_singular_results(logs, method):
    """
    Averages the results of a SINGLE variable (ie, not contained in a tuple)

    :param logs: self-explanatory
    :param method: The metric to average, should be obtainable in the log
    :return:
    """
    averaged = []
    for log in logs:
        averaged.append([ind[method] for ind in log])
    averaged = pd.DataFrame(averaged)
    acc = [np.mean([entry for entry in averaged[col]]) for col in averaged]
    d = {'gen': [entry['gen'] for entry in logs[0]], method: acc}
    averaged = pd.DataFrame(d)  # This collects the average over the runs at each generation
    averaged = [{'gen': entry[0], method: entry[1][1]} for entry in averaged.iterrows()]
    return averaged


def draw_solutions_from_data(fname, measure, fitness, *args):
    """
    Loads and graphs multiple files containing evolutionary data

    :param fname: name of the figure once saved
    :param measure: the metric from the files to visualize. Normally ONE OF (best, balanced)
    :param fitness: which fitness are we plotting? ONE OF (complexity, accuracy)
    :param args: pointers to files containing logs able to be loaded & graphed
    :return:
    """
    logs = [np.load(Path.cwd() / '..' / 'docs' / 'Data' / arg, allow_pickle=True) for arg in args]
    c = "-data.npy"
    i = 0
    s = "(RMSE)"
    if fitness == "complexity":
        i = 1
        s = "(tree size)"

    # Create plot, add titles & initialize the axes axis
    fig, ax1 = plot.subplots()
    fig.suptitle(fitness + " of " + measure + " solution during evolution, 50 runs")
    fig.tight_layout()
    ax1.set_xlabel('generation')
    xax = list(log['gen'] for log in logs[0])
    for file, color, arg in zip(logs, log_colors, args):
        # Draw first y axis ACCURACY
        ax1.set_ylabel(fitness+' '+s)
        ax1.plot(xax, list(log[measure][i] for log in file), color=color, alpha=0.6, label=arg.replace(c, ''))
        ax1.tick_params(axis='y')
        fig.tight_layout()
    ax1.legend(loc='center left', bbox_to_anchor=(0.0, 0.9), shadow=False, ncol=1)

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Figures' / str(fname+'-'+fitness+'-'+measure)
    plot.savefig(fname=path)
    plot.clf()
