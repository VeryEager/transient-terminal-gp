"""
Loads evolutionary data & draws corresponding figures. SHould be performed only AFTER all data has been collected.

Written by Asher Stout.
"""

import os
import numpy as np
import test_shared as ts
import matplotlib.pyplot as plot
from pathlib import Path

log_colors = ['#FCD12A', '#990F02', '#151FB0', '#028A0F', '#A45EE5']
_measures = ['complexity', 'accuracy']


def draw_comparison(name, feature, *logs):

    fig, ax1 = plot.subplots()
    fig.suptitle("Best solution of " + name + " data: 50 seed avg")
    fig.tight_layout()
    ax1.set_xlabel('Generation')
    genlog_ref = np.load(Path.cwd() / '..' / 'docs' / 'Data' / str(logs[0][0]), allow_pickle=True)
    ax2 = ax1.twinx()
    xax = list(log['gen'] for log in genlog_ref)

    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='y')
    ax2.set_ylabel('complexity (tree size)', color="#191970")
    ax2.tick_params(axis='y')

    # Add each dataset evolution as a new line, and print the name of the dataset next to it
    for filename, color in zip(logs[0], log_colors):
        file = np.load(Path.cwd() / '..' / 'docs' / 'Data' / str(filename), allow_pickle=True)
        ax1.plot(xax, list(log[feature][0] for log in file), color=color, alpha=0.8, label=filename.split('-')[2])
        ax2.plot(xax, list(log[feature][1] for log in file), color=color, alpha=0.4)
        fig.tight_layout()
    ax1.legend(loc='center left', bbox_to_anchor=(0.0, 0.8), shadow=False, ncol=1)


    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Figures' / str(name + '-' + feature + '-evo')
    plot.savefig(fname=path)
    plot.clf()


if __name__ == "__main__":
    path = Path.cwd() / '..' / 'docs' / 'Data'
    data = os.listdir(path)

    data = [a for a in data if not a.__contains__('eval')]
    for filetype in ['best', 'besttrain']:
        redwine = [a for a in data if a.__contains__(str(filetype+'-')) and a.__contains__('red')]
        whitewine = [a for a in data if a.__contains__(str(filetype+'-')) and a.__contains__('white')]
        concrete = [a for a in data if a.__contains__(str(filetype+'-')) and a.__contains__('concrete')]
        house = [a for a in data if a.__contains__(str(filetype+'-')) and a.__contains__('house')]
        draw_comparison('Red Wine Quality (50 generations)', filetype, [r for r in redwine if r.__contains__('g50')])
        draw_comparison('Red Wine Quality (250 generations)', filetype, [r for r in redwine if r.__contains__('g250')])
        draw_comparison('White Wine Quality (50 generations)', filetype, [r for r in whitewine if r.__contains__('g50')])
        draw_comparison('White Wine Quality (250 generations)', filetype, [r for r in whitewine if r.__contains__('g250')])
        draw_comparison('Concrete Strength (50 generations)', filetype, [r for r in concrete if r.__contains__('g50')])
        draw_comparison('Concrete Strength (250 generations)', filetype, [r for r in concrete if r.__contains__('g250')])
        draw_comparison('Boston House Price (50 generations)', filetype, [r for r in house if r.__contains__('g50')])
        draw_comparison('Boston House Price (250 generations)', filetype, [r for r in house if r.__contains__('g250')])

    # This prints out all data, including parameter experiment data
    # for file in data:
    #     _type = None
    #     if file.__contains__('best-'):
    #         _type = 'best'
    #     elif file.__contains__('train-'):
    #         _type = 'besttrain'
    #     elif file.__contains__('balance-'):
    #         _type = 'balanced'
    #     elif file.__contains__('time'):
    #         _type = 'time'
    #     else:
    #         continue
    #     # filename = filename[0] + '-' + filename[1]
    #     # ts.draw_solutions_from_data(filename, _type, _measures[0], files[0], files[1])
    #     # ts.draw_solutions_from_data(filename, _type, _measures[1], files[0], files[1])
    #     if file.__contains__('thresh') or file.__contains__('tmutpb'):
    #         logs = np.load(Path.cwd() / '..' / 'docs' / 'Data' / file, allow_pickle=True)
    #         print(file)
    #         if _type != 'time':
    #             for log in logs:
    #                 print(log[49][_type])
    #         else:
    #             print(np.mean(logs[0]))
    #     else:
    #         ts.print_solutions_from_data(_type, file)
