"""
Loads evolutionary data & draws corresponding figures. SHould be performed only AFTER all data has been collected.

Written by Asher Stout.
"""

import os
import test_shared as ts
from pathlib import Path

_measures = ['complexity', 'accuracy']
if __name__ == "__main__":
    path = Path.cwd() / '..' / 'docs' / 'Data'
    data = os.listdir(path)

    for file in data:
        _type = None
        if file.__contains__('best-'):
            _type = 'best'
        elif file.__contains__('train-'):
            _type = 'besttrain'
        elif file.__contains__('balance-'):
            _type = 'balanced'
        else:
            _type = 'time'
        # filename = filename[0] + '-' + filename[1]
        # ts.draw_solutions_from_data(filename, _type, _measures[0], files[0], files[1])
        # ts.draw_solutions_from_data(filename, _type, _measures[1], files[0], files[1])
        ts.print_solutions_from_data(_type, file)
