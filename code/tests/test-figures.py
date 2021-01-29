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

    for files in zip(data[::1], data[5::1]):
        if files[0][0:14] != files[1][0:14]:    # Ensures only equivalent datasets are checked
            continue
        _type = None
        if files[0].__contains__('best'):
            _type = 'best'
            if files[0].__contains__('train'):
                _type = 'besttrain'
        else:
            _type = 'balanced'

        print(files)
        break
        filename = files[0].split('-')
        filename = filename[0] + '-' + filename[1]
        ts.draw_solutions_from_data(filename, _type, _measures[0], files[0], files[1])
        ts.draw_solutions_from_data(filename, _type, _measures[1], files[0], files[1])
        ts.print_solutions_from_data(_type, files[0], files[1])
