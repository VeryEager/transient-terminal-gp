"""
Contains functions shared by both standard & modified MOGP.

Written by Asher Stout, 300432820
"""
from sklearn.metrics import mean_squared_error
import numpy as np


def protected_division(x, y):
    """
    Performs a protected division on the two operators

    :param x: numerator
    :param y: denominator
    :return: x/y when y=/=0 and x=/=0, otherwise 0 or 1
    """
    if x == 0:
        return 0
    elif y == 0:
        return 1
    else:
        return x/y


def eval_solution(function, tb, data, actual):
    """
    Evaluates the MSE and complexity of a candidate solution.

    :param function: the candidate solution to evaluate
    :param pset: the primitive set
    :param tb: toolbox reference
    :param data: points to evaluate on
    :param actual: the correct predictions for the data
    :return: a tuple of fitnesses
    """
    func = tb.compile(expr=function)
    results = [func(*res) for res in data]
    acc = mean_squared_error(actual, results)
    return acc, function.height,
