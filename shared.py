"""
Contains functions shared by both standard & modified MOGP.

Written by Asher Stout, 300432820
"""
from sklearn.metrics import mean_squared_error
from deap import gp
import matplotlib.pyplot as plot
import networkx # used for plotting trees


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

    accuracy = mean_squared_error(actual, results)
    complexity = function.height
    return accuracy, complexity,


def draw_solution(individual):
    """
    Displays a connected tree representing an individual. Presumably this individual scores highly in the population
    using the method eval_solution.

    :param individual: the individual (represented as a tree) to draw
    :return:
    """
    graph = networkx.Graph()
    node, edge, label = gp.graph(individual)
    graph.add_edges_from(edge)
    graph.add_nodes_from(node)

    pos = networkx.graphviz_layout(graph, prog="dot")
    networkx.draw_networkx_nodes(graph, pos)
    networkx.draw_networkx_edges(graph, pos)
    networkx.draw_networkx_labels(graph, pos, label)
    plot.show()
    return
