"""
Contains functions shared by both standard & modified MOGP.

Written by Asher Stout, 300432820
"""
from sklearn.metrics import mean_squared_error
from networkx.drawing.nx_agraph import graphviz_layout
from deap import gp
import operator as op
import matplotlib.pyplot as plot
import random as rand
import networkx     # used for plotting trees

seed = 96127708431
rand.seed(seed)


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
    :param tb: toolbox reference
    :param data: points to evaluate on
    :param actual: the correct predictions for the data
    :return: a tuple of fitnesses
    """
    func = tb.compile(expr=function)
    results = [func(*res) for res in data]

    accuracy = mean_squared_error(actual, results, squared=False)
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

    pos = graphviz_layout(graph, prog="dot")
    networkx.draw_networkx_nodes(graph, pos)
    networkx.draw_networkx_edges(graph, pos)
    networkx.draw_networkx_labels(graph, pos, label)
    plot.title(individual.fitness.values)
    plot.show()


def draw_descent(logs, measure):
    """
    Plots the accuracy of a selected measure over generations, for both accuracy and complexity

    :param logs: the logbook from the main execution
    :param measure: the measure to plot from the logbook
    :return:
    """
    # Create plot, add titles & initialize the axes axis
    fig, ax1 = plot.subplots()
    fig.suptitle("Accuracy & Complexity of best solutions in generations over evolution: winequality-red")
    fig.tight_layout()
    ax1.set_xlabel('generation')
    ax2 = ax1.twinx()
    xax = list(log['gen'] for log in logs)

    # Draw first y axis ACCURACY
    ax1.set_ylabel('accuracy (rmse)', color="#800000")
    ax1.plot(xax, list(log[measure][0] for log in logs), color='#800000', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor="#800000")

    # Draw second y axis COMPLEXITY
    ax2.set_ylabel('complexity (tree depth)', color="#191970")
    ax2.plot(xax, list(log[measure][1] for log in logs), color='#191970', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor="#191970")

    fig.tight_layout()
    plot.show()


def create_primitives(names, attrs=1):
    """
    Creates the terminal/function sets for GP

    :param names: the names of features (as would appear in graph)
    :param attrs: number of features present in the data
    :return: terminal_function_set, the generated primitive set
    """
    terminal_function_set = gp.PrimitiveSet(name="PSET", arity=attrs)
    terminal_function_set.addEphemeralConstant(name="PSET", ephemeral=lambda: rand.uniform(-1.0, 1.0))
    terminal_function_set.addPrimitive(op.add, 2)
    terminal_function_set.addPrimitive(op.sub, 2)
    terminal_function_set.addPrimitive(op.mul, 2)
    terminal_function_set.addPrimitive(protected_division, 2)

    # Replace whitespaces with '_' in feature names
    n = []
    for name in names:
       n.append(name.replace(" ", "_"))
    # TODO this is an utterly horrible, hardcoded solution to setting multiple feature names.
    terminal_function_set.renameArguments(ARG0=n[0], ARG1=n[1], ARG2=n[2], ARG3=n[3], ARG4=n[4], ARG5=n[5], ARG6=n[6],
                                          ARG7=n[7], ARG8=n[8], ARG9=n[9], ARG10=n[10])
    return terminal_function_set
