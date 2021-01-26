"""
Contains functions shared by both standard & modified MOGP.

Written by Asher Stout, 300432820
"""
import networkx                # for plotting trees
import sklearn.preprocessing as pre
import operator as op
import numpy as np
import matplotlib.pyplot as plot
import random as rand
from deap import gp, tools
from pathlib import Path                 # for saving figures to a directory post-run
from datetime import datetime            # for naming identical ephemeral constants between runs
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance       # for identifying best balanced solution in a Pareto front


seeds = [39256911, 933855996, 967670959, 968137054, 590138938, 297331027, 755510051, 692539982, 955575529, 462966506,
         575520985, 614618594, 689935942, 638114944, 691154779, 224772871, 822094948, 811947924, 259107591, 784778275,
         87347336, 80635188, 661477758, 785773283, 950108759, 223073113, 309659173, 670766008, 370663937, 77914081,
         74406820, 94203230, 510105635, 717950752, 895387929, 865939420, 696230280, 695258916, 241343355, 720042387,
         232736156, 424335977, 353975857, 517983807, 674857291, 139546984, 446846029, 69735089, 876193725, 323506402]

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


def eval_solution(function, data, actual, _tb):
    """
    Evaluates the RMSE and complexity of a candidate solution.

    :param function: the candidate solution to evaluate
    :param data: points to evaluate on
    :param actual: the correct predictions for the data
    :param _tb:
    :return: a tuple of fitnesses
    """
    if function is None:    # In exceptional circumstances only
        return np.Infinity, np.Infinity,

    func = _tb.compile(expr=function)
    results = [func(*res) for res in data]

    accuracy = mean_squared_error(actual, results, squared=False)
    complexity = len(function)
    return accuracy, complexity,


def draw_solution(individual, show=False, fname='solution'):
    """
    Displays a connected tree representing an individual. Presumably this individual scores highly in the population
    using the method eval_solution.

    :param individual: the individual (represented as a tree) to draw
    :param show: whether to display the figure after saving
    :param fname: file name of the figure
    :return:
    """
    graph = networkx.Graph()
    node, edge, label = gp.graph(individual)
    graph.add_edges_from(edge)
    graph.add_nodes_from(node)

    pos = graphviz_layout(graph, prog="dot")
    networkx.draw_networkx_nodes(graph, pos, node_size=0)
    networkx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='#1338BE')
    networkx.draw_networkx_labels(graph, pos, label, font_size=9,font_family="Times New Roman",font_weight="bold")
    plot.title(label="Training Fitness: " + str(individual.fitness.values[0]) + " " + str(individual.fitness.values[1]))

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Figures' / fname
    plot.savefig(fname=path)
    if show:
        plot.show()
    plot.clf()


def draw_descent(logs, measure, method, show=False, fname='descent'):
    """
    Plots the accuracy of a selected measure over generations, for both accuracy and complexity

    :param logs: the logbook from the main execution
    :param measure: the measure to plot from the logbook
    :param method: method used, in string format
    :param show: whether to display the figure after saving
    :param fname: file name of the figure
    :return:
    """
    # Create plot, add titles & initialize the axes axis
    fig, ax1 = plot.subplots()
    fig.suptitle("Accuracy & Complexity of " + measure + " solution during evolution: " + method + ", 50 runs")
    fig.tight_layout()
    ax1.set_xlabel('generation')
    ax2 = ax1.twinx()
    xax = list(log['gen'] for log in logs)

    # Draw first y axis ACCURACY
    ax1.set_ylabel('accuracy (rmse)', color="#800000")
    ax1.plot(xax, list(log[measure][0] for log in logs), color='#800000', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor="#800000")

    # Draw second y axis COMPLEXITY
    ax2.set_ylabel('complexity (tree size)', color="#191970")
    ax2.plot(xax, list(log[measure][1] for log in logs), color='#191970', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor="#191970")
    fig.tight_layout()

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Figures' / str(fname + '-' + method)
    plot.savefig(fname=path)
    if show:
        plot.show()
    plot.clf()


def create_primitives(names, attrs=1):
    """
    Creates the terminal/function sets for GP crossover/mutation

    :param names: the names of features (as would appear in graph)
    :param attrs: number of features present in the data
    :return: terminal_function_set, the generated primitive set
    """
    terminal_function_set = gp.PrimitiveSet(name="PSET", arity=attrs)
    terminal_function_set.addEphemeralConstant(name="PSET"+str(datetime.now()), ephemeral=lambda: rand.uniform(-1.0, 1.0))
    terminal_function_set.addPrimitive(op.add, 2)
    terminal_function_set.addPrimitive(op.sub, 2)
    terminal_function_set.addPrimitive(op.mul, 2)
    terminal_function_set.addPrimitive(protected_division, 2)

    # Ensure feature names are legal & rename arguments to match them
    n = []
    [n.append(name.replace(" ", "_")) for name in names]
    terminal_function_set.renameArguments(**{'ARG' + str(i) : n[i] for i in range(0, len(names))})
    return terminal_function_set


def init_logger(*names):
    """
    Initializes the statistics & logbook for tracking across all implementations

    :param names: string names for other variables to intermittenly record
    :return: a tuple composed of stats and the logbook
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("25th percentile", np.percentile, q=25, axis=0)
    stats.register("median", np.median, axis=0)
    stats.register("75th percentile", np.percentile, q=75, axis=0)
    log = tools.Logbook()
    log.header = names + tuple(stats.fields)
    return stats, log


def getBalancedInd(pareto):
    """
    Retrieves the most balanced individual from a Pareto front.

    :param pareto: The Pareto front to pull the individual from
    :return: the individual
    """
    root = (0, 0)
    scale = protected_division([ind.fitness.values[1] for ind in pareto], max([ind.fitness.values[0] for ind in pareto]))
    distances = [[ind.fitness.values[1], ind.fitness.values[0]*scale] for ind in pareto]
    distances = [distance.euclidean(root, ind) for ind in distances]
    return pareto[distances.index(np.min(distances))]


def applyOps(population, toolbox, cxpb, mutpb, tmutpb, tmut=False):
    """
    Applies exclusive mutation, crossover, and transient mutation to a population. Adapted from deap.algorithms.varOr

    :param population: population to dynamize
    :param toolbox: toolbox reference
    :param cxpb: crossover probability
    :param mutpb: mutation probability
    :param tmutpb: transient mutation probability
    :param tmut: Is transient mutation allowed? Should be disallowed when len(TTS) = 0
    :return:
    """
    # Individuals are evolved pairwise
    for ind1, ind2 in zip(population[::2], population[1::2]):
        op_choice = rand.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = toolbox.mate(ind1, ind2)
        elif op_choice < cxpb+mutpb or not tmut:  # Apply mutation, OR when transient mutation isn't available
            ind1, = toolbox.mutate(ind1)
            ind2, = toolbox.mutate(ind2)
        elif op_choice < cxpb+mutpb+tmutpb:  # Apply transient mutation
            ind1, = toolbox.transient_mutate(ind1)
            ind2, = toolbox.transient_mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values
    return population


def draw_pareto(pareto, gen):
    """
    Draws the normalized pareto front of an evolution. USED FOR ANALYSIS

    :param pareto: the pareto front to graph
    :param gen: the current generation (appears in file name)
    :return:
    """
    pareto_x = [i.fitness.values[0] for i in pareto]
    pareto_y = [i.fitness.values[1] for i in pareto]
    scaler = max(pareto_y) / max(pareto_x)
    pareto_x = [ind*scaler for ind in pareto_x]
    plot.scatter(pareto_x, pareto_y, alpha=0.5)
    plot.title('Scatter plot pythonspot.com')
    plot.xlabel('Accuracy')
    plot.ylabel('Complexity')
    plot.xlim([0, max(pareto_x)])

    # Save the figure & display the plot
    path = Path.cwd() / '..' / 'docs' / 'Figures' / str('pareto_front_evo_' + str(gen))
    plot.savefig(fname=path)
    plot.clf()
