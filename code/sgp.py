"""
Contains code for single-objectove GP algorithm using DEAP.

Written by Asher Stout, 300432820
"""
from deap import base, creator, tools, gp
from sklearn.metrics import mean_squared_error
import operator as op
import random as rand
import numpy
import os.path
import pandas
import shared


def rmse_evaluation(function, data, actual, _tb):
    """
    Evaluates the RMSE of a candidate solution. Used for SOGP

    :param function: the candidate solution to evaluate
    :param data: points to evaluate on
    :param actual: the correct predictions for the data
    :param _tb:
    :return: the accuracy metric, as a tuple
    """
    if function is None:  # In exceptional circumstances only
        return numpy.Infinity,

    func = _tb.compile(expr=function)
    results = [func(*res) for res in data]

    accuracy = mean_squared_error(actual, results, squared=False)
    return accuracy,


def create_definitions(tb, pset):
    """
    Initializes a variety of parameters using the DEAP creator & toolbox

    :param tb: reference to the DEAP toolbox
    :param pset: the primitive set
    :return:
    """
    # Initialize individual, fitness, and population
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    tb.register("initialize", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, container=creator.Individual, generator=tb.initialize)
    tb.register("population", tools.initRepeat, container=list, func=tb.individual)

    # Register genetic operators & decorate bounds
    tb.register("crossover", gp.cxOnePoint)
    tb.decorate("crossover", gp.staticLimit(key=op.attrgetter("height"), max_value=12))
    tb.register("expr_mut", gp.genFull, min_=1, max_=3)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    tb.decorate("mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=12))

    # Register selection, evaluation, compiliation
    tb.register("selection", tools.selTournament)
    tb.register("evaluation", rmse_evaluation, _tb=tb)
    tb.register("compile", gp.compile, pset=pset)
    return


def evolve(data, labels, names, tdata, tlabels, generations=50, pop_size=100, cxpb=0.9, mutpb=0.1):
    """
    Performs the setup for the main evolutionary process
    :param data: training data to use during evolution
    :param labels: target variables for the training data
    :param names: names for primitives of the data, used for constructing the primitive set
    :param tdata: testing data used during solution evaluation
    :param tlabels: target variables for the testing data
    :param generations: number of generations
    :param pop_size: population size
    :param cxpb: crossover probability
    :param mutpb: mutation probability

    :return: the best individual of the evolution & the log
    """
    # Initialize toolbox & creator parameters
    toolbox = base.Toolbox()
    primitives = shared.create_primitives(names, data.shape[1])
    create_definitions(toolbox, primitives)

    # Initialize stats & logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("mean", numpy.mean, axis=0)
    stats.register("max", numpy.max, axis=0)
    stats.register("std", numpy.std, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "min", "mean", "max", "std", "best", "bestsize"

    # Initialize population & compute initial fitnesses
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    fitness = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in pop]
    for ind, fit in zip([ind for ind in pop if not ind.fitness.valid], fitness):
        ind.fitness.values = fit

    hof.update(pop)
    logbook.record(gen=0, best=toolbox.evaluation(function=hof[0], data=tdata, actual=tlabels), **stats.compile(pop))
    print(logbook.stream)

    # Begin evolution of population
    for g in range(1, generations):
        nextgen = toolbox.selection(pop, len(pop), tournsize=10)
        nextgen = [toolbox.clone(ind) for ind in nextgen]

        # Perform crossover
        for child1, child2 in zip(nextgen[::2], nextgen[1::2]):
            if rand.random() < cxpb:
                toolbox.crossover(child1, child2)
                del child1.fitness.values, child2.fitness.values

        # Perform mutation
        for ind in nextgen:
            if rand.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Update fitness
        invalidind = [ind for ind in nextgen if not ind.fitness.valid]
        fitness = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in invalidind]
        for ind, fit in zip(invalidind, fitness):
            ind.fitness.values = fit

        # Record generational log
        logbook.record(gen=g, best=toolbox.evaluation(function=hof[0], data=tdata, actual=tlabels), bestsize=len(hof[0]),
                       **stats.compile(pop))
        print(logbook.stream)

        # Replace population, update HoF
        pop[:] = nextgen
        hof.update(pop)
    return hof[0], logbook
