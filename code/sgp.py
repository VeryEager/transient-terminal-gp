"""
Contains code for single-objective GP algorithm using DEAP and TTS.

Written by Asher Stout, 300432820
"""
from deap.algorithms import varOr
from deap import base, creator, tools, gp
from sklearn.metrics import mean_squared_error
import operator as op
import ttsclasses as tts
import ttsfunctions as ttsf
import numpy
import shared

transient = tts.TransientSet(name="transient", arity=1, lifespan=5)


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
    # Initialize individual, fitness, population, etc
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", tts.TransientTree, fitness=creator.Fitness)
    tb.register("initialize", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, container=creator.Individual, generator=tb.initialize)
    tb.register("population", tools.initRepeat, container=list, func=tb.individual)
    tb.register("selection", tools.selTournament)
    tb.register("evaluation", rmse_evaluation, _tb=tb)
    tb.register("compile", gp.compile, pset=pset)

    # Register genetic operators & decorate bounds
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=1, max_=3)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    tb.decorate("mate", gp.staticLimit(key=op.attrgetter("height"), max_value=90))
    tb.decorate("mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=90))
    tb.register("expr_trans_mut", ttsf.genRand)
    tb.register("transient_mutate", ttsf.transientMutUniform, expr=tb.expr_trans_mut, pset=transient)
    tb.decorate("transient_mutate", gp.staticLimit(key=op.attrgetter('height'), max_value=90))


def evolve(data, labels, names, tdata, tlabels, generations=50, pop_size=100, cxpb=0.9, mutpb=0.1, tmutpb=0.1):
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
    :param tmutpb: transient mutation probability

    :return: the best individual of the evolution & the log
    """
    # Initialize toolbox, population, hof, and logs
    toolbox = base.Toolbox()
    primitives = shared.create_primitives(names, data.shape[1])
    create_definitions(toolbox, primitives)
    stats, logbook = shared.init_logger("gen", "best", "besttrain", "balanced")
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    # Update initial fitnesses & print log for 0th generation
    fitness = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in pop]
    for ind, fit in zip([ind for ind in pop if not ind.fitness.valid], fitness):
        ind.fitness.values = fit
    hof.update(pop)
    logbook.record(gen=0, best=tuple(list(toolbox.evaluation(function=hof[0], data=tdata, actual=tlabels))+[len(hof[0])]),
                   besttrain=tuple([hof[0].fitness.values]+[len(hof[0])]), balanced=hof[0].fitness.values, **stats.compile(pop))
    print(logbook.stream)

    # Begin evolution of population
    for g in range(1, generations):
        for ind in pop:
            ind.update_last()  # Update the metadata on evolution prior to this generation's evolution
        nextgen = toolbox.selection(pop, len(pop), tournsize=10)
        nextgen = [toolbox.clone(ind) for ind in nextgen]
        nextgen = shared.applyOps(nextgen, toolbox, cxpb, mutpb, tmutpb, (transient.trans_count > 0))

        # Update fitness & population, update HoF, record generation log
        invalidind = [ind for ind in nextgen if not ind.fitness.valid]
        fitness = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in invalidind]
        for ind, fit in zip(invalidind, fitness):
            ind.fitness.values = fit
        pop[:] = nextgen
        hof.update(nextgen)
        logbook.record(gen=g, best=tuple(list(toolbox.evaluation(function=hof[0], data=tdata, actual=tlabels))+[len(hof[0])]),
                       besttrain=hof[0].fitness.values, balanced=hof[0].fitness.values, **stats.compile(pop))
        print(logbook.stream)

        # Update Transient Terminal Set for next generation
        transient.update_set(pop, g)
    return hof[0], logbook
