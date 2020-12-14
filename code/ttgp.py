"""
Contains code for the transient-terminal GP algorithm using DEAP.

Written by Asher Stout, 300432820
"""
from deap import base, creator, tools, gp
import operator as op
import random as rand
import numpy
import os.path
import pandas
import shared
import ttsclasses as tts

rand.seed(shared.seed)
transient = tts.TransientSet(name="transient", arity=1, lifespan=5)


def create_definitions(tb, pset):
    """
    Initializes a variety of parameters using the DEAP creator & toolbox

    :param tb: reference to the DEAP toolbox
    :param pset: the primitive set
    :return:
    """
    # Initialize individual, fitness, and population
    creator.create("MOFitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", tts.TransientTree, fitness=creator.MOFitness)
    tb.register("initialize", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, container=creator.Individual, generator=tb.initialize)
    tb.register("population", tools.initRepeat, container=list, func=tb.individual)

    # Register genetic operators & decorate bounds
    tb.register("crossover", gp.cxOnePoint)
    tb.decorate("crossover", gp.staticLimit(key=op.attrgetter("height"), max_value=12))
    tb.register("expr_mut", gp.genFull, min_=1, max_=3)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    tb.decorate("mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=12))
    tb.register("expr_trans_mut", gp.genFull, min_=1, max_=1)
    tb.register("transient_mutate", gp.mutUniform, expr=tb.expr_trans_mut, pset=transient)
    tb.decorate("transient_mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=12))

    # Register selection, evaluation, compiliation
    tb.register("selection", tools.selTournament, tournsize=5)
    tb.register("evaluation", shared.eval_solution, tb=tb)
    tb.register("compile", gp.compile, pset=pset)
    return


def main(data, labels, attrs, names, generations=50, pop_size=100, cxpb=0.5, mutpb=0.1):
    """
    Performs the setup for the main evolutionary process

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
    logbook.header = "gen", "min", "mean", "max", "std"

    # Initialize population & compute initial fitnesses
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    fitness = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in pop]
    for ind, fit in zip([ind for ind in pop if not ind.fitness.valid], fitness):
        ind.fitness.values = fit

    logbook.record(gen=0, **stats.compile(pop))
    print(logbook.stream)
    hof.update(pop)

    # Begin evolution of population
    for g in range(1, generations):
        nextgen = toolbox.selection(pop, len(pop))
        nextgen = [toolbox.clone(ind) for ind in nextgen]
        for ind in nextgen:
            ind = ind.update_last()  # Update the metadata on evolution

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

        # Perform transient mutation
        for ind in nextgen:
            if (rand.random() < mutpb/4) & (transient.terms_count > 1):
                toolbox.transient_mutate(ind)
                del ind.fitness.values

        # Update fitness
        invalidind = [ind for ind in nextgen if not ind.fitness.valid]
        fitness = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in invalidind]
        for ind, fit in zip(invalidind, fitness):
            ind.fitness.values = fit

        # Record generational log
        logbook.record(gen=g, **stats.compile(pop))
        print(logbook.stream)

        # Replace population, update HoF
        pop[:] = nextgen
        hof.update(pop)

        # Update Transient Terminal Set after each generation
        transient.update_set(pop)
    return hof[10], logbook


if __name__ == "__main__":
    # Load wine data
    path = os.path.relpath('..\\data\\winequality-red.csv', os.path.dirname(__file__))

    winered = pandas.read_csv(path, sep=";")
    winered_data = winered.drop(['quality'], axis=1).values
    winered_target = winered['quality'].values

    # Evolve population, then draw descent & trees
    best, logs = main(winered_data, winered_target, winered_data.shape[1], winered.columns.drop(['quality']))
    shared.draw_descent(logs, measure='min')
    shared.draw_solution(best)