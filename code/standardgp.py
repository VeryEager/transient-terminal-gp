"""
Contains code for the standard GP algorithm using DEAP.

Written by Asher Stout, 300432820
"""
from deap import base, creator, tools, gp
import operator as op
import random as rand
import numpy
import os.path
import pandas
import shared

rand.seed(96127708431)


def create_primitives(names, attrs=1):
    """
    Creates the terminal/function sets for standard GP

    :param names: the names of features (as would appear in graph)
    :param attrs: number of features present in the data
    :return: terminal_function_set, the generated primitive set
    """
    terminal_function_set = gp.PrimitiveSet(name="PSET", arity=attrs)
    terminal_function_set.addEphemeralConstant(name="PSET", ephemeral=lambda: rand.uniform(-1.0, 1.0))
    terminal_function_set.addPrimitive(op.add, 2)
    terminal_function_set.addPrimitive(op.sub, 2)
    terminal_function_set.addPrimitive(op.mul, 2)
    terminal_function_set.addPrimitive(shared.protected_division, 2)

    # Replace whitespaces with '_' in names
    n = []
    for name in names:
       n.append(name.replace(" ", "_"))
    # TODO this is an utterly horrible, hardcoded solution to setting multiple feature names.
    terminal_function_set.renameArguments(ARG0=n[0], ARG1=n[1], ARG2=n[2], ARG3=n[3], ARG4=n[4], ARG5=n[5], ARG6=n[6],
                                          ARG7=n[7], ARG8=n[8], ARG9=n[9], ARG10=n[10])
    return terminal_function_set


def create_definitions(tb, pset):
    """
    Initializes a variety of parameters using the DEAP creator & toolbox

    :param tb: reference to the DEAP toolbox
    :param pset: the primitive set
    :return:
    """
    # Initialize individual, fitness, and population
    creator.create("MOFitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.MOFitness)
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
    primitives = create_primitives(names, data.shape[1])
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
        logbook.record(gen=g, **stats.compile(pop))
        print(logbook.stream)

        # Replace population, update HoF
        pop[:] = nextgen
        hof.update(pop)
        print(len(hof))
    return hof[0], logbook


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

