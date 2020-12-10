"""
Contains code for the standard GP algorithm using DEAP.

Written by Asher Stout, 300432820
"""
from deap import base, creator, tools, gp
from shared import protected_division, eval_solution
import operator as op
import random as rand
import pandas
import numpy
import networkx


def create_primitives(attrs=1):
    """
    Creates the terminal/function sets for standard GP

    :param attrs: number of features present in the data
    :return: terminal_function_set, the generated primitive set
    """
    terminal_function_set = gp.PrimitiveSet(name="PSET", arity=attrs)
    terminal_function_set.addEphemeralConstant(name="PSET", ephemeral=lambda: rand.uniform(-5.0, 5.0))
    terminal_function_set.addPrimitive(op.add, 2)
    terminal_function_set.addPrimitive(op.sub, 2)
    terminal_function_set.addPrimitive(op.mul, 2)
    terminal_function_set.addPrimitive(protected_division, 2)
    terminal_function_set.addPrimitive(op.abs, 1)
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
    tb.register("initialize", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
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
    tb.register("evaluation", eval_solution, tb=tb)
    tb.register("compile", gp.compile, pset=pset)
    return


def main(data, labels, attrs, generations=50, pop_size=100, cxpb=0.5, mutpb=0.1):
    """
    Performs the setup for the main evolutionary process

    :return:
    """
    # Initialize toolbox & creator parameters
    toolbox = base.Toolbox()
    primitives = create_primitives(data.shape[1])
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
    fitnesses = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in pop]
    for ind, fit in zip([ind for ind in pop if not ind.fitness.valid], fitnesses):
        ind.fitness.values = fit
    logbook.record(gen=0, **stats.compile(pop))
    print(logbook.stream)

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

        # Update fitness & print log
        fitnesses = [toolbox.evaluation(function=ind, data=data, actual=labels) for ind in nextgen]
        for ind, fit in zip([ind for ind in nextgen if not ind.fitness.valid], fitnesses):
            ind.fitness.values = fit
        logbook.record(gen=g, **stats.compile(pop))
        print(logbook.stream)

        pop[:] = nextgen
    return


if __name__ == "__main__":
    # Load wine data
    winered = pandas.read_csv("winequality-red.csv", sep=";")
    winered_data = winered.drop(['quality'], axis=1).values
    winered_target = winered['quality'].values
    main(winered_data, winered_target, winered_data.shape[1])
