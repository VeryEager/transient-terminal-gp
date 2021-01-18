"""
Contains general functions for use with a transient terminal set

Written by Asher Stout, 300432820
"""
import random


######################################
# Transient Mutation                 #
######################################
def transientMutUniform(individual, expr, pset):
    """
    Select a random subtree of an individual and replace it with a member of the TTS.
    Adapted from gp.mutUniform

    :param individual: the individual to mutate
    :param expr: function which selects which member to replace the subtree with
    :param pset: the TTS
    :returns: The mutated individual, as a tuple
    """
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    subtree_ = expr(pset=pset, type_=type_)
    individual[slice_] = list(subtree_.tree)
    return individual,


def genRand(pset, type_=None):
    """
    Generates a subtree by pulling a random member of the TTS

    :param pset: the Transient Terminal Set to select subtrees from
    :param type_: return type of the tree during execution
    :return: a random member of the ttset
    """
    subtree = random.choice(pset.transient)
    return subtree


def _percent_improve(ind, index=0):
    """
    Calculates the % improvement in a solution's fitness acc. to the formula (b-a)/a

    :param ind: the solution
    :param index: the index of the fitness calculated. 0 in the case of single-fitness
    :return: the % improvement in results
    """
    return (ind.fitness.values[index]-ind.former.fitness.values[index])/ind.former.fitness.values[index]
