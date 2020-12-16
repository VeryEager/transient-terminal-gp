"""
Contains general functions for use with a transient terminal set

Written by Asher Stout, 300432820
"""
from deap import gp
import random


######################################
# Transient Mutation                 #
######################################
def transMutUniform(individual, selector, ttset):
    """
    Select a random subtree of an individual and replace it with a member of the TTS.
    Adapted from gp.mutUniform

    :param individual: the individual to mutate
    :param selector: function which selects which member to replace the subtree with
    :returns: The mutated individual, as a tuple
    """
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = selector(pset=ttset, type_=type_)
    return individual,


def genRand(ttset):
    """
    Generates a subtree by pulling a random member of the TTS

    :param ttset: the Transient Terminal Set to select subtrees from
    :return: a random member of the ttset
    """
    return random.choice(ttset)

