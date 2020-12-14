"""
This file contains classes and functions requried for the implementation of Transient Terminal Set GP

Written by Asher Stout, 300432820
"""

from deap.gp import PrimitiveTree, PrimitiveSet


class TransientTree(PrimitiveTree):
    """
    Represents a Primitive Tree used in GP, containing both the current & previous tree
    (as well as relevant fitness values), so that meta-information on evolution can be
    stored & analyzed during transient mutation.
    """
    last = PrimitiveTree([])  # Tree of the previous generation; should be empty to begin

    def update_last(self):
        """
        Updates the last generation tree by overwriting the existing entry with the current

        :return:
        """
        last = self


class TransientSet(PrimitiveSet):
    """
    Represents a dynamic set which is updated after each consecutive generation. The set
    contains subtrees pulled from the population which are subsequently mutated.
    """
    lifespan = 5              # Number of generations before an entry is removed

    def update_set(self, population):
        """
        Updates the transient terminal set by adding new subtrees and removing deprecated ones.

        :param population: the population in the current generation
        :return: the updated transient terminal set
        """
        ntran = population
        return ntran
