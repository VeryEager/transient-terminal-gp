"""
This file contains classes and functions requried for the implementation of Transient Terminal Set GP

Written by Asher Stout, 300432820
"""

from deap.gp import PrimitiveTree, PrimitiveSet, Terminal
from numpy import mean, percentile
from datetime import datetime
from collections import defaultdict
from os.path import commonprefix


class TransientTree(PrimitiveTree):
    """
    Represents a Primitive Tree used in GP, containing both the current & previous tree
    (as well as relevant fitness values), so that meta-information on evolution can be
    stored & analyzed during transient mutation.
    """

    def __init__(self, content):
        self.former = None
        PrimitiveTree.__init__(self, content)

    def update_last(self):
        """
        Updates the last generation's tree by overwriting the existing entry with the current

        :return:
        """
        self.former = self.__deepcopy__({})
        return

    def difference(self):
        """
        Computes the subtree difference between the current & former tree

        :return: the computed subtree
        """
        # Remove prefix of trees
        temp = self.__deepcopy__({})
        prefix = self.__prefix(False)
        [temp.remove(pre) for pre in prefix]
        temp.reverse()
        prefix = self.__prefix(True)
        [temp.remove(pre) for pre in prefix]
        temp.reverse()
        return temp

    def __prefix(self, reverse=False):
        """
        Computes the longest possible prefix of a tree

        :return: The longest prefix between the current & previous tree
        """
        # TODO: bugged currently, this does not take into account that the subtree change may be partially identical
        # to the existing one
        sel = self.__deepcopy__({})
        frm = self.former.__deepcopy__({})
        if reverse:  # If needed, reverse
            sel.reverse()
            frm.reverse()
        prefix = []
        for i in range(0, len(sel)):
            if sel[i] == frm[i]:
                prefix.append(sel[i])
            else:
                break
        return prefix


class TransientSet(PrimitiveSet):
    """
    Represents a dynamic set which is updated after each consecutive generation. The set
    contains subtrees pulled from the population and are subsequently used during mutation.
    """

    def __init__(self, name, arity, lifespan):
        PrimitiveSet.__init__(self, name, arity)
        self.transient = []
        self.trans_count = 0
        self.lifespan = lifespan
        self.entry_life = []

    def update_set(self, population, generation):
        """
        Updates the transient terminal set by adding new subtrees and removing deprecated ones.

        :param population: the population in the current generation
        :param generation: the current generation
        :return:
        """
        ntran = self

        # Remove deprecated subtrees
        self.entry_life = [life-1 for life in self.entry_life]
        for life in self.entry_life:
            if life + self.lifespan < generation:
                self.entry_life.pop()
                self.removeOldestSubtree()

        # Calculate mean change in fitness measures
        acc_threshold = mean([ind.former.fitness.values[0]-ind.fitness.values[0] for ind in population])
        com_threshold = mean([ind.former.fitness.values[1]-ind.fitness.values[1] for ind in population])

        for ind in population:
            acc = ind.former.fitness.values[0]-ind.fitness.values[0]
            com = ind.former.fitness.values[1]-ind.fitness.values[1]  # Add subtree if it improves enough
            if (acc > 0) & (acc > acc_threshold) & (com > 0) & (com > com_threshold):
                diff = ind.difference()
                subtree = TransientSubtree(diff, str(diff))
                if diff and not self.context.keys().__contains__(subtree.name):  # TODO: diff should never be none, this is a temporary solution
                    self.addSubtree(subtree)
                    self.entry_life.append(generation)
        return ntran

    def addSubtree(self, subtree):
        """
        Adds a subtree (composed of primitives & terminals) to the TTS

        :param subtree: subtree to add
        :param name: internal name of the subtree
        :return:
        """
        self.mapping[subtree.name] = subtree
        self.context[subtree.name] = subtree
        self.transient.append(subtree)
        self.trans_count += 1
        return

    def removeOldestSubtree(self):
        """
        Removes the oldest subtree from the TTS. As subtree are added sequentially, the oldest subtrees
        are stored in the front of the set.

        :return:
        """
        print("OLD", [s.name for s in self.transient])
        subtree = self.transient.pop(0)
        del self.mapping[subtree.name]
        del self.context[subtree.name]
        self.trans_count -= 1
        print("NEW", [s.name for s in self.transient])

        return


class TransientSubtree(object):
    """
    Defines a subtree used during transient mutation; composed of a list of primitives/terminals

    """
    def __init__(self, tree, name):
        self.tree = tree
        self.name = name

    def arity(self):
        """
        Arity of a Transient Subtree is always 0

        :return:
        """
        return 0

    def format(self):
        form = ""
        form = [form + node.format() for node in self.tree]
        return form


