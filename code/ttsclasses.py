"""
This file contains classes and functions requried for the implementation of Transient Terminal Set GP

Written by Asher Stout, 300432820
"""
from ttsfunctions import _percent_improve
from deap.gp import PrimitiveTree, PrimitiveSet
from numpy import percentile, abs
from scipy import stats


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

        :return: the computed subtree difference
        """
        diff = []
        i = 0
        while i != len(self):
            if self[i] == self.former[i]:
                # If there is more than one changed subtree, then add all subtrees & parent node as the change
                changed_subtrees = 0
                new_subtrees = PrimitiveTree(self[self.searchSubtree(i)])
                new_subtrees.pop(0)
                old_subtrees = PrimitiveTree(self.former[self.former.searchSubtree(i)])
                old_subtrees.pop(0)
                while len(new_subtrees) > 0:
                    # Evaluate changes between this subtree
                    new_sub = new_subtrees.searchSubtree(0)
                    old_sub = old_subtrees.searchSubtree(0)
                    if new_subtrees[new_sub] != old_subtrees[old_sub]:
                        changed_subtrees += 1
                    del new_subtrees[new_sub]
                    del old_subtrees[old_sub]
                if changed_subtrees > 1:
                    diff = self.searchSubtree(i)
                    break
                # Otherwise just increment to next node
                else:
                    i += 1
                pass
            # If nodes do not match then only one changed subtree exists; return it
            else:
                diff = self.searchSubtree(i)
                break
        return self[diff]


class TransientSet(PrimitiveSet):
    """
    Represents a dynamic set which is updated after each consecutive generation. The set
    contains subtrees pulled from the population and are subsequently used during mutation.
    """

    def __init__(self, name, arity, lifespan=5, thresh=90):
        PrimitiveSet.__init__(self, name, arity)
        self.trans_count = 0
        self.transient = []
        self.entry_life = []
        self.lifespan = lifespan
        self.thresh = thresh

    def update_set(self, population, generation):
        """
        Updates the transient terminal set by adding new subtrees and removing deprecated ones.

        :param population: the population in the current generation
        :param generation: the current generation
        :return:
        """

        # Remove deprecated subtrees
        self.entry_life = [life-1 for life in self.entry_life]
        for life in self.entry_life:
            if life + self.lifespan < generation:
                self.entry_life.pop()
                self.removeOldestSubtree()

        fitness_count = range(0, len(population[0].fitness.values))   # How many objectives do we have?

        # Calculate the % improvement in the population, and remove all which are not Pareto improvements
        relative_improvements = [[_percent_improve(ind, i) for ind in population] for i in fitness_count]
        relative_improvements = [ind for i, ind in enumerate(population) if all([relative_improvements[n][i] <= 0 for n
                                                                                 in fitness_count])]
        # Then calculate the Nth percentile improvement using the % improvement
        relative_improvements = [[abs(_percent_improve(ind, i)) for ind in relative_improvements] for i in
                                 fitness_count]  # needs to be abs() so percentiles correctly work
        change_thresholds = [percentile(relative_improvements[i], q=self.thresh) for i in fitness_count]

        for ind in population:
            changes = [_percent_improve(ind, i) for i in fitness_count]
            # If the change in fitness is valid and the subtree is not already in the TTS, then include it
            if all([(changes[i] < 0 and abs(changes[i]) >= change_thresholds[i]) for i in fitness_count]):
                diff = ind.difference()
                subtree = TransientSubtree(diff, str(diff))
                if not self.context.keys().__contains__(subtree.name):
                    self.addSubtree(subtree)
                    self.entry_life.append(generation)

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
        subtree = self.transient.pop(0)
        del self.mapping[subtree.name]
        del self.context[subtree.name]
        self.trans_count -= 1


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


