"""
Contains code for the transient-terminal GP algorithm using DEAP.

Written by Asher Stout, 300432820
"""
from deap import base, creator, tools, gp

transient = gp.PrimitiveSet(name="transient", arity=1)


def update_transient():
    """
    Updates the transient terminal set by adding new subtrees and removing deprecated ones.
    """
    pass
    return
