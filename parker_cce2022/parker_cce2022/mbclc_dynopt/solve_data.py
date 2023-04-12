from collections import namedtuple

"""
This is a tuple containing the data we need to analyze the result
of an attempt to solve a problem instance. Tuples of this type are
returned by the dynamic optimization functions in the full space
and implicit function files.
"""

SolveData = namedtuple("SolveData", ["status", "values", "time"])
