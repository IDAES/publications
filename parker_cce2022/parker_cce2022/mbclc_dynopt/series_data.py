from collections import namedtuple

"""
This is a tuple intended to contain time-indexed data used
as an input or output from an optimization problem.
"""

TimeSeriesTuple = namedtuple("TimeSeriesTuple", ["data", "time"])
