from collections import namedtuple

"""
Tuples to hold data resulting from optimization solves and parameter
sweeps of optimization solves.
"""

# A tuple for holding metadata and a list of SweepData "results" objects
# from a parameter sweep
SweepDataContainer = namedtuple("SweepDataContainer", ["metadata", "results"])

# A tuple to hold the inputs, the setpoint data structure, and the
# solve status (including solve time and dof variable values) resulting
# from a dynamic optimization solve.
# Note that there is some redundancy in this data. From the inputs,
# we could compute the setpoint (if we know exactly what steady state
# optimization problem to solve).
SweepData = namedtuple("SweepData", ["inputs", "setpoint", "solve"])
