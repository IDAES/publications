import os
import pyomo.environ as pyo

def get_ipopt():
    ipopt_exec = os.environ.get("IPOPT_EXECUTABLE", None)
    if ipopt_exec is None:
        return pyo.SolverFactory("ipopt")
    else:
        return pyo.SolverFactory("ipopt", executable=ipopt_exec)
