import os
import pyomo.environ as pyo


def get_nominal_values():
    return [
        # NOTE: It is dangerous to use these strings as keys in a dictionary,
        # as a string with "0.0" replaced by "0" will correspond to a
        # different item, without there being a corresponding different Pyomo
        # variable.
        ("fs.MB.gas_phase._flow_terms[0.0,*,Vap,CH4]", 100.0),
        ("fs.MB.gas_phase._flow_terms[0.0,*,Vap,H2O]", 100.0),
        ("fs.MB.gas_phase._flow_terms[0.0,*,Vap,CO2]", 100.0),
        ("fs.MB.gas_phase._enthalpy_flow[0.0,*,Vap]", 1e3),
        ("fs.MB.gas_phase.pressure[0.0,*]", 1.0),
        ("fs.MB.solid_phase._flow_terms[0.0,*,Sol,Fe2O3]", 100.0),
        ("fs.MB.solid_phase._flow_terms[0.0,*,Sol,Fe3O4]", 100.0),
        ("fs.MB.solid_phase._flow_terms[0.0,*,Sol,Al2O3]", 100.0),
        ("fs.MB.solid_phase._enthalpy_flow[0.0,*,Sol]", 1e5),
        ]


def get_ipopt():
    ipopt_exec = os.environ.get("IPOPT_EXECUTABLE", None)
    if ipopt_exec is not None:
        ipopt = pyo.SolverFactory("ipopt", executable=ipopt_exec)
    else:
        ipopt = pyo.SolverFactory("ipopt")
    return ipopt


def get_cyipopt(options=None):
    return pyo.SolverFactory("cyipopt", options=options)


def get_ipopt_options():
    return {
            "nlp_scaling_method": "user-scaling",
            "inf_pr_output": "internal",
            "tol": 5e-5,
            "dual_inf_tol": 1.0,
            "constr_viol_tol": 1.0,
            "compl_inf_tol": 1.0,
            #"max_cpu_time": 240.0,
            "max_iter": 3000,
            }


def get_temperature_list():
    return [1800.0, 1750.0, 1700.0, 1650.0, 1600.0, 1550.0, 1500.0, 1450.0,
            1400.0, 1350.0, 1300.0, 1250.0, 1200.0, 1150.0,
            1100.0, 1050.0, 1000.0, 950.0, 900.0, 850.0,
            800.0, 750.0, 700.0, 650.0, 600.0]


def get_nxfe_list():
    return [10, 20, 30, 40, 50]


class MockSolverResults(object):

    def __init__(self, **kwargs):
        self.termination_condition = kwargs.pop("termination_condition", None)
        self.wallclock_time = kwargs.pop("wallclock_time", None)
        self.time = kwargs.pop("time", None)


class MockResults(object):

    def __init__(self, **kwargs):
        termination_condition = kwargs.pop("termination_condition", None)
        wallclock_time = kwargs.pop("wallclock_time", None)
        time = kwargs.pop("time", None)
        self.solver = MockSolverResults(
                termination_condition=termination_condition,
                wallclock_time=wallclock_time,
                time=time,
                )
