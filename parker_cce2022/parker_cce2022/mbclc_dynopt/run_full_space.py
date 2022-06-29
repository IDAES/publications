from time import time as get_current_time

import pyomo.environ as pyo

from idaes.apps.caprese.categorize import (
    VariableCategory as VC,
)

from parker_cce2022.mbclc_dynopt.model import (
    get_steady_state_data,
    get_dynamic_model,
)
from parker_cce2022.mbclc_dynopt.series_data import TimeSeriesTuple
from parker_cce2022.mbclc_dynopt.solve_data import SolveData
from parker_cce2022.mbclc_dynopt.config import get_ipopt

import pyomo.repn.plugins.ampl.ampl_ as nl_module

from pyomo.common.timing import HierarchicalTimer
TIMER = HierarchicalTimer()
nl_module.TIMER = TIMER


"""
This module defines a function for setting up and solving a
dynamic optimization problem instance using the full space
formulation.
"""


def run_dynamic_optimization(
        initial_conditions=None,
        setpoint=None,
        scalar_data=None,
        nxfe=10,
        nxcp=1,
        ntfe_per_sample=2,
        ntcp=1,
        sample_width=120.0,
        samples_per_horizon=15,
        ipopt_options=None,
        differential_bounds=False,
        input_bounds=False,
        ):
    m, var_cat, con_cat = get_dynamic_model(
        nxfe=nxfe,
        nxcp=nxcp,
        ntfe_per_sample=ntfe_per_sample,
        ntcp=ntcp,
        sample_width=sample_width,
        samples_per_horizon=samples_per_horizon,
        initial_data=initial_conditions,
        setpoint=setpoint,
        scalar_data=scalar_data,
    )
    time = m.fs.time
    if differential_bounds:
        # Set lower bounds on differential variables if desired
        for var in var_cat[VC.DIFFERENTIAL]:
            for t in time:
                var[t].setlb(0.0)
    if input_bounds:
        sample_points = [i*sample_width for i in range(samples_per_horizon + 1)]
        sample_points = [
            time.at(time.find_nearest_index(t)) for t in sample_points
        ]
        for t in sample_points:
            m.fs.MB.gas_inlet.flow_mol[t].setlb(0.0)
            m.fs.MB.gas_inlet.flow_mol[t].setub(200.0)

    ipopt = get_ipopt()
    TIMER.start("solve")
    if ipopt_options is None:
        ipopt_options = {
            "tol": 5e-5,
            # This is not a valid option for this IPOPT interface
            #"inf_pr_output": "internal",
            "dual_inf_tol": 1e2,
            "constr_viol_tol": 1e2,
            "compl_inf_tol": 1e2,
            "nlp_scaling_method": "user-scaling",
        }
    ipopt.options.update(ipopt_options)
    t_start = get_current_time()
    try:
        res = ipopt.solve(m, tee=True)
    except ValueError as err:
        # ValueError when restoration fails, I believe
        # TODO: return something useful here
        pass
    t_end = get_current_time()
    TIMER.stop("solve")
    # We frequently call this function and expect the solve to
    # fail, so we don't assert optimal termination here.
    # Instead, we just send the results back to the caller to
    # process however they want.
    #pyo.assert_optimal_termination(res)

    m.fs.MB.gas_inlet.flow_mol.pprint()
    m.fs.MB.solid_inlet.flow_mass.pprint()

    control_values = TimeSeriesTuple(
        {
            # Use strings as keys here for json-serializability
            str(pyo.ComponentUID(m.fs.MB.gas_inlet.flow_mol.referent)): [
                m.fs.MB.gas_inlet.flow_mol[t].value for t in m.fs.time
            ],
            str(pyo.ComponentUID(m.fs.MB.solid_inlet.flow_mass.referent)): [
                m.fs.MB.solid_inlet.flow_mass[t].value for t in m.fs.time
            ],
            str(pyo.ComponentUID(m.tracking_cost[:])): [
                pyo.value(m.tracking_cost[t]) for t in time
            ],
        },
        list(m.fs.time),
    )

    print(TIMER)
    solve_time = t_end - t_start
    solve_data = SolveData(res, control_values, solve_time)
    return solve_data


if __name__ == "__main__":
    nxfe = 10
    samples_per_horizon = 10
    ic_scalar_data, ic_dae_data = get_steady_state_data(
        nxfe=nxfe,
    )
    sp_input_map = {"fs.MB.solid_inlet.flow_mass[*]": 700.0}
    #sp_input_map = {"fs.MB.solid_inlet.flow_mass[*]": 591.4}
    sp_dof_names = ["fs.MB.gas_inlet.flow_mol[*]"]
    #sp_dof_names = ["fs.MB.solid_inlet.flow_mass[*]"]
    # Note that this key needs to have index [*,0], not [*,0.0]
    # This doesn't make sense to me. It seems like
    # str-> ComponentUID-> component-> slice-> ComponentUID-> str
    # is somehow changing this index from a float to an int...
    # It was, because of how ComponentUID is processing the string index.
    sp_state_list = [("fs.MB.solid_phase.reactions[*,0.0].OC_conv", 0.95)]
    #sp_state_list = [("fs.MB.solid_phase.properties[*,1.0].flow_mass", 591.5)]

    _, sp_dae_data = get_steady_state_data(
        nxfe=nxfe,
        input_map=sp_input_map,
        to_unfix=sp_dof_names,
        setpoint_list=sp_state_list,
    )

    run_dynamic_optimization(
        initial_conditions=ic_dae_data,
        setpoint=sp_dae_data,
        scalar_data=ic_scalar_data,
        nxfe=nxfe,
        samples_per_horizon=samples_per_horizon,
        differential_bounds=False,
        input_bounds=True,
    )
