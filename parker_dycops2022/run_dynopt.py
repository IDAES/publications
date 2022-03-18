import pyomo.environ as pyo
import pyomo.dae as dae

from pyomo.dae.flatten import flatten_dae_components
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.common.collections import ComponentSet
#from pyomo.common.timing import HierarchicalTimer
from workspace.common.timing import HierarchicalTimer
from workspace.common.initialize import initialize_by_time_element

import workspace.mbclc.model as model_module
from workspace.mbclc.model import (
    make_square_dynamic_model,
    make_square_model,
    add_constraints_for_missing_variables,
)
import workspace.mbclc.initialize as initialize_module
from workspace.mbclc.initialize import (
    set_default_design_vars,
    set_default_inlet_conditions,
    initialize_steady,
    initialize_dynamic_from_steady,
)
from workspace.mbclc.plot import (
    plot_outlet_states_over_time,
    plot_inputs_over_time,
)
from workspace.mbclc.experiments.dynamic_opt.load_inputs import (
    load_inputs_into_model,
    get_inputs_at_time,
)
import workspace.mbclc.results.dycops2022.model as dynopt_module
from workspace.mbclc.results.dycops2022.model import (
    get_steady_state_model,
    get_data_from_steady_model,
    get_model_for_dynamic_optimization,
    initialize_dynamic,
    get_state_variable_names,
    get_tracking_cost_expressions,
)
from workspace.mbclc.results.dycops2022.scaling import (
    get_max_values_from_steady,
)
from workspace.mbclc.results.dycops2022.deserialize import (
    get_variance_data,
    get_scaling_factor_data,
    get_variance_of_time_slices,
    get_scaling_of_time_slices,
)


TIMER = HierarchicalTimer()
initialize_module.TIMER = TIMER
model_module.TIMER = TIMER
dynopt_module.TIMER = TIMER


def main():
    """
    """
    nxfe = 10
    # NOTE: Default inputs: (128.2, 591.4)
    ic_inputs = {
        #"fs.MB.gas_phase.properties[*,0.0].flow_mol": 120.0,
        #"fs.MB.solid_phase.properties[*,1.0].flow_mass": 550.0,
    }
    ic_model_params = {"nxfe": nxfe}
    m_ic = get_steady_state_model(
        ic_inputs,
        solve_kwds={"tee": True},
        model_params=ic_model_params,
    )
    time = m_ic.fs.time
    scalar_data, dae_data = get_data_from_steady_model(m_ic, time)
    # TODO: get steady state data (scalar and dae both necessary)
    x0 = 0.0
    x1 = 1.0

    # These (as well as nxfe above) are the parameters for a small model
    # that I'm using to test NMPC (defaults are 900, 15, 60), nxfe=10
    horizon = 1800
    tfe_width = 60
    sample_width = 120
    sample_points = [
        # Calculate sample points first with integer arithmetic
        # to avoid roundoff error
        float(sample_width*i) for i in range(0, horizon//sample_width + 1)
    ]
    horizon = float(horizon)
    tfe_width = float(tfe_width)
    model_params = {
        "horizon": horizon,
        "tfe_width": tfe_width,
        "ntcp": 1,
        "nxfe": nxfe,
    }

    # These are approximately the default values:
    #disturbance_dict = {"CO2": 0.03, "H2O": 0.0, "CH4": 0.97}
    disturbance_dict = {"CO2": 0.5, "H2O": 0.0, "CH4": 0.5}
    disturbance = dict(
        (
            "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[%s]" % (x0, j),
            {(0.0, horizon): val},
        )
        for j, val in disturbance_dict.items()
    )

    # Create solver here as it is needed to solve for the setpoint
    solver = pyo.SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["max_cpu_time"] = 1500

    #
    # Get setpoint data
    #
    sp_inputs = get_inputs_at_time(disturbance, horizon)
    #sp_inputs.update({
    #    "fs.MB.gas_phase.properties[*,0.0].flow_mol": 272.8,
    #    "fs.MB.solid_phase.properties[*,1.0].flow_mass": 591.4,
    #})
    sp_model_params = {"nxfe": nxfe}
    m_sp = get_steady_state_model(
        sp_inputs,
        solve_kwds={"tee": True},
        model_params=sp_model_params,
    )
    time = m_sp.fs.time
    space = m_sp.fs.MB.gas_phase.length_domain
    t0 = time.first()
    # Solve optimization problem for setpoint
    sp_objective_states = [
        "fs.MB.solid_phase.reactions[*,%s].OC_conv" % x0,
    ]
    sp_target = {
        "fs.MB.solid_phase.reactions[*,%s].OC_conv" % x0: 0.95,
    }
    m_sp.fs.MB.gas_inlet.flow_mol[:].unfix()
    m_sp.setpoint_expr = get_tracking_cost_expressions(
        sp_objective_states, time, sp_target
    )
    m_sp.objective = pyo.Objective(expr=m_sp.setpoint_expr[t0])
    solver.solve(m_sp, tee=True)
    scalar_vars, dae_vars = flatten_dae_components(m_sp, time, pyo.Var)
    setpoint = {
        str(pyo.ComponentUID(var.referent)): var[t0].value
        for var in dae_vars
    }
    ###

    max_data = get_max_values_from_steady(m_sp)
    variance_data = get_variance_of_time_slices(m_sp, time, space)
    #weight_data = None
    weight_data = {
        name: 1.0/s if s != 0 else 1.0 for name, s in variance_data.items()
        #name: 1/w if w != 0 else 1.0 for name, w in max_data.items()
        # Note: 1/w**2 does not converge with states in objective...
    }
    objective_states = get_state_variable_names(space)

    flattened_vars = [None, None]
    m = get_model_for_dynamic_optimization(
        sample_points=sample_points,
        parameter_perturbation=disturbance,
        model_params=model_params,
        ic_scalar_data=scalar_data,
        ic_dae_data=dae_data,
        setpoint_data=setpoint,
        objective_weights=weight_data,
        objective_states=objective_states,

        # this argument is a huge hack to get the flattened
        # vars without having to do a bit more work.
        flatten_out=flattened_vars,
    )
    add_constraints_for_missing_variables(m)
    time = m.fs.time
    t0 = time.first()
    scalar_vars, dae_vars = flattened_vars
    initialize_dynamic(m, dae_vars)

    # Should we initialize to setpoint inputs?
    #sp_input_dict = {
    #    "fs.MB.gas_phase.properties[*,0.0].flow_mol": {(t0, horizon): 250.0},
    #    "fs.MB.solid_phase.properties[*,1.0].flow_mass": {(t0, horizon): 591.4},
    #}
    #load_inputs_into_model(m, time, sp_input_dict)

    # TODO: Should I set inlet flow rates to their target values for
    # this simulation?
    input_vardata = (
        [m.fs.MB.gas_inlet.flow_mol[t] for t in time if t != t0]
        + [m.fs.MB.solid_inlet.flow_mass[t] for t in time if t != t0]
    )
    with TemporarySubsystemManager(
            to_fix=input_vardata,
            to_deactivate=[m.piecewise_constant_constraint],
            ):
        print("Initializing by time element...")
        with TIMER.context("elem-init"):
            initialize_by_time_element(m, time, solver)

    m.fs.MB.solid_phase.reactions[:,0.0].OC_conv.setlb(0.89)

    print("Starting dynamic optimization solve...")
    with TIMER.context("solve dynamic"):
        solver.solve(m, tee=True)

    extra_states = [
        pyo.Reference(m.fs.MB.solid_phase.reactions[:,0.0].OC_conv),
    ]
    plot_outlet_states_over_time(m, show=False, extra_states=extra_states)
    inputs = [
        "fs.MB.gas_phase.properties[*,0.0].flow_mol",
        "fs.MB.solid_phase.properties[*,1.0].flow_mass",
    ]
    plot_inputs_over_time(m, inputs, show=False)
    print(m.tracking_cost.name)
    for t in m.fs.time:
        print(t, pyo.value(m.tracking_cost[t]))
    print()


if __name__ == "__main__":
    with TIMER.context("main"):
        main()
    print(TIMER)
