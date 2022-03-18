import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components
from pyomo.common.collections import ComponentSet

from workspace.common.timing import HierarchicalTimer
from workspace.common.dynamic_data import (
    set_values,
    set_values_at_time,
    load_inputs_into_model,
    get_inputs_at_time,
    get_values_from_model_at_time,
    copy_values_from_time,
)
import workspace.mbclc.model as model_module
from workspace.mbclc.model import (
    make_square_model,
    make_square_dynamic_model,
    add_constraints_for_missing_variables,
    VariableCategory as VC,
    ConstraintCategory as CC,
)
import workspace.mbclc.initialize as initialize_module
from workspace.mbclc.initialize import (
    set_default_design_vars,
    set_default_inlet_conditions,
    initialize_steady,
    initialize_dynamic_from_steady,
)
from workspace.common.dae_utils import (
    generate_diff_deriv_disc_components_along_set,
)

TIMER = HierarchicalTimer()
initialize_module.TIMER = TIMER
model_module.TIMER = TIMER


def get_tracking_cost_expressions(states, time, setpoint, weights=None):
    if weights is None:
        weights = {name: 1.0 for name in states}
    # No more "implicit weights."
    # Either you supply all weights or no weights.
    #for name in states:
    #    if name not in weights:
    #        weights[name] = 1.0
    def tracking_cost_rule(m, t):
        return sum(
            weights[name]*(m.find_component(name)[t] - setpoint[name])**2
            for name in states
        )
    tracking_cost = pyo.Expression(time, rule=tracking_cost_rule)
    return tracking_cost


def get_piecewise_constant_constraints(inputs, time, sample_points):
    sample_point_set = set(sample_points)
    def piecewise_constant_rule(m, t, i):
        if t in sample_points:
            return pyo.Constraint.Skip
        else:
            var = m.find_component(inputs[i])
            t_next = time.next(t)
            return var[t_next] - var[t] == 0
    pwc_con = pyo.Constraint(
        time, range(len(inputs)), rule=piecewise_constant_rule
    )
    return pwc_con


def get_steady_state_model(inputs=None, solve_kwds=None, model_params=None):
    """
    Maps scalar data to a steady state model.
    """
    if solve_kwds is None:
        solve_kwds = {}
    if model_params is None:
        model_params = {}
    with TIMER.context("make steady"):
        m_steady = make_square_model(steady=True, **model_params)
    set_default_design_vars(m_steady)
    set_default_inlet_conditions(m_steady)
    time = m_steady.fs.time
    t0 = time.first()
    if inputs is not None:
        for name, val in inputs.items():
            var = m_steady.find_component(name)
            var[t0].set_value(val)
    with TIMER.context("initialize steady"):
        initialize_steady(m_steady)
    solver = pyo.SolverFactory("ipopt")
    with TIMER.context("solve steady"):
        solver.solve(m_steady, **solve_kwds)
    return m_steady


def get_state_variable_names(space):
    """
    These are a somewhat arbitrary set of time-indexed variables that are
    (more than) sufficient to solve for the rest of the variables.
    They happen to correspond do the property packages' state variables.

    """
    #space = m.fs.MB.gas_phase.length_domain
    setpoint_states = []
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].flow_mol" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].temperature" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].pressure" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[CH4]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[H2O]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[CO2]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].flow_mass" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].temperature" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Fe2O3]" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Fe3O4]" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Al2O3]" % x
        for x in space if x != space.last()
    )
    return setpoint_states


def get_data_from_steady_model(m, time):
    assert len(time) == 1
    t0 = next(iter(time))
    scalar_vars, dae_vars = flatten_dae_components(m, time, pyo.Var)
    scalar_data = {
        str(pyo.ComponentUID(var)): var.value for var in scalar_vars
    }
    dae_data = {
        str(pyo.ComponentUID(var.referent)): var[t0].value for var in dae_vars
    }
    return scalar_data, dae_data


def initialize_dynamic(m, dae_vars):
    time = m.fs.time
    t0 = time.first()
    # Initialize to initial conditions
    # flattened_vars comes from the hack above...
    copy_values_from_time(dae_vars, time, t0, include_fixed=False)

    # Initialize derivatives to zero
    diff_deriv_disc_list = list(
        generate_diff_deriv_disc_components_along_set(m, time)
    )
    derivs = [var for _, var, _ in diff_deriv_disc_list]
    for var in derivs:
        for t in time:
            if not var[t].fixed:
                var[t].set_value(0.0)


def get_model_for_dynamic_optimization(
        parameter_perturbation=None,
        model_params=None,
        sample_points=None,
        initial_conditions=None,
        setpoint=None,
        ic_scalar_data=None,
        ic_dae_data=None,
        setpoint_data=None,
        objective_weights=None,
        objective_states=None,

        flatten_out=None,
        ):
    if model_params is None:
        model_params = {}
    # This is only necessary because horizon is used elsewhere in this
    # script.
    # TODO: horizon should not be used elsewhere in this script
    horizon = model_params.pop("horizon", 300.0)
    with TIMER.context("make dynamic"):
        flattened_vars = [None, None]
        m, var_cat, con_cat = make_square_dynamic_model(
            # TODO: Bounds?
            horizon=horizon,
            **model_params,

            # HACK: This list will be modified to hold
            # scalar_vars and dae_vars so I can re-use them below
            flatten_out=flattened_vars,
        )
    time = m.fs.time
    t0 = time.first()
    if sample_points is None:
        sample_points = [
            time.at(i) for i in range(1, len(time) + 1) if not (i - 1) % 4
        ]

    # Set initial conditions
    if ic_scalar_data:
        # TODO: This should probably go in initialization?
        # Why is this necessary? What scalar variables are we actually
        # setting here? Probably all of them...
        set_values(m, ic_scalar_data)
    if ic_dae_data:
        # initial_conditions don't have to be only the variables that
        # are fixed... we don't touch the structure of the model, just
        # set values.
        set_values_at_time(m, t0, ic_dae_data)

    # NOTE: flattened_vars comes from hack above
    dynamic_scalar_vars, dynamic_dae_vars = flattened_vars
    if flatten_out is not None:
        flatten_out[0] = dynamic_scalar_vars
        flatten_out[1] = dynamic_dae_vars
    # We rely on this call to set time-varying "parameter" vars
    copy_values_from_time(dynamic_dae_vars, time, t0, include_fixed=True)

    if setpoint_data:
        # FIXME: This branch does not work as expected
        setpoint = setpoint_data
    else:
        raise RuntimeError()

    if objective_states is None:
        objective_states = [
            str(pyo.ComponentUID(var.referent))
            for var in var_cat[VC.DIFFERENTIAL]
        ]
    # Add setpoint to dynamic model
    tracking_cost = get_tracking_cost_expressions(
        objective_states, time, setpoint, weights=objective_weights
    )
    m.tracking_cost = tracking_cost
    m.tracking_objective = pyo.Objective(
        expr=sum(m.tracking_cost[t] for t in time if t != time.first())
    )

    # Add piecewise constant constraints and unfix inputs
    inputs = [
        "fs.MB.gas_phase.properties[*,0.0].flow_mol",
        "fs.MB.solid_phase.properties[*,1.0].flow_mass",
    ]
    piecewise_constant_constraint = get_piecewise_constant_constraints(
        inputs, time, sample_points
    )
    m.piecewise_constant_constraint = piecewise_constant_constraint
    for name in inputs:
        var = m.find_component(name)
        for t in time:
            if t != t0:
                var[t].unfix()

    # Set disturbance values in dynamic model
    if parameter_perturbation is None:
        parameter_perturbation = {}
    load_inputs_into_model(m, time, parameter_perturbation)

    return m


def get_nmpc_plant_model(
        parameter_perturbation=None,
        model_params=None,
        setpoint=None,
        ic_scalar_data=None,
        ic_dae_data=None,
        setpoint_data=None,
        objective_weights=None,
        objective_states=None,

        flatten_out=None,
        ):
    if model_params is None:
        model_params = {}
    # This is only necessary because horizon is used elsewhere in this
    # script.
    # TODO: horizon should not be used elsewhere in this script
    horizon = model_params.pop("horizon", 300.0)
    with TIMER.context("make dynamic"):
        flattened_vars = [None, None]
        m, var_cat, con_cat = make_square_dynamic_model(
            # TODO: Bounds?
            horizon=horizon,
            **model_params,

            # HACK: This list will be modified to hold
            # scalar_vars and dae_vars so I can re-use them below
            flatten_out=flattened_vars,
        )
    time = m.fs.time
    t0 = time.first()

    # Set initial conditions
    if ic_scalar_data:
        # TODO: This should probably go in initialization?
        # Why is this necessary? What scalar variables are we actually
        # setting here? Probably all of them...
        set_values(m, ic_scalar_data)
    if ic_dae_data:
        # initial_conditions don't have to be only the variables that
        # are fixed... we don't touch the structure of the model, just
        # set values.
        set_values_at_time(m, t0, ic_dae_data)

    # NOTE: flattened_vars comes from hack above
    dynamic_scalar_vars, dynamic_dae_vars = flattened_vars
    if flatten_out is not None:
        flatten_out[0] = dynamic_scalar_vars
        flatten_out[1] = dynamic_dae_vars
    # We rely on this call to set time-varying "parameter" vars
    copy_values_from_time(dynamic_dae_vars, time, t0, include_fixed=True)

    if setpoint_data:
        # FIXME: This branch does not work as expected
        setpoint = setpoint_data
    else:
        raise RuntimeError()

    if objective_states is None:
        objective_states = [
            str(pyo.ComponentUID(var.referent))
            for var in var_cat[VC.DIFFERENTIAL]
        ]
    # Add setpoint to dynamic model
    tracking_cost = get_tracking_cost_expressions(
        objective_states, time, setpoint, weights=objective_weights
    )
    m.tracking_cost = tracking_cost

    # Set disturbance values in dynamic model
    if parameter_perturbation is None:
        parameter_perturbation = {}
    load_inputs_into_model(m, time, parameter_perturbation, time_tol=None)

    return m


def get_model_for_simulation(
        horizon=None,
        ntfe=None,
        **kwds,
        ):
    # NOTE: This gets the model that I use for stability analysis
    """ 
    Constructs a dynamic model with default inputs and initializes to
    steady state.

    Parameters
    ----------
    horizon:
        length of time horizon to simulate
    ntfe:
        number of time finite elements in the horizon
    t1:
        time at which input perturbation will be applied
    input_value:
        perturbed value of gas inlet flow rate

    """
    if horizon is not None:
        kwds["horizon"] = horizon
    if ntfe is not None:
        kwds["ntfe"] = ntfe
    m, var_cat, con_cat = make_square_dynamic_model(**dict(kwds))
    add_constraints_for_missing_variables(m)
    time = m.fs.time
    t0 = time.first()

    m_steady = make_square_model(steady=True, **dict(kwds))
    add_constraints_for_missing_variables(m_steady)
    set_default_design_vars(m_steady)
    set_default_inlet_conditions(m_steady)
    initialize_steady(m_steady)
    solver = pyo.SolverFactory("ipopt")
    m_steady._obj = pyo.Objective(expr=0.0)
    res = solver.solve(m_steady, tee=True)
    scalar_vars, dae_vars = initialize_dynamic_from_steady(m, m_steady)

    return m, var_cat, con_cat


if __name__ == "__main__":
    m, var_cat, con_cat = get_model_for_simulation(300.0, 20)
    import pdb; pdb.set_trace()
