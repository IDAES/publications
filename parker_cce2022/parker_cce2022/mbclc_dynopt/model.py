import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components

from idaes.apps.caprese.categorize import (
    VariableCategory,
    ConstraintCategory,
)

from workspace.common.dynamic_data import (
    get_tracking_cost_expression,
)
from workspace.mbclc.model import (
    make_square_model,
    make_square_dynamic_model,
)
from workspace.mbclc.initialize import (
    initialize_steady,
    set_default_design_vars,
    set_default_inlet_conditions,
)


"""
This file provides functions for constructing a dynamic optimization
problem involving the MBCLC reactor model. If run as a script, it
constructs and tries to solve a nominal instance of the full-space
dynamic optimization problem.
"""


def get_state_variable_names(space):
    """
    These are a somewhat arbitrary set of time-indexed variables that are
    (more than) sufficient to solve for the rest of the variables.
    They happen to correspond to the property packages' state variables.

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


def get_piecewise_constant_constraints(inputs, time, sample_points):
    sample_point_set = set(sample_points)
    input_set = pyo.Set(initialize=range(len(inputs)))
    def piecewise_constant_rule(m, i, t):
        if t in sample_point_set:
            return pyo.Constraint.Skip
        else:
            var = inputs[i]
            t_next = time.next(t)
            return var[t_next] - var[t] == 0
    pwc_con = pyo.Constraint(
        input_set, time, rule=piecewise_constant_rule
    )
    return input_set, pwc_con

def get_dynamic_model(
        nxfe=10,
        nxcp=1,
        objective_states=None,
        setpoint=None,
        objective_weights=None,
        initial_data=None,
        scalar_data=None,
        sample_width=120.0,
        samples_per_horizon=15,
        ntfe_per_sample=2,
        ntcp=1,
        bounds=False,
        ):
    horizon = sample_width * samples_per_horizon
    ntfe = ntfe_per_sample * samples_per_horizon
    m, var_cat, con_cat = make_square_dynamic_model(
        nxfe=nxfe,
        nxcp=nxcp,
        horizon=horizon,
        ntfe=ntfe,
        ntcp=ntcp,
        bounds=bounds,
    )
    time = m.fs.time
    t0 = time.first()
    space = m.fs.MB.gas_phase.length_domain
    inputs = [
        pyo.Reference(m.fs.MB.gas_inlet.flow_mol[:]),
        pyo.Reference(m.fs.MB.solid_inlet.flow_mass[:]),
    ]
    if objective_states is None:
        if objective_weights is not None:
            raise RuntimeError("Can't specify weights without states")
        #objective_state_names = get_state_variable_names(space)
        #objective_states = [
        #    m.find_component(name) for name in objective_state_names
        #]
        objective_states = (
            var_cat[VariableCategory.DIFFERENTIAL]
            + inputs
        )
        objective_state_names = ([
                str(pyo.ComponentUID(str(pyo.ComponentUID(var.referent))))
                for var in var_cat[VariableCategory.DIFFERENTIAL]
            ] + [
                str(pyo.ComponentUID(str(pyo.ComponentUID(var.referent))))
                for var in inputs
            ]
        )

    # Set up dictionaries for nominal values and scaling factors.
    # These should probably be configurable by function arguments.
    nominal_values = {}
    for name in objective_state_names:
        if "material_holdup" in name and "Vap" in name:
            nominal_values[name] = 100.0
        elif "material_holdup" in name and "Sol" in name:
            nominal_values[name] = 1e4
        elif "energy_holdup" in name and "Vap" in name:
            nominal_values[name] = 1e4
        elif "energy_holdup" in name and "Sol" in name:
            nominal_values[name] = 1e7
        elif "flow_mol" in name:
            nominal_values[name] = 100.0
        elif "flow_mass" in name:
            nominal_values[name] = 100.0

    diff_var_names = [
        str(pyo.ComponentUID(str(pyo.ComponentUID(var.referent))))
        for var in var_cat[VariableCategory.DIFFERENTIAL]
    ]

    diff_var_scaling_factors = [
        1/nominal_values[name] for name in diff_var_names
    ]

    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    diff_vars = var_cat[VariableCategory.DIFFERENTIAL]
    deriv_vars = var_cat[VariableCategory.DERIVATIVE]
    disc_cons = con_cat[ConstraintCategory.DISCRETIZATION]
    for i, diff in enumerate(diff_vars):
        deriv = deriv_vars[i]
        disc = disc_cons[i]
        sf = diff_var_scaling_factors[i]
        for t in time:
            m.scaling_factor[diff[t]] = sf
            m.scaling_factor[deriv[t]] = sf
            if t in disc:
                m.scaling_factor[disc[t]] = sf

    if objective_weights is None:
        # Default objective weights are inverse squared nominal values
        # of the differential variables
        objective_weights = {
            name: 1/val**2 for name, val in nominal_values.items()
        }
    m.tracking_cost = get_tracking_cost_expression(
        objective_states, time, setpoint, objective_weights,
    )
    time_step = time.at(2) - time.at(1)
    m.tracking_objective = pyo.Objective(
        expr=sum(time_step*m.tracking_cost[t] for t in time if t != t0)
    )
    sample_points = [sample_width * i for i in range(samples_per_horizon + 1)]

    input_set, pwc_con = get_piecewise_constant_constraints(
        inputs, time, sample_points
    )
    m.input_set = input_set
    m.pwc_con = pwc_con
    for var in inputs:
        var[:].unfix()
        var[t0].fix()

    # Initialize scalar variables in model
    for name, val in scalar_data.items():
        var = m.find_component(name)
        var.set_value(val)

    # Initialize to initial conditions
    for name, val in initial_data.items():
        var = m.find_component(name)
        for t in time:
            var[t].set_value(val)

    return m, var_cat, con_cat


def get_steady_state_data(
        nxfe=10,
        nxcp=1,
        to_unfix=None,
        input_map=None,
        setpoint_list=None,
        objective_weights=None,
        tee=True,
        ):
    m = make_square_model(
        steady=True,
        nxfe=nxfe,
        nxcp=nxcp,
    )
    time = m.fs.time
    t0 = time.first()

    if to_unfix is None:
        to_unfix = []

    if input_map is None:
        input_map = dict()

    set_default_design_vars(m)
    set_default_inlet_conditions(m)

    initialize_steady(m)
    ipopt = pyo.SolverFactory("ipopt")
    ipopt.solve(m, tee=tee)

    # These are the degrees of freedom in our steady state optimization solve
    for name in to_unfix:
        var = m.find_component(name)
        var[:].unfix()

    for name, val in input_map.items():
        var = m.find_component(name)
        var[:].set_value(val)

    if setpoint_list is None:
        setpoint_list = []
    # NOTE: Potentially breaking order here
    setpoint_list = [
        # Need to make sure strings get passed through ComponentUID. A
        # better solution would be to just use ComponentUIDs as keys. I
        # haven't hit this before because I usually provide my own reference-
        # to-slices as objective_states, rather than get them from
        # find_component.
        (str(pyo.ComponentUID(name)), val) for name, val in setpoint_list
    ]
    objective_states = [m.find_component(name) for name, _ in setpoint_list]
    setpoint_dict = dict(setpoint_list)

    m.tracking_cost = get_tracking_cost_expression(
        # Using setpoint_dict is dangerous. If I use strings, I need
        # to make sure they go through CUID.
        objective_states, time, setpoint_dict, objective_weights
    )
    m.obj = pyo.Objective(expr=m.tracking_cost[t0])

    if pyo.value(m.obj.expr) != 0:
        ipopt.solve(m, tee=tee)

    scalar_vars, dae_vars = flatten_dae_components(m, time, pyo.Var)
    scalar_data = {
        str(pyo.ComponentUID(var)): var.value
        for var in scalar_vars
    }
    dae_data = {}
    for var in dae_vars:
        # Copy the logic from get_time_indexed_cuid here
        if var.is_reference() and var.parent_block() is None:
            var_id = var.referent
        name = str(pyo.ComponentUID(str(pyo.ComponentUID(var_id))))
        dae_data[name] = var[t0].value
    #dae_data = {
    #    # HACK: Make sure strings go through ComponentUID, otherwise
    #    # mixing indices "0" and "0.0" will lead to KeyErrors
    #    str(pyo.ComponentUID(str(pyo.ComponentUID(var.referent)))):
    #    #str(pyo.ComponentUID(var.referent)):
    #        var[t0].value
    #    for var in dae_vars
    #}
    return scalar_data, dae_data

if __name__ == "__main__":
    nxfe = 10
    samples_per_horizon = 10
    nxcp = 1
    ntfe_per_sample = 2
    ntcp = 1
    sample_width = 120.0
    differential_bounds = False
    input_bounds = False
    ipopt_options = None
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
    m, var_cat, con_cat = get_dynamic_model(
        nxfe=nxfe,
        nxcp=nxcp,
        ntfe_per_sample=ntfe_per_sample,
        ntcp=ntcp,
        sample_width=sample_width,
        samples_per_horizon=samples_per_horizon,
        initial_data=ic_dae_data,
        setpoint=sp_dae_data,
        scalar_data=ic_scalar_data,
    )
    time = m.fs.time
    if differential_bounds:
        # Set lower bounds on differential variables if desired
        for var in var_cat[VC.DIFFERENTIAL]:
            for t in time:
                var[t].setlb(0.0)
    sample_points = [i*sample_width for i in range(samples_per_horizon + 1)]
    sample_points = [
        time.at(time.find_nearest_index(t)) for t in sample_points
    ]
    if input_bounds:
        for t in sample_points:
            m.fs.MB.gas_inlet.flow_mol[t].setlb(0.0)
            m.fs.MB.gas_inlet.flow_mol[t].setub(200.0)

    ipopt = pyo.SolverFactory("ipopt")
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
    try:
        res = ipopt.solve(m, tee=True)
    except ValueError as err:
        # ValueError when restoration fails, I believe
        # TODO: return something useful here
        pass
