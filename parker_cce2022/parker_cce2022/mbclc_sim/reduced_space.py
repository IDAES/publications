import itertools
import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_components_along_sets
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
    TemporarySubsystemManager,
    create_subsystem_block,
)
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.util import (
    solve_strongly_connected_components,
    generate_strongly_connected_components,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)

from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    large_residuals_set,
)

from workspace.mbclc import (
    add_constraints_for_missing_variables,
    make_square_model,
    get_block_triangularization,
    get_minimal_subsystem_for_variables,
    set_default_design_vars,
    set_default_inlet_conditions,
    set_optimal_design_vars,
    set_optimal_inlet_conditions,
    set_values_to_inlets,
    set_gas_values_to_inlets,
    set_solid_values_to_inlets,
    set_gas_temperature_to_solid_inlet,
    initialize_steady,
    initialize_steady_without_solid_temperature,
)

import time as time_module

from workspace.common.dae_utils import (
    generate_discretization_components_along_set,
)
from workspace.common.expr_util import (
    generate_trivial_constraints,
)
from workspace.common.categorize import (
    categorize_dae_variables_and_constraints,
    VariableCategory,
    ConstraintCategory,
)
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
)

# NOTE: Relative import
from parker_cce2022.mbclc_sim.config import (
    get_nominal_values,
    get_ipopt,
    get_cyipopt,
    get_ipopt_options,
    get_temperature_list,
    get_nxfe_list,
    MockResults,
)

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from pyomo.common.timing import HierarchicalTimer
TIMER = HierarchicalTimer()

import pyomo.repn.plugins.ampl.ampl_ as nl_module
import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_module
import pyomo.contrib.pynumero.interfaces.external_pyomo_model as epm_module
import pyomo.contrib.pynumero.interfaces.nlp_projections as proj_module

nl_module.TIMER = TIMER
cyipopt_module.TIMER = TIMER
epm_module.TIMER = TIMER
proj_module.TIMER = TIMER


def _consolidate_category_dicts(cat1, cat2, union, intersect):
    """
    I have not set up categorization to accept multiple sets, because
    it is not clear what this would mean (look for diff wrt either set
    or both?). This function combines two category dicts by taking
    the union of their differential/derivative components and the
    intersection of their algebraic components.
    """
    category_dict = dict()
    accounted_for = ComponentSet()
    for categ in union:
        # combine differential/derivative/discretization components
        category_dict[categ] = cat1[categ] + cat2[categ]
        accounted_for.update(category_dict[categ])

    for categ in intersect:
        alg_set1 = ComponentSet(cat1[categ])
        category_dict[categ] = [comp for comp in cat2[categ]
                if comp in alg_set1 and comp not in accounted_for]

    return category_dict


def initialize_and_solve(m):
    initialize_steady(m)
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=True)


def solve_diff_vars_at_inlets(m, var_cat, con_cat, gas_var_cat, solid_var_cat):
    space = m.fs.MB.gas_phase.length_domain
    x0 = space.first()
    xf = space.last()

    VC = VariableCategory
    CC = ConstraintCategory
        
    # Solve for flow vars (diff vars) at the inlets.
    x = x0
    constraints = []
    variables = []
    for con in con_cat[CC.ALGEBRAIC]:
        try:
            constraints.append(con[x])
        except KeyError:
            pass
    for var in var_cat[VC.ALGEBRAIC]+gas_var_cat[VC.DIFFERENTIAL]:
        if not var[x].fixed:
            variables.append(var[x])
    system = (variables, constraints)
    target_vars = [v[x] for v in gas_var_cat[VC.DIFFERENTIAL]
            if not v[x].fixed]
    subsystems = get_minimal_subsystem_for_variables(system, target_vars)

    solver = pyo.SolverFactory("ipopt")
    for cons, vars in subsystems:
        subsystem = create_subsystem_block(cons, vars)
        to_fix = list(subsystem.input_vars.values())
        with TemporarySubsystemManager(to_fix=to_fix):
            solver.solve(subsystem, tee=True)

    x = xf
    constraints = []
    variables = []
    for con in con_cat[CC.ALGEBRAIC]:
        try:
            constraints.append(con[x])
        except KeyError:
            pass
    # Since I am parsing expressions to get the relevant variables, I can
    # encounter variables that are not matched with the space-indexed
    # algebraic variables. The corresponding equations must be added manually.
    # I think the right way to get around this is to have the same variables
    # and constraints exist at every point in space, so I don't have to
    # parse these expressions.
    constraints.append(m.fs.MB.bed_area_eqn)

    to_fix = [var[x] for var in gas_var_cat[VC.DIFFERENTIAL]]
    with TemporarySubsystemManager(to_fix=to_fix):
        variables = []
        added = ComponentSet()
        for con in constraints:
            for var in identify_variables(con.body, include_fixed=False):
                if var not in added:
                    added.add(var)
                    variables.append(var)

    system = (variables, constraints)
    target_vars = [v[x] for v in solid_var_cat[VC.DIFFERENTIAL]
            if not v[x].fixed]
    subsystems = get_minimal_subsystem_for_variables(system, target_vars)

    solver = pyo.SolverFactory("ipopt")
    for cons, vars in subsystems:
        subsystem = create_subsystem_block(cons, vars)
        to_fix = list(subsystem.input_vars.values())
        with TemporarySubsystemManager(to_fix=to_fix):
            solver.solve(subsystem, tee=True)


def flatten_and_categorize(m, sets, indices):
    dae_vars = []
    dae_cons = []

    for idx_sets, vars in zip(
            *flatten_components_along_sets(m, sets, pyo.Var, indices)
            ):
        if len(idx_sets) == 1 and idx_sets[0] in sets:
            dae_vars.extend(vars)

    for idx_sets, cons in zip(
            *flatten_components_along_sets(m, sets, pyo.Constraint, indices)
            ):
        if len(idx_sets) == 1 and idx_sets[0] in sets:
            dae_cons.extend(cons)

    categories = ComponentMap()
    for s in sets:
        idx = indices[s]
        categories[s] = categorize_dae_variables_and_constraints(
                m,
                dae_vars,
                dae_cons,
                s,
                index=idx,
                )

    return categories


def add_category_blocks(m, cat_dict, cat_attrs):
    comp_name = {pyo.Var: "var", pyo.Constraint: "con"}
    for categ, name in cat_attrs.items():
        components = cat_dict[categ]
        ctype = components[0].ctype
        n = len(components)
        # TODO: Should I add a reference instead of the actual component?
        rule = lambda b, i: setattr(b, comp_name[ctype], components[i])
        m.add_component(name, pyo.Block(range(n), rule=rule))


def initialize_differential_variables(
        m,
        gas_var_cat,
        solid_var_cat,
        var_cat,
        con_cat,
        ):
    VC = VariableCategory
    gas_length = m.fs.MB.gas_phase.length_domain
    x0 = gas_length.first()
    xf = gas_length.last()
    # CUSTOM INITIALIZATION:
    # Note that this overrides the inlet temperature for our
    # diff-var initialization
    set_gas_temperature_to_solid_inlet(m, include_fixed=True)

    # Solve for flow variables from state variables
    solve_diff_vars_at_inlets(m, var_cat, con_cat, gas_var_cat, solid_var_cat)

    for var in gas_var_cat[VC.DIFFERENTIAL]:
        for x in gas_length:
            var[x].set_value(var[x0].value)
    for var in solid_var_cat[VC.DIFFERENTIAL]:
        for x in gas_length:
            var[x].set_value(var[xf].value)

    # Reset the values of inlet variables (inlet gas temperature)
    set_default_inlet_conditions(m)
    solve_diff_vars_at_inlets(m, var_cat, con_cat, gas_var_cat, solid_var_cat)


def fix_differential_variables_at_inlets(space, gas_var_cat, solid_var_cat):
    VC = VariableCategory
    x0 = space.first()
    xf = space.last()
    for var in gas_var_cat[VC.DIFFERENTIAL]:
        var[x0].fix()
    for var in gas_var_cat[VC.ALGEBRAIC]:
        var[x0].unfix()
    for var in solid_var_cat[VC.DIFFERENTIAL]:
        var[xf].fix()
    for var in solid_var_cat[VC.ALGEBRAIC]:
        var[xf].unfix()


def solve_reduced_space(
        T=298.15,
        nxfe=10,
        flow=128.2,
        solid_temp=1183.15,
        use_cyipopt=True,
        ):
    """
    Initializes and solves the CLC model in the reduced space, with
    the scaling and options necessary to do so.
    """
    m = make_square_model(steady=True, nxfe=nxfe)
    add_constraints_for_missing_variables(m)
    time = m.fs.time
    t0 = time.first()
    gas_length = m.fs.MB.gas_phase.length_domain
    solid_length = m.fs.MB.solid_phase.length_domain
    mb_length = m.fs.MB.length_domain

    design_vars = set_default_design_vars(m)
    inlet_vars = set_default_inlet_conditions(m)
    m.fs.MB.gas_inlet.temperature[:].set_value(T)
    m.fs.MB.gas_inlet.flow_mol[:].set_value(flow)
    m.fs.MB.solid_inlet.temperature[:].set_value(solid_temp)
    set_gas_values_to_inlets(m)
    set_solid_values_to_inlets(m)

    x0 = gas_length.first()
    xf = gas_length.last()

    x1 = gas_length[2]

    # Set-up for flattening and categorization
    sets = ComponentSet((gas_length, solid_length))
    indices = ComponentMap((s, x1) for s in sets)

    VC = VariableCategory
    CC = ConstraintCategory

    # Categorize wrt gas and solid length domains
    categories = flatten_and_categorize(m, sets, indices)
    gas_var_cat, gas_con_cat = categories[gas_length]
    solid_var_cat, solid_con_cat = categories[solid_length]

    # Combine category dicts for two length sets
    var_cat = _consolidate_category_dicts(
            gas_var_cat, solid_var_cat,
            [VC.DIFFERENTIAL, VC.DERIVATIVE], [VC.ALGEBRAIC],
            )
    con_cat = _consolidate_category_dicts(
            gas_con_cat, solid_con_cat,
            [CC.DIFFERENTIAL, CC.DISCRETIZATION], [CC.ALGEBRAIC],
            )

    initialize_steady_without_solid_temperature(m)

    nominal_values = get_nominal_values()
    scaling_factors = [(name, 1/val) for name, val in nominal_values]

    n_diff = len(var_cat[VC.DIFFERENTIAL])
    n_alg = len(var_cat[VC.ALGEBRAIC])

    assert len(var_cat[VC.DERIVATIVE]) == n_diff
    assert len(con_cat[CC.DIFFERENTIAL]) == n_diff
    assert len(con_cat[CC.DISCRETIZATION]) == n_diff
    assert len(con_cat[CC.ALGEBRAIC]) == n_alg

    # Add constraints at boundaries that have been omitted by IDAES
    m.fs.MB.gas_phase.properties[t0, x0].sum_component_eqn = pyo.Constraint(
            expr=sum(m.fs.MB.gas_inlet.mole_frac_comp[t0, :]) - 1.0 == 0
            )
    m.fs.MB.solid_phase.properties[t0, xf].sum_component_eqn = pyo.Constraint(
            expr=sum(m.fs.MB.solid_inlet.mass_frac_comp[t0, :]) - 1.0 == 0
            )
    fix_differential_variables_at_inlets(gas_length, gas_var_cat, solid_var_cat)

    reduced_space = pyo.Block(concrete=True)

    # Add scaling factors
    diff_components = (
            var_cat[VC.DIFFERENTIAL],
            var_cat[VC.DERIVATIVE],
            con_cat[CC.DISCRETIZATION],
            con_cat[CC.DIFFERENTIAL],
            )
    reduced_space.scaling_factor = pyo.Suffix()
    scaling_factors = dict(scaling_factors)
    for state, deriv, disc, diff in zip(*diff_components):
        name = str(pyo.ComponentUID(state.referent))
        con_name = str(pyo.ComponentUID(disc.referent))
        print(name, con_name)
        if name in scaling_factors:
            sf = scaling_factors[name]
            for data in disc.values():
                reduced_space.scaling_factor[data] = sf
            for data in state.values():
                reduced_space.scaling_factor[data] = sf
            for data in deriv.values():
                reduced_space.scaling_factor[data] = sf
            for data in diff.values():
                reduced_space.scaling_factor[data] = sf

    # Add appropriate variables to reduced space model, on indexed blocks
    cat_dict = {}
    cat_dict.update(var_cat)
    cat_dict.update(con_cat)
    cat_attrs = {
            VC.DIFFERENTIAL: "diff",
            CC.DISCRETIZATION: "disc",
            VC.DERIVATIVE: "deriv",
            }
    add_category_blocks(reduced_space, cat_dict, cat_attrs)

    reduced_space.external_block = ExternalGreyBoxBlock(gas_length)
    ex_block = reduced_space.external_block

    sp = m.fs.MB.solid_phase
    rs = reduced_space

    # This loop constructs the external models
    for x in gas_length:
        if x == gas_length.first():
            input_vars = (
                [v[x] for v in var_cat[VC.DIFFERENTIAL] if not v[x].fixed] +
                [v[x] for v in solid_var_cat[VC.DERIVATIVE]]
                )
            external_vars = ([v[x] for v in var_cat[VC.ALGEBRAIC]]
                    + [v[x] for v in gas_var_cat[VC.DERIVATIVE]])
            residual_cons = [c[x] for c in solid_con_cat[CC.DIFFERENTIAL]]
            external_cons = ([c[x] for c in con_cat[CC.ALGEBRAIC]]
                    + [v[x] for v in gas_con_cat[CC.DIFFERENTIAL]])

        elif x == gas_length.last():
            input_vars = (
                [v[x] for v in var_cat[VC.DIFFERENTIAL] if not v[x].fixed] +
                [v[x] for v in gas_var_cat[VC.DERIVATIVE]]
                )
            # Particle porosity should be fixed at inlet
            external_vars = ([v[x] for v in var_cat[VC.ALGEBRAIC]
                    if v[x] is not sp.properties[t0, x].particle_porosity]
                    + [v[x] for v in solid_var_cat[VC.DERIVATIVE]])
            residual_cons = [c[x] for c in gas_con_cat[CC.DIFFERENTIAL]]
            external_cons = ([c[x] for c in con_cat[CC.ALGEBRAIC]]
                    + [c[x] for c in solid_con_cat[CC.DIFFERENTIAL]])

            # TODO: Need to walk the algebraic variable expressions to get
            # the variables if they solve for a non-space-indexed variable
            m.fs.MB.solid_inlet.particle_porosity.fix()
            external_vars.append(m.fs.MB.velocity_superficial_solid[t0])

        else:
            input_vars = list(rs.diff[:].var[x]) + list(rs.deriv[:].var[x])
            residual_cons = [con[x] for con in con_cat[CC.DIFFERENTIAL]]
            external_vars = [var[x] for var in var_cat[VC.ALGEBRAIC]]
            external_cons = [con[x] for con in con_cat[CC.ALGEBRAIC]]

        ex_model = ExternalPyomoModel(
                input_vars, external_vars, residual_cons, external_cons,
                use_cyipopt=use_cyipopt,
                )
        scaling_factors = [
                reduced_space.scaling_factor[c] for c in residual_cons
                ]
        ex_model.set_equality_constraint_scaling_factors(scaling_factors)
        ex_block[x].set_external_model(ex_model, inputs=input_vars)

    reduced_space._obj = pyo.Objective(expr=0)

    #for x in gas_length:
    #    print("External model at x = %s" % x)
    #    ex_model = ex_block[x].get_external_model()
    #    vars = ex_model.external_vars
    #    cons = ex_model.external_cons

    #    constraints = (list(reduced_space.disc[:].con[:])
    #            + ex_model.residual_cons)
    #    residual_block = create_subsystem_block(constraints)
    #    for con in large_residuals_set(residual_block, tol=1e-7):
    #        resid = pyo.value(con.body-con.lower)
    #        print(con.name, resid)

    # Solve reduced space model
    options = get_ipopt_options()
    solver = get_cyipopt(options)
    try:
        t_start = time_module.time()
        TIMER.start("solve")
        res = solver.solve(rs, tee=True)
        TIMER.stop("solve")
        t_end = time_module.time()
        wallclock_time = t_end - t_start
        res = MockResults(
            termination_condition=res.solver.termination_condition,
            wallclock_time=wallclock_time,
        )
    except (RuntimeError, ValueError, AssertionError):
        t_end = time_module.time()
        wallclock_time = t_end - t_start
        res = MockResults(
            termination_condition="implicit_function_error",
            wallclock_time=wallclock_time,
        )

    return res, m


def run_sweep():
    #T_list = [1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0,
    #        1900.0, 2000.0, 2100.0, 2200.0]
    T_list = [1000.0]
    solid_temp_list = [1200.0]
    #flow_list = [120.0]
    #nxfe_list = [10]
    #T_list = get_temperature_list()
    #nxfe_list = get_nxfe_list()
    res_list = []
    #import warnings
    #warnings.simplefilter("error")
    for T, flow in itertools.product(T_list, flow_list):
        res, m = solve_reduced_space(T=T, flow=flow)
        res_list.append(res)
        print(
            T,
            res.solver.termination_condition,
            res.solver.wallclock_time,
        )
    for res, (T, flow) in zip(res_list, itertools.product(T_list, flow_list)):
        print(
            T,
            flow,
            res.solver.termination_condition,
            res.solver.wallclock_time,
        )


def main():
    gas_temp = 1000.0
    solid_temp = 1200.0
    flow = 128.2

    # We only ever want to not use CyIpopt if we want to display problem
    # statistics in the IPOPT header. For some reason, when we call CyIpopt
    # recursively, this header doesn't get sent to stdout.
    use_cyipopt = True
    res, m = solve_reduced_space(
        T=gas_temp,
        solid_temp=solid_temp,
        #flow=flow,
        use_cyipopt=use_cyipopt,
    )
    print(TIMER)


if __name__ == "__main__":
    main()
