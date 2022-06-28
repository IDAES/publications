import time as time_module
import itertools
import pyomo.common.unittest as unittest

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

nl_module.TIMER = TIMER


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
            solver.solve(subsystem)

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
            solver.solve(subsystem)


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
    diff_vars = [v[x0] for v in gas_var_cat[VC.DIFFERENTIAL]]
    with TemporarySubsystemManager(to_reset=diff_vars):
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
    #set_default_inlet_conditions(m)
    # This is where gas temperature is getting reset
    # It seems gas temperature is not actually getting reset; just
    # gas enthalpy flow is getting reset.
    # gas temperature is reset by the temporary subsystem manager
    #solve_diff_vars_at_inlets(m, var_cat, con_cat, gas_var_cat, solid_var_cat)


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


def solve_full_space(T=298.25, nxfe=10, flow=128.2, solid_temp=1183.15):
    """
    T is the temperature of the gas inlet
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

    # Initialize by deactivating discretization equations and solving
    # SCC decomposition.
    initialize_steady_without_solid_temperature(m)

    # Assign scaling factors
    nominal_values = get_nominal_values()
    scaling_factors = [(name, 1/val) for name, val in nominal_values]

    # Add scaling factors
    diff_components = (
            var_cat[VC.DIFFERENTIAL],
            var_cat[VC.DERIVATIVE],
            con_cat[CC.DISCRETIZATION],
            con_cat[CC.DIFFERENTIAL],
            )
    m.scaling_factor = pyo.Suffix()
    scaling_factors = dict(scaling_factors)
    for state, deriv, disc, diff in zip(*diff_components):
        name = str(pyo.ComponentUID(state.referent))
        con_name = str(pyo.ComponentUID(disc.referent))
        print(name, con_name)
        if name in scaling_factors:
            sf = scaling_factors[name]
            for data in disc.values():
                m.scaling_factor[data] = sf
            for data in state.values():
                m.scaling_factor[data] = sf
            for data in deriv.values():
                m.scaling_factor[data] = sf
            for data in diff.values():
                m.scaling_factor[data] = sf

    options = get_ipopt_options()
    ipopt = get_ipopt()
    options.pop("inf_pr_output")
    ipopt.set_options(options)
    t_start = time_module.time()
    try:
        TIMER.start("solve")
        res = ipopt.solve(m, tee=True)
        TIMER.stop("solve")
        t_end = time_module.time()
        res = MockResults(
            termination_condition=res.solver.termination_condition,
            time=t_end-t_start,
        )
    except (ValueError, RuntimeError):
        t_end = time_module.time()
        res = MockResults(
                termination_condition="solver_error",
                time=t_end-t_start,
                )
    return res, m


def run_sweep():
    #T_list = [1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
    #        1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0]
    #nxfe_list = [10]
    T_list = [1000.0]
    #flow_list = [128.2]
    flow_list = [120, 160]
    #T_list = get_temperature_list()
    #nxfe_list = get_nxfe_list()
    res_list = []
    for T, flow in itertools.product(T_list, flow_list):
        res, m = solve_full_space(T=T, flow=flow)
        res_list.append(res)
        print(
            T,
            res.solver.termination_condition,
            res.solver.time,
        )
    for res, (T, flow) in zip(res_list, itertools.product(T_list, flow_list)):
        print(
            T,
            flow,
            res.solver.termination_condition,
            res.solver.time,
        )


def main():
    """
    If run as a script, we just solve the full space problem for some
    nominal set of inputs.
    """
    gas_temp = 1000.0
    solid_temp = 1200.0
    gas_flow = 128.2
    res, m = solve_full_space(
        T=gas_temp,
        solid_temp=solid_temp,
    )
    print(TIMER)


if __name__ == "__main__":
    main()
