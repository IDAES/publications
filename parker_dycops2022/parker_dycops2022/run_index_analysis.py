import itertools
import pyomo.environ as pyo

from pyomo.common.collections import ComponentSet
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    _generate_variables_in_constraints,
)

from parker_dycops2022.mbclc.model import (
    make_square_dynamic_model,
    make_square_model,
)
from parker_dycops2022.mbclc.initialize import (
    set_default_design_vars,
    set_default_inlet_conditions,
    initialize_steady,
    initialize_dynamic_from_steady,
)
from parker_dycops2022.mbclc.plot import (
    plot_outlet_states_over_time,
)
from parker_dycops2022.common.initialize import initialize_by_time_element
from idaes.apps.caprese.categorize import (
    VariableCategory as VC,
    ConstraintCategory as CC,
)
from parker_dycops2022.common.dynamic_data import (
    load_inputs_into_model,
)
from parker_dycops2022.model import (
    get_model_for_simulation,
)

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt


def try_factorization(matrix):
    try:
        sps.linalg.splu(matrix.tocsc())
        print("Factorization successful")
    except RuntimeError:
        print("Factorization unsuccessful")


def plot_spy(matrix, **kwds):
    fig = plt.figure()    
    ax = fig.add_subplot()
    ax.spy(matrix, **kwds)
    return fig, ax


def get_condition_number(model):
    if isinstance(model, pyo.Block):
        if len(list(
                model.component_data_objects(pyo.Objective, active=True))) == 0:
            model._obj = pyo.Objective(expr=0.0)
        nlp = PyomoNLP(model)
        jacobian = nlp.evaluate_jacobian()
        model.del_component(model._obj)
    elif isinstance(model, sps.coo_matrix):
        jacobian = model
    else:
        raise ValueError()
    if jacobian.shape == (1, 1):
        return 1.0
    else:
        # HACK: Use dense linear algebra for minimum singular value
        #jjt = jacobian.dot(jacobian.transpose())
        #jjt_dense = jjt.toarray()
        #sv, _ = np.linalg.eig(jjt_dense)
        _, smin, _ = sps.linalg.svds(jacobian, k=24, which="SM", solver="lobpcg")
        _, smax, _ = sps.linalg.svds(jacobian, k=1, which="LM", solver="lobpcg")
        cond = smax[0]/smin[0]
        return cond


def get_polynomial_degree_wrt(constraints, variables=None):
    # TODO: Move this function elsewhere
    from pyomo.core.expr.visitor import polynomial_degree
    from pyomo.util.subsystems import TemporarySubsystemManager
    vars_in_cons = list(_generate_variables_in_constraints(constraints))
    if variables is None:
        variables = vars_in_cons
    var_set = ComponentSet(variables)
    other_vars = [v for v in vars_in_cons if v not in var_set]
    with TemporarySubsystemManager(to_fix=other_vars):
        con_poly_degree = [polynomial_degree(con.expr) for con in constraints]
    if any(d is None for d in con_poly_degree):
        # General nonlinear
        return None
    else:
        return max(con_poly_degree)


def main():
    horizon = 600.0
    ntfe = 40
    #horizon = 30.0
    #ntfe = 2
    t1 = 0.0
    ch4_cuid = "fs.MB.gas_inlet.mole_frac_comp[*,CH4]"
    co2_cuid = "fs.MB.gas_inlet.mole_frac_comp[*,CO2]"
    h2o_cuid = "fs.MB.gas_inlet.mole_frac_comp[*,H2O]"
    input_dict = {
        ch4_cuid: {(t1, horizon): 0.5},
        co2_cuid: {(t1, horizon): 0.5},
        h2o_cuid: {(t1, horizon): 0.0},
    }

    m, var_cat, con_cat = get_model_for_simulation(horizon, ntfe)
    time = m.fs.time
    load_inputs_into_model(m, time, input_dict)

    solver = pyo.SolverFactory("ipopt")
    solve_kwds = {"tee": True}
    res_list = initialize_by_time_element(
        m, time, solver=solver, solve_kwds=solve_kwds
    )
    res = solver.solve(m, **solve_kwds)
    msg = res if type(res) is str else res.solver.termination_condition
    print(horizon, ntfe, msg)

    m._obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    igraph = IncidenceGraphInterface()

    # TODO: I should be able to do categorization in the pre-time-discretized
    # model. This is somewhat nicer as the time points are all independent
    # in that case.
    solid_enth_conds = []
    gas_enth_conds = []
    solid_dens_conds = []
    gas_dens_conds = []
    for t in time:
        var_set = ComponentSet(var[t] for var in var_cat[VC.ALGEBRAIC])
        constraints = [con[t] for con in con_cat[CC.ALGEBRAIC] if t in con]
        variables = [
            var for var in _generate_variables_in_constraints(constraints)
            if var in var_set
        ]

        assert len(variables) == len(constraints)

        alg_jac = nlp.extract_submatrix_jacobian(variables, constraints)
        N, M = alg_jac.shape
        assert N == M
        matching = igraph.maximum_matching(variables, constraints)
        assert len(matching) == N
        try_factorization(alg_jac)

        # Condition number of the entire algebraic Jacobian seems
        # inconsistent, so I don't calculate it.
        #cond = np.linalg.cond(alg_jac.toarray())
        #cond = get_condition_number(alg_jac)

        var_blocks, con_blocks = igraph.get_diagonal_blocks(
            variables, constraints
        )
        block_matrices = [
            nlp.extract_submatrix_jacobian(vars, cons)
            for vars, cons in zip(var_blocks, con_blocks)
        ]
        gas_enth_blocks = [
            i for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks))
            if any(
                "gas_phase" in var.name and "temperature" in var.name
                for var in vars
            )
        ]
        solid_enth_blocks = [
            i for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks))
            if any(
                "solid_phase" in var.name and "temperature" in var.name
                for var in vars
            )
        ]
        gas_dens_blocks = [
            i for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks))
            if any(
                "gas_phase" in con.name and "sum_component_eqn" in con.name
                for con in cons
            )
        ]
        solid_dens_blocks = [
            i for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks))
            if any(
                "solid_phase" in con.name and "sum_component_eqn" in con.name
                for con in cons
            )
        ]
        gas_enth_cond = [
            np.linalg.cond(block_matrices[i].toarray())
            for i in gas_enth_blocks
        ]
        solid_enth_cond = [
            np.linalg.cond(block_matrices[i].toarray())
            for i in solid_enth_blocks
        ]
        gas_dens_cond = [
            np.linalg.cond(block_matrices[i].toarray())
            for i in gas_dens_blocks
        ]
        solid_dens_cond = [
            np.linalg.cond(block_matrices[i].toarray())
            for i in solid_dens_blocks
        ]
        max_gas_enth_cond = max(gas_enth_cond)
        max_solid_enth_cond = max(solid_enth_cond)
        max_gas_dens_cond = max(gas_dens_cond)
        max_solid_dens_cond = max(solid_dens_cond)
        gas_enth_conds.append(max_gas_enth_cond)
        solid_enth_conds.append(max_solid_enth_cond)
        gas_dens_conds.append(max_gas_dens_cond)
        solid_dens_conds.append(max_solid_dens_cond)

    # Plot condition numbers over time
    plt.rcParams.update({"font.size": 16})
    fig = plt.figure()
    ax = fig.add_subplot()
    t_list = list(time)
    ax.plot(t_list, gas_enth_conds, label="Gas enth.",
            linewidth=3, linestyle="solid")
    ax.plot(t_list, solid_enth_conds, label="Solid enth.",
            linewidth=3, linestyle="dotted")
    ax.plot(t_list, gas_dens_conds, label="Gas dens.",
            linewidth=3, linestyle="dashed")
    ax.plot(t_list, solid_dens_conds, label="Solid dens.",
            linewidth=3, linestyle="dashdot")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0, top=1e7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Condition number")
    fig.legend(loc="center right", bbox_to_anchor=(1.0, 0.65))
    fig.tight_layout()
    fig.show()
    fig.savefig("condition_over_time.png", transparent=True)

    # Generate some structural results with the incidence matrix at a single
    # point in time.
    t = time.at(2)
    var_set = ComponentSet(var[t] for var in var_cat[VC.ALGEBRAIC])
    constraints = [con[t] for con in con_cat[CC.ALGEBRAIC] if t in con]
    variables = [
        var for var in _generate_variables_in_constraints(constraints)
        if var in var_set
    ]
    alg_jac = nlp.extract_submatrix_jacobian(variables, constraints)

    var_blocks, con_blocks = igraph.get_diagonal_blocks(
        variables, constraints
    )
    dim = len(constraints)
    n_blocks = len(var_blocks)
    print("Number of variables/constraints: %s" % dim)
    print("Number of diagonal blocks: %s" % n_blocks)
    block_polynomial_degrees = [
        get_polynomial_degree_wrt(cons, vars)
        for cons, vars in zip(con_blocks, var_blocks)
    ]
    nonlinear_blocks = [
        i for i, d in enumerate(block_polynomial_degrees)
        if d is None or d > 1
    ]
    print("Number of nonlinear blocks: %s" % len(nonlinear_blocks))

    print("\nNonlinear blocks:")
    for i in nonlinear_blocks:
        vars = var_blocks[i]
        cons = con_blocks[i]
        dim = len(vars)
        print("  Block %s, dim = %s" % (i, dim))
        print("    Variables:")
        for var in vars:
            print("      %s" % var.name)
        print("    Constraints:")
        for con in cons:
            print("      %s" % con.name)

    ordered_variables = [var for vars in var_blocks for var in vars]
    ordered_constraints = [con for cons in con_blocks for con in cons]
    ordered_jacobian = nlp.extract_submatrix_jacobian(
        ordered_variables, ordered_constraints
    )
    plt.rcParams.update({"font.size": 18})
    fig, ax = plot_spy(
        ordered_jacobian,
        markersize=3,
    )
    ax.xaxis.set_tick_params(bottom=False)
    ax.xaxis.set_label_position("top")
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([0, 200, 400, 600])
    ax.set_xlabel("Column (variable) coordinates")
    ax.set_ylabel("Row (equation) coordinates")
    fig.tight_layout()
    fig.savefig("block_triangular_alg_jac.png", transparent=True)
    fig.show()

    #plot_outlet_states_over_time(m, show=True)


if __name__ == "__main__":
    main()
