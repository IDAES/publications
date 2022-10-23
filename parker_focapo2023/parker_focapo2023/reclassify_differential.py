from time import time as get_current_time
from parker_focapo2023.model import (
    make_model,
    add_piecewise_constant_constraints,
    add_objective,
    ModelVersion,
)
from parker_focapo2023.kkt import (
    get_equality_constrained_kkt_matrix,
)
from parker_focapo2023.plot import (
    plot_incidence_matrix,
)

from workspace.common.categorize import (
    categorize_dae_variables_and_constraints,
    VariableCategory,
    ConstraintCategory,
)
from workspace.common.dae_utils import (
    DifferentialHelper,
    generate_diff_deriv_disc_components_along_set,
)

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.dae.flatten import flatten_dae_components
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    _generate_variables_in_constraints,
    get_structural_incidence_matrix,
)

import scipy.sparse as sps


def reclassify_differential():
    model_version = ModelVersion.IDAES_1_7
    m = make_model(
        steady=False,
        version=model_version,
        initialize=False,
        #nxfe=1,
    )
    helper = DifferentialHelper(m, m.fs.time)
    igraph = IncidenceGraphInterface()

    t = m.fs.time.get_finite_elements()[1]

    variables, constraints = helper.get_subsystem_at_time(t)
    deriv_vars, diff_eqns = helper.get_naive_differential_subsystem_at_time(t)
    alg_vars, alg_eqns = helper.get_naive_algebraic_subsystem_at_time(t)

    deriv_alg_set = ComponentSet(deriv_vars + alg_vars)
    diff_alg_set = ComponentSet(diff_eqns + alg_eqns)

    diff_vars = [var for var in variables if var not in deriv_alg_set]
    disc_eqns = [con for con in constraints if con not in diff_alg_set]

    #alg_var_dmp, alg_con_dmp = igraph.dulmage_mendelsohn(alg_vars, alg_eqns)
    #diff_var_dmp, diff_con_dmp = igraph.dulmage_mendelsohn(deriv_vars, diff_eqns)

    print("N. vars: %s" % len(variables))
    print("N. eqns: %s" % len(constraints))
    print("N. diff vars: %s" % len(diff_vars))
    print("N. disc eqns: %s" % len(disc_eqns))
    print("N. deriv vars: %s" % len(deriv_vars))
    print("N. diff eqns: %s" % len(diff_eqns))
    print("N. alg vars: %s" % len(alg_vars))
    print("N. alg eqns: %s" % len(alg_eqns))

    # I want to display the t-Jacobian in irreducible block triangular form,
    # then identify the differential variables and discretization equations
    # that "aren't valid"

    var_dmp, con_dmp = igraph.dulmage_mendelsohn(variables, constraints)
    var_block_map, con_block_map = igraph.block_triangularize(
        var_dmp.square, con_dmp.square
    )
    var_blocks, con_blocks = igraph.get_diagonal_blocks(
        var_dmp.square, con_dmp.square
    )

    print("Invalid diff. vars/disc. eqns:")
    valid = []
    diff_var_blocks = []
    disc_eqn_blocks = []
    for state, deriv, disc in helper._diff_deriv_disc_list:
        if (
            state[t] in var_block_map
            and deriv[t] in var_block_map
            and disc[t] in con_block_map
        ):
            # All three components are part of the square subsystem.
            # We can make some statement about whether they can be
            # matched with each other.
            #
            # TODO: What if they are in the underconstrained subsystem?
            if var_block_map[state[t]] == con_block_map[disc[t]]:
                valid.append((state, deriv, disc))
            else:
                print()
                print("Diff. var: %s" % state[t].name)
                print("Disc. eqn: %s" % disc[t].name)
                diff_var_blocks.append(var_block_map[state[t]])
                disc_eqn_blocks.append(con_block_map[disc[t]])

    print()
    print("These diff. vars live in blocks:")
    print(diff_var_blocks)
    print("These disc. eqns live in blocks:")
    print(disc_eqn_blocks)
    # There is no overlap between these blocks, so we can colorcode
    # the nonzero entries.

    var_order = sum(var_blocks, [])
    con_order = sum(con_blocks, [])
    matrix = get_structural_incidence_matrix(var_order, con_order)
    var_idx_map = ComponentMap((var, i) for i, var in enumerate(var_order))
    con_idx_map = ComponentMap((con, i) for i, con in enumerate(con_order))

    fname = "blt_subsystem"
    transparent = True
    fig, ax = plot_incidence_matrix(
        matrix,
        fname=fname,
        transparent=transparent,
        save=True,
        markersize=2,
    )

    for idx in diff_var_blocks:
        variables = var_blocks[idx]
        constraints = con_blocks[idx]
        submatrix = get_structural_incidence_matrix(variables, constraints)
        row_offset = con_idx_map[constraints[0]]
        col_offset = var_idx_map[variables[0]]
        row = [r + row_offset for r in submatrix.row]
        col = [c + col_offset for c in submatrix.col]
        data = submatrix.data
        projection = sps.coo_matrix(
            (data, (row, col)),
            shape=matrix.shape,
        )
        ax.spy(projection, markersize=3, color="orange")

    for idx in disc_eqn_blocks:
        variables = var_blocks[idx]
        constraints = con_blocks[idx]
        submatrix = get_structural_incidence_matrix(variables, constraints)
        row_offset = con_idx_map[constraints[0]]
        col_offset = var_idx_map[variables[0]]
        row = [r + row_offset for r in submatrix.row]
        col = [c + col_offset for c in submatrix.col]
        data = submatrix.data
        projection = sps.coo_matrix(
            (data, (row, col)),
            shape=matrix.shape,
        )
        ax.spy(projection, markersize=3, color="red")

    fig.savefig(fname + ".pdf", transparent=transparent)


def main():
    reclassify_differential()


if __name__ == "__main__":
    main()
