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
from parker_focapo2023.config import (
    get_light_color,
    get_dark_color,
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


USE_PDF = True
FNAME = "valid_dae_partition"


def get_dmp_order(alg_vars, alg_eqns):
    igraph = IncidenceGraphInterface()
    var_dmp, con_dmp = igraph.dulmage_mendelsohn(alg_vars, alg_eqns)
    sq_var_blocks, sq_con_blocks = igraph.get_diagonal_blocks(
        var_dmp.square, con_dmp.square
    )
    sq_varorder = list(reversed(sum(sq_var_blocks, [])))
    sq_conorder = list(reversed(sum(sq_con_blocks, [])))

    undercon_vars = var_dmp.unmatched + var_dmp.underconstrained
    undercon_cons = con_dmp.underconstrained
    overcon_vars = var_dmp.overconstrained
    overcon_cons = con_dmp.overconstrained + con_dmp.unmatched
    undercon_var_blocks, undercon_con_blocks = igraph.get_connected_components(
        undercon_vars, undercon_cons
    )
    undercon_varorder = sum(undercon_var_blocks, [])
    undercon_conorder = sum(undercon_con_blocks, [])
    overcon_var_blocks, overcon_con_blocks = igraph.get_connected_components(
        overcon_vars, overcon_cons
    )
    overcon_varorder = sum(overcon_var_blocks, [])
    overcon_conorder = sum(overcon_con_blocks, [])
    dmp_varorder = undercon_varorder + sq_varorder + overcon_varorder
    dmp_conorder = undercon_conorder + sq_conorder + overcon_conorder
    return (dmp_varorder, dmp_conorder)


def partition_subsystem_by_dae(show=False, save=False):
    model_version = ModelVersion.IDAES_1_7
    m = make_model(
        steady=False,
        version=model_version,
        initialize=False,
    )
    helper = DifferentialHelper(m, m.fs.time)
    igraph = IncidenceGraphInterface()

    t = m.fs.time.get_finite_elements()[1]

    variables, constraints = helper.get_subsystem_at_time(t)
    deriv_vars, diff_eqns = helper.get_differential_subsystem_at_time(t)
    alg_vars, alg_eqns = helper.get_algebraic_subsystem_at_time(t)

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

    # I want to plot the discretization, then algebraic, then differential
    # subsystems at this time point.
    # Can just combine the lists of equations then use
    # get_structural_incidence_matrix.

    #var_order = diff_vars + alg_vars + deriv_vars
    #con_order = disc_eqns + alg_eqns + diff_eqns
    alg_varorder, alg_conorder = get_dmp_order(alg_vars, alg_eqns)
    var_order = diff_vars + alg_varorder + deriv_vars
    con_order = disc_eqns + alg_conorder + diff_eqns
    matrix = get_structural_incidence_matrix(var_order, con_order)
    fname = FNAME
    fig, ax = plot_incidence_matrix(
        matrix,
        fname=fname,
        transparent=True,
        save=save,
        show=show,
        markersize=2,
        color=get_light_color(),
    )

    row_offset = len(disc_eqns)
    col_offset = len(diff_vars)
    alg_jac = get_structural_incidence_matrix(alg_varorder, alg_conorder)
    # TODO: Permute to
    row = [r + row_offset for r in alg_jac.row]
    col = [c + col_offset for c in alg_jac.col]
    data = alg_jac.data
    full_shape = matrix.shape
    # Projection of Jacobian onto algebraic coordinates
    projection = sps.coo_matrix(
        (data, (row, col)),
        shape=matrix.shape,
    )
    ax.spy(projection, markersize=2, color=get_dark_color())
    ax.tick_params(bottom=False)
    extension = ".pdf" if USE_PDF else ".png"
    if save:
        fig.savefig(fname + extension, transparent=True)

    #
    # Plot incidence matrix of discretization equations with respect to
    # differential variables
    #
    disc_submatrix = get_structural_incidence_matrix(diff_vars, disc_eqns)
    fname = "disc_jacobian"
    fig, ax = plot_incidence_matrix(
        disc_submatrix,
        fname=fname,
        transparent=True,
        save=save,
        show=show,
        markersize=4,
        pdf=USE_PDF,
    )

    #
    # Plot incidence matrix of derivative variables on differential equations
    #
    diff_submatrix = get_structural_incidence_matrix(deriv_vars, diff_eqns)
    fname = "diff_jacobian"
    fig, ax = plot_incidence_matrix(
        disc_submatrix,
        fname=fname,
        transparent=True,
        save=save,
        show=show,
        markersize=4,
        pdf=USE_PDF,
    )


def main():
    partition_subsystem_by_dae(show=True, save=False)


if __name__ == "__main__":
    main()
