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


FNAME = "dae_partition"


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

    # I want to plot the discretization, then algebraic, then differential
    # subsystems at this time point.
    # Can just combine the lists of equations then use
    # get_structural_incidence_matrix.

    var_order = diff_vars + alg_vars + deriv_vars
    con_order = disc_eqns + alg_eqns + diff_eqns
    matrix = get_structural_incidence_matrix(var_order, con_order)
    fname = FNAME
    fig, ax = plot_incidence_matrix(
        matrix,
        fname=fname,
        transparent=True,
        save=save,
        show=show,
        markersize=2
    )

    row_offset = len(disc_eqns)
    col_offset = len(diff_vars)
    alg_jac = get_structural_incidence_matrix(alg_vars, alg_eqns)
    row = [r + row_offset for r in alg_jac.row]
    col = [c + col_offset for c in alg_jac.col]
    data = alg_jac.data
    full_shape = matrix.shape
    # Projection of Jacobian onto algebraic coordinates
    projection = sps.coo_matrix(
        (data, (row, col)),
        shape=matrix.shape,
    )
    ax.spy(projection, markersize=2, color="orange")
    ax.tick_params(bottom=False)
    if save:
        fig.savefig(fname + ".pdf", transparent=True)


def main():
    partition_subsystem_by_dae()


if __name__ == "__main__":
    main()
