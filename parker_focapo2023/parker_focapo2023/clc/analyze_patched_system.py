from time import time as get_current_time
from parker_focapo2023.clc.model import (
    make_model,
    add_piecewise_constant_constraints,
    add_objective,
    ModelVersion,
)
from parker_focapo2023.clc.kkt import (
    get_equality_constrained_kkt_matrix,
)
from parker_focapo2023.clc.plot import plot_incidence_matrix
from parker_focapo2023.clc.config import (
    get_light_color,
    get_dark_color,
)

from parker_focapo2023.common.dae_utils import (
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
import matplotlib.pyplot as plt
import numpy as np


USE_PDF = True


def check_subsystems(m, show=False, save=False):
    time = m.fs.time
    helper = DifferentialHelper(m, time)
    igraph = IncidenceGraphInterface(m)

    n_var = len(igraph.variables)
    n_con = len(igraph.constraints)
    if n_var != n_con:
        print("System is not square: %s x %s" % (n_con, n_var))
    else:
        dim = n_var
        print("System is square: %s x %s" % (n_con, n_var))

    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    if var_dmp.unmatched or con_dmp.unmatched:
        print("System is structurally singular")
        raise RuntimeError()
    else:
        print("System is structurally nonsingular")

    vars_by_time = []
    cons_by_time = []
    for t in time:
        subsystem = helper.get_subsystem_at_time(t)
        variables, constraints = subsystem
        vars_by_time.append(variables)
        cons_by_time.append(constraints)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn(*subsystem)
        if var_dmp.unmatched or con_dmp.unmatched:
            print("Subsystem at t=%s is structurally singular" % t)
            raise RuntimeError()
        else:
            print("Subsystem at t=%s is structurally nonsingular" % t)
        alg_subsystem = helper.get_algebraic_subsystem_at_time(t)
        alg_var_dmp, alg_con_dmp = igraph.dulmage_mendelsohn(*subsystem)
        if alg_var_dmp.unmatched or alg_con_dmp.unmatched:
            print("Algebraic subsystem at t=%s is structurally singular" % t)
            raise RuntimeError()
        else:
            print("Algebraic subsystem at t=%s is structurally nonsingular" % t)

    var_order = sum(vars_by_time, [])
    con_order = sum(cons_by_time, [])
    imat = get_structural_incidence_matrix(var_order, con_order)
    fname = "patched_time_partition"
    fig, ax = plot_incidence_matrix(
        imat,
        show=False,
        save=False,
        #fname=fname,
        color=get_dark_color(),
        markersize=2,
    )

    full_shape = imat.shape
    offset = 0
    for variables, constraints in zip(vars_by_time, cons_by_time):
        # Want these submatrices to be projected onto their coordinates
        # in the full system.
        submatrix = get_structural_incidence_matrix(variables, constraints)
        dim = submatrix.shape[0]
        row = [r + offset for r in submatrix.row]
        col = [c + offset for c in submatrix.col]
        data = submatrix.data
        projected = sps.coo_matrix(
            (data, (row, col)),
            shape=full_shape,
        )
        ax.spy(projected, markersize=2)
        offset += dim
    ax.tick_params(bottom=False)
    extension = ".pdf" if USE_PDF else ".png"
    if save:
        fig.savefig(fname + extension, transparent=True)
    if show:
        plt.show()


def plot_subsystem_at_t(m, t=None, show=False, save=False):
    time = m.fs.time
    t0 = time.first()
    if t is None:
        t = time.next(t0)
    helper = DifferentialHelper(m, time)

    variables, constraints = helper.get_subsystem_at_time(t)
    deriv_vars, diff_eqns = helper.get_differential_subsystem_at_time(t)
    alg_vars, alg_eqns = helper.get_algebraic_subsystem_at_time(t)

    deriv_alg_set = ComponentSet(deriv_vars + alg_vars)
    diff_alg_set = ComponentSet(diff_eqns + alg_eqns)

    diff_vars = [var for var in variables if var not in deriv_alg_set]
    disc_eqns = [con for con in constraints if con not in diff_alg_set]

    print("N. vars: %s" % len(variables))
    print("N. eqns: %s" % len(constraints))
    print("N. diff vars: %s" % len(diff_vars))
    print("N. disc eqns: %s" % len(disc_eqns))
    print("N. deriv vars: %s" % len(deriv_vars))
    print("N. diff eqns: %s" % len(diff_eqns))
    print("N. alg vars: %s" % len(alg_vars))
    print("N. alg eqns: %s" % len(alg_eqns))

    igraph = IncidenceGraphInterface()
    var_blocks, con_blocks = igraph.get_diagonal_blocks(alg_vars, alg_eqns)
    alg_varorder = sum(reversed(var_blocks), [])
    alg_conorder = sum(reversed(con_blocks), [])

    var_order = diff_vars + alg_varorder + deriv_vars
    con_order = disc_eqns + alg_conorder + diff_eqns
    matrix = get_structural_incidence_matrix(var_order, con_order)
    fname = "patched_dae_partition"
    fig, ax = plot_incidence_matrix(
        matrix,
        fname=fname,
        transparent=True,
        show=False,
        save=False,
        markersize=2,
        color=get_light_color(),
    )

    row_offset = len(disc_eqns)
    col_offset = len(diff_vars)
    alg_jac = get_structural_incidence_matrix(alg_varorder, alg_conorder)
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
    if show:
        plt.show()


def plot_algebraic_system_at_t(m, t=None, show=False, save=False):
    time = m.fs.time
    t0 = time.first()
    if t is None:
        t = time.next(t0)
    helper = DifferentialHelper(m, time)
    igraph = IncidenceGraphInterface()

    m._obj = pyo.Objective(expr=0)
    nlp = PyomoNLP(m)

    alg_vars, alg_eqns = helper.get_algebraic_subsystem_at_time(t)
    var_blocks, con_blocks = igraph.get_diagonal_blocks(alg_vars, alg_eqns)

    n_higher_dim = 0
    for i, (variables, constraints) in enumerate(zip(var_blocks, con_blocks)):
        dim = len(variables)
        submatrix = nlp.extract_submatrix_jacobian(variables, constraints)
        if dim > 1:
            cond = np.linalg.cond(submatrix.toarray())
            print("Block %s, dim = %s" % (i, dim))
            print("  Condition no. = %1.2e" % cond)
            n_higher_dim += 1
        else:
            entry = submatrix.toarray()[0, 0]
            if entry == 0.0:
                raise RuntimeError("Matrix is singular")
    print("N. blocks with dim > 1: %s" % n_higher_dim)

    var_order = sum(reversed(var_blocks), [])
    con_order = sum(reversed(con_blocks), [])
    imat = get_structural_incidence_matrix(var_order, con_order)
    fname = "patched_alg_jac_dmp"
    plot_incidence_matrix(
        imat,
        save=save,
        show=show,
        fname=fname,
        pdf=USE_PDF,
        transparent=True,
        color=get_dark_color(),
    )


def main():
    model_version = ModelVersion.IDAES_1_7_patch1
    m = make_model(
        steady=False,
        version=model_version,
        initialize=True,
    )
    check_subsystems(m, show=True)
    plot_subsystem_at_t(m, show=True)
    plot_algebraic_system_at_t(m, show=True)


if __name__ == "__main__":
    main()
