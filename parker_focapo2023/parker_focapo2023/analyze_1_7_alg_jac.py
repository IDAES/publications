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
from parker_focapo2023.plot import plot_incidence_matrix
from parker_focapo2023.config import (
    get_dark_color,
    get_light_color,
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
import matplotlib.pyplot as plt


USE_PDF = True


def analyze_algebraic_jacobian_at_time(m, t, show=False, save=False):
    time = m.fs.time
    helper = DifferentialHelper(m, time)
    igraph = IncidenceGraphInterface()
    alg_vars, alg_eqns = helper.get_algebraic_subsystem_at_time(t)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn(alg_vars, alg_eqns)

    matching = igraph.maximum_matching(alg_vars, alg_eqns)
    inv_matching = ComponentMap((var, con) for con, var in matching.items())
    con_order = alg_eqns
    unmatched_vars = [var for var in alg_vars if var not in inv_matching]

    idx = 0
    var_order = []
    unmatched_coords = []
    for con in alg_eqns:
        idx += 1
        if con in matching:
            var_order.append(matching[con])
        else:
            var_order.append(unmatched_vars.pop())
            unmatched_coords.append(idx)

    print("Unmatched variables at coordinate %s" % unmatched_coords)
    alg_jac = get_structural_incidence_matrix(var_order, con_order)
    fname = "alg_jac_matching"
    plot_incidence_matrix(
        alg_jac,
        save=save,
        show=show,
        fname=fname,
        markersize=1,
        pdf=USE_PDF,
        color=get_dark_color(),
    )

    print("N. alg vars: %s" % len(alg_vars))
    print("N. alg eqns: %s" % len(alg_eqns))

    sq_var_blocks, sq_con_blocks = igraph.get_diagonal_blocks(
        var_dmp.square, con_dmp.square
    )
    sq_varorder = list(reversed(sum(sq_var_blocks, [])))
    sq_conorder = list(reversed(sum(sq_con_blocks, [])))

    #
    # Print info and display Dulmage-Mendelsohn partition
    #
    undercon_vars = var_dmp.unmatched + var_dmp.underconstrained
    undercon_cons = con_dmp.underconstrained
    overcon_vars = var_dmp.overconstrained
    overcon_cons = con_dmp.overconstrained + con_dmp.unmatched
    print(
        "Underconstrained system: %s x %s"
        % (len(undercon_vars), len(undercon_cons))
    )
    print(
        "Square system: %s x %s"
        % (len(var_dmp.square), len(con_dmp.square))
    )
    print(
        "Overconstrained system: %s x %s"
        % (len(overcon_vars), len(overcon_cons))
    )
    dmp_varorder = undercon_vars + sq_varorder + overcon_vars
    dmp_conorder = undercon_cons + sq_conorder + overcon_cons
    dmp_imat = get_structural_incidence_matrix(dmp_varorder, dmp_conorder)
    fname = "alg_jac_dmp"
    plot_incidence_matrix(
        dmp_imat,
        save=save,
        show=show,
        fname=fname,
        pdf=USE_PDF,
        color=get_dark_color(),
    )
    ###

    #
    # Partition under- and over-constrained systems by connected components
    #
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

    print("N. conn comp, underconstrained: %s" % len(undercon_var_blocks))
    print("N. conn comp, overconstrained: %s" % len(overcon_var_blocks))
    print()
    ###

    #dmp_varorder = undercon_vars + sq_varorder + overcon_vars
    #dmp_conorder = undercon_cons + sq_conorder + overcon_cons
    dmp_varorder = undercon_varorder + sq_varorder + overcon_varorder
    dmp_conorder = undercon_conorder + sq_conorder + overcon_conorder
    dmp_imat = get_structural_incidence_matrix(dmp_varorder, dmp_conorder)
    fname = "alg_jac_dmp"
    plot_incidence_matrix(
        dmp_imat,
        save=save,
        show=show,
        fname=fname,
        pdf=USE_PDF,
        color=get_dark_color(),
    )

    #
    # Plot incidence matrices
    #
    print("Underconstrained system:")
    for i, (vars, cons) in enumerate(zip(undercon_var_blocks, undercon_con_blocks)):
        print("Conn. comp. %s" % i)
        print("  Variables:")
        for var in vars:
            print("    %s" % var.name)
        print("  Constraints:")
        for con in cons:
            print("    %s" % con.name)
    print()
    print("Overconstrained system:")
    for i, (vars, cons) in enumerate(zip(overcon_var_blocks, overcon_con_blocks)):
        print("Conn. comp. %s" % i)
        print("  Variables:")
        for var in vars:
            print("    %s" % var.name)
        print("  Constraints:")
        for con in cons:
            print("    %s" % con.name)

    undercon_imat = get_structural_incidence_matrix(
        undercon_varorder, undercon_conorder
    )
    overcon_imat = get_structural_incidence_matrix(
        overcon_varorder, overcon_conorder
    )
    fname = "alg_jac_underconstrained"
    plot_incidence_matrix(
        undercon_imat,
        save=save,
        show=show,
        fname=fname,
        markersize=3,
        pdf=USE_PDF,
        color=get_dark_color(),
    )
    fname = "alg_jac_overconstrained"
    plot_incidence_matrix(
        overcon_imat,
        save=save,
        show=show,
        fname=fname,
        markersize=3,
        pdf=USE_PDF,
        color=get_dark_color(),
    )
    ###

    m.fs.MB.solid_phase.properties[t, 0].density_skeletal_constraint.pprint()
    m.fs.MB.solid_phase.properties[t, 0].density_particle_constraint.pprint()


def main():
    model_version = ModelVersion.IDAES_1_7
    m = make_model(
        steady=False,
        version=model_version,
        initialize=False,
    )
    t1 = m.fs.time.get_finite_elements()[1]
    analyze_algebraic_jacobian_at_time(m, t1, show=True, save=False)


if __name__ == "__main__":
    main()
