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
    get_light_color,
    get_dark_color,
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
)

import scipy.sparse as sps
import matplotlib.pyplot as plt


FNAME = "time_partition"
USE_PDF = False


def check_structural_singularity(variables, constraints, t=None):
    igraph = IncidenceGraphInterface()
    if len(variables) != len(constraints):
        raise RuntimeError()
    matching = igraph.maximum_matching(variables, constraints)
    if len(matching) != len(variables):
        deficiency = len(variables) - len(matching)
        print(
            "Subsystem at %s is structurally singular with deficiency %s/%s"
            % (t, deficiency, len(variables))
        )
    else:
        print("Subsystem at %s is structurally nonsingular" % t)


def partition_system_by_time(show=False, save=False):
    model_version = ModelVersion.IDAES_1_7
    m = make_model(
        steady=False,
        version=model_version,
        initialize=False,
    )
    helper = DifferentialHelper(m, m.fs.time)
    igraph = IncidenceGraphInterface()
    # Identify subsystem at each point in time.
    # List of subsystems.
    # Combine them to get a variable order
    # Plot incidence matrix in this order

    subsystems = []
    for t in m.fs.time:
        subsystem = helper.get_subsystem_at_time(t)
        subsystems.append(subsystem)
        check_structural_singularity(*subsystem, t=t)

    var_order = sum([variables for variables, _ in subsystems], [])
    con_order = sum([constraints for _, constraints in subsystems], [])
    print("Checking structural singularity of entire system:")
    check_structural_singularity(var_order, con_order)
    m._obj = pyo.Objective(expr=0)
    nlp = PyomoNLP(m)
    jacobian = nlp.extract_submatrix_jacobian(var_order, con_order)
    fname = FNAME
    fig, ax = plot_incidence_matrix(
        jacobian,
        save=save,
        show=show,
        fname=fname,
        color=get_light_color(),
        pdf=USE_PDF,
    )

    #
    # Plot block diagonal in a darker color for emphasis
    #
    full_shape = jacobian.shape
    offset = 0
    for subsystem in subsystems:
        # Want these submatrices to be projected onto their coordinates
        # in the full system.
        submatrix = nlp.extract_submatrix_jacobian(*subsystem)
        dim = submatrix.shape[0]
        row = [r + offset for r in submatrix.row]
        col = [c + offset for c in submatrix.col]
        data = submatrix.data
        projected = sps.coo_matrix(
            (data, (row, col)),
            shape=full_shape,
        )
        ax.spy(projected, markersize=2, color=get_dark_color())
        offset += dim
    ax.tick_params(bottom=False)
    extension = ".pdf" if USE_PDF else ".png"
    if save:
        fig.savefig(fname + extension, transparent=True)


def main():
    partition_system_by_time(show=True)


if __name__ == "__main__":
    main()
