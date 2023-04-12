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
from parker_focapo2023.clc.plot import (
    plot_incidence_matrix,
)
from parker_focapo2023.clc.config import (
    get_light_color,
    get_dark_color,
)

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.dae.flatten import flatten_dae_components
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27

import numpy as np
import scipy.sparse as sps


# The solve with the 1.7 (singular) version takes a while to terminate
# (either timeout or diverging iterates, depending on Ipopt version),
# so by default this script does not actually solve the optimization
# problem.
SOLVE = False


def analyze_with_ma27(
    kkt,
    pivtol=1e-8,
):
    ma27 = MA27(
        a_factor=100.0,
    )

    # Print level to "medium"
    ma27.set_icntl(3, 1)

    t0 = get_current_time()
    ma27.do_symbolic_factorization(kkt)
    t1 = get_current_time()
    t_symbolic = t1 - t0

    pred_double_factor_length = ma27.get_info(7)
    pred_integer_factor_length = ma27.get_info(8)

    double_size = 8
    int_size = 4
    pred_total_factor_size = (
        double_size*pred_double_factor_length
        + int_size*pred_integer_factor_length
    )
    # Would also like to get OPS, the predicted number of flops
    # to be used during numeric factorization (excluding numeric pivoting),
    # but I did not implement an interface for this.

    # Set pivot tolerance
    ma27.set_cntl(3, pivtol)

    t0 = get_current_time()
    try:
        ma27.do_numeric_factorization(kkt)
    except RuntimeError as err:
        print(err)
    t1 = get_current_time()
    t_numeric = t1 - t0

    actual_double_factor_length = ma27.get_info(9)
    actual_integer_factor_length = ma27.get_info(10)
    actual_total_factor_size = (
        double_size*actual_double_factor_length
        + int_size*actual_integer_factor_length
    )

    nrow, ncol = kkt.shape
    print()
    print("Predicted factor size (bytes): %s" % pred_total_factor_size)
    print("Actual total factor size (bytes): %s" % actual_total_factor_size)
    ratio = actual_total_factor_size/float(pred_total_factor_size)
    print("Ratio of actual to predicted memory: %s" % ratio)

    if ma27.get_info(1) == -5:
        # -5 appears to be "structurally singular", while
        # 3 appears to be "numerically singular". In either case,
        # INFO(2) holds the (structural) rank.
        print("Matrix is structurally singular")
        rank = ma27.get_info(2)
    elif ma27.get_info(1) == 3:
        print("Matrix is numerically singular")
        rank = ma27.get_info(2)
    else:
        rank = nrow
    print("Rank: %s/%s" % (rank, nrow))
    print("Time required for symbolic factorization: %1.2f" % t_symbolic)
    print("Time required for numeric factorization: %1.2f" % t_numeric)


def get_dynamic_optimization_model(
    version=ModelVersion.IDAES_1_7,
    target_inputs=None,
):
    if target_inputs is None:
        target_inputs = {}
    m = make_model(
        steady=False,
        version=version,
    )
    t0 = m.fs.time.first()
    t1 = m.fs.time.get_finite_elements()[1]
    tf = m.fs.time.last()

    inputs = add_piecewise_constant_constraints(m)
    for var in inputs:
        var[:].unfix()
        var[t0].fix()

    x0 = m.fs.MB.gas_phase.length_domain.first()
    xf = m.fs.MB.gas_phase.length_domain.last()
    add_objective(m, target_inputs, version=version)
    return m


def main(version, show=False, save=False):
    input_cuids = [
        pyo.ComponentUID("fs.MB.gas_phase.properties[*,0.0].flow_mol"),
        pyo.ComponentUID("fs.MB.solid_phase.properties[*,1.0].flow_mass"),
    ]
    input_values = [140.0, 600.0]

    # Experiment originally ran with these values:
    #input_values = [120.0, 600.0]
    target_inputs = dict(zip(input_cuids, input_values))
    model_version = version
    m = get_dynamic_optimization_model(
        version=model_version,
        target_inputs=target_inputs,
    )

    # TODO: Check if PyNumero MA27 is installed.
    ma27 = True
    if ma27:
        nlp = PyomoNLP(m)
        jac = nlp.evaluate_jacobian()
        kkt = get_equality_constrained_kkt_matrix(nlp, exclude_hessian=False)
        pivtol = 1e-24
        analyze_with_ma27(kkt, pivtol=pivtol)

        nlp.set_duals(np.ones(nlp.n_constraints()))
        kkt = get_equality_constrained_kkt_matrix(nlp, exclude_hessian=False)
        fname = "kkt_matrix"
        pdf = True
        plot_incidence_matrix(
            kkt,
            save=save,
            show=show,
            fname=fname,
            pdf=pdf,
            color=get_dark_color(),
        )
        nlp.set_duals(np.zeros(nlp.n_constraints()))

    if SOLVE:
        ipopt = pyo.SolverFactory("ipopt")
        ipopt.options["max_iter"] = 1000
        ipopt.solve(m, tee=True)


if __name__ == "__main__":
    version = ModelVersion.IDAES_1_7_patch1
    main(version, show=True)
    version = ModelVersion.IDAES_1_7
    main(version, show=True)
