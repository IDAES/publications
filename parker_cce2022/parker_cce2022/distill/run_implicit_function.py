#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import pyomo.environ as pyo
from pyomo.dae import *
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
)
from parker_cce2022.distill.distill_DAE import make_model

PLOT = False


def discretize_model(instance):
    # Discretize using Finite Difference Approach
    discretizer = pyo.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(instance,nfe=50,scheme='BACKWARD')
    
    # Discretize using Orthogonal Collocation
    # discretizer = TransformationFactory('dae.collocation')
    # discretizer.apply_to(instance,nfe=50,ncp=3)
    
    # The objective function in the manually discretized pyomo model
    # iterated over all finite elements and all collocation points.  Since
    # the objective function is not explicitly indexed by a ContinuousSet
    # we add the objective function to the model after it has been
    # discretized to ensure that we include all the discretization points
    # when we take the sum.


def add_objective(instance):
    def obj_rule(m):
        return m.alpha*sum((m.y[1,i] - m.y1_ref)**2 for i in m.t if i != 1) + m.rho*sum((m.u1[i] - m.u1_ref)**2 for i in m.t if i!=1)
    instance.OBJ = pyo.Objective(rule=obj_rule)

    # Calculate setpoint for the reduced space.
    # Reduced space objective must not use algebraic variables.
    t0 = instance.t.first()
    to_reset = [instance.y[1, t0], instance.x[1, t0]]
    with TemporarySubsystemManager(to_reset=to_reset):
        instance.y[1, t0].set_value(pyo.value(instance.y1_ref))
        calculate_variable_from_constraint(
                instance.x[1, t0],
                instance.mole_frac_balance[1, t0],
                )
        instance.x1_ref = pyo.Param(initialize=instance.x[1, t0].value)

    def rs_obj_rule(m):
        return (
                m.alpha*sum((m.x[1,i] - m.x1_ref)**2 for i in m.t if i != 1) +
                m.rho*sum((m.u1[i] - m.u1_ref)**2 for i in m.t if i != 1)
                )
    instance.REDUCED_SPACE_OBJ = pyo.Objective(rule=rs_obj_rule)
    instance.OBJ.deactivate()


def run_optimization(instance):
    diff_vars = [pyo.Reference(instance.x[i, :]) for i in instance.S_TRAYS]
    deriv_vars = [pyo.Reference(instance.dx[i, :]) for i in instance.S_TRAYS]
    disc_eqns = [pyo.Reference(instance.dx_disc_eq[i, :]) for i in instance.S_TRAYS]
    diff_eqns = [pyo.Reference(instance.diffeq[i, :]) for i in instance.S_TRAYS]

    n_diff = len(diff_vars)
    assert n_diff == len(deriv_vars)
    assert n_diff == len(disc_eqns)
    assert n_diff == len(diff_eqns)

    alg_vars = []
    alg_eqns = []
    alg_vars.extend(pyo.Reference(instance.y[i, :]) for i in instance.S_TRAYS)
    alg_eqns.extend(pyo.Reference(instance.mole_frac_balance[i, :])
            for i in instance.S_TRAYS)
    # Since we are not adding them to the reduced space model, alg vars do not
    # need to be references.
    alg_vars.append(instance.rr)
    alg_vars.append(instance.L)
    alg_vars.append(instance.V)
    alg_vars.append(instance.FL)

    alg_eqns.append(instance.reflux_ratio)
    alg_eqns.append(instance.flowrate_rectification)
    alg_eqns.append(instance.vapor_column)
    alg_eqns.append(instance.flowrate_stripping)

    input_vars = [pyo.Reference(instance.u1[:])]

    # Create a block to hold the reduced space model
    reduced_space = pyo.Block(concrete=True)
    reduced_space.obj = pyo.Reference(instance.REDUCED_SPACE_OBJ)

    n_input = len(input_vars)

    def differential_block_rule(b, i):
        b.state = diff_vars[i]
        b.deriv = deriv_vars[i]
        b.disc = disc_eqns[i]

    def input_block_rule(b, i):
        b.var = input_vars[i]

    reduced_space.differential_block = pyo.Block(
            range(n_diff),
            rule=differential_block_rule,
            )
    reduced_space.input_block = pyo.Block(
            range(n_input),
            rule=input_block_rule,
            )

    reduced_space.external_block = ExternalGreyBoxBlock(instance.t)

    # Add reference to the constraint that specifies the initial conditions
    reduced_space.init_rule = pyo.Reference(instance.init_rule)

    for t in instance.t:
        if t == instance.t.first():
            reduced_space.external_block[t].deactivate()
            continue
        # Create and set external model for every external block
        reduced_space_vars = (
                list(reduced_space.input_block[:].var[t]) +
                list(reduced_space.differential_block[:].state[t]) +
                list(reduced_space.differential_block[:].deriv[t])
                )
        external_vars = [v[t] for v in alg_vars]
        residual_cons = [c[t] for c in diff_eqns]
        external_cons = [c[t] for c in alg_eqns]
        reduced_space.external_block[t].set_external_model(
                ExternalPyomoModel(
                    reduced_space_vars,
                    external_vars,
                    residual_cons,
                    external_cons,
                    ),
                inputs=reduced_space_vars,
                )

    solver = pyo.SolverFactory("cyipopt")

    import time as time_lib
    t_start = time_lib.time()
    results = solver.solve(reduced_space, tee=True)
    t_end = time_lib.time()

    print("Solve time:", t_end-t_start)

    import json
    data_dict = {
        str(pyo.ComponentUID(var)): var.value
        for var in instance.component_data_objects(pyo.Var)
    }
    #with open("reduced_space_data.json", "w") as fp:
    #    json.dump(data_dict, fp)

    # Display some values
    print(pyo.ComponentUID(instance.x[1, :]), pyo.ComponentUID(instance.rr[:]))
    for t in instance.t:
        print(instance.x[1, t].value, instance.rr[t].value)

    if PLOT:
        # If you have matplotlib you can use the following code to plot the
        # results
        t = [] 
        x5 = [] 
        x20 = []

        for i in sorted(instance.t): 
            x5.append(pyo.value(instance.x[5,i]))
            x20.append(pyo.value(instance.x[20,i]))
            t.append(i)

        import matplotlib.pyplot as plt

        plt.plot(t,x5)
        plt.plot(t,x20)
        plt.show()

    return data_dict


def run_implicit_function_optimization():
    model = make_model()
    file_dir = os.path.dirname(__file__)
    fname = os.path.join(file_dir, "distill.dat")
    instance = model.create_instance(fname)
    discretize_model(instance)
    add_objective(instance)
    data = run_optimization(instance)
    return data


def main():
    run_implicit_function_optimization()


if __name__ == "__main__":
    main()
