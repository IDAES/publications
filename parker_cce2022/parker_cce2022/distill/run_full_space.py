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
from pyomo.environ import *
from pyomo.dae import *
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.common.timing import HierarchicalTimer
from parker_cce2022.distill.distill_DAE import make_model
from parker_cce2022.distill.config import get_ipopt

import time as time_lib

PLOT = False
DISPLAY = False

def discretize_model(instance):
    # Discretize using Finite Difference Approach
    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(instance,nfe=50,scheme='BACKWARD')
    
    # Discretize using Orthogonal Collocation
    # discretizer = TransformationFactory('dae.collocation')
    # discretizer.apply_to(instance,nfe=50,ncp=3)


def add_objective(instance):
    # The objective function in the manually discretized pyomo model
    # iterated over all finite elements and all collocation points.  Since
    # the objective function is not explicitly indexed by a ContinuousSet
    # we add the objective function to the model after it has been
    # discretized to ensure that we include all the discretization points
    # when we take the sum.
    
    def obj_rule(m):
        return m.alpha*sum((m.y[1,i] - m.y1_ref)**2 for i in m.t if i != 1) + m.rho*sum((m.u1[i] - m.u1_ref)**2 for i in m.t if i!=1)
    instance.OBJ = Objective(rule=obj_rule) 
    
    # Calculate setpoint in terms of diff vars and create new objective accordingly
    t0 = instance.t.first()
    to_reset = [instance.y[1, t0], instance.x[1, t0]]
    with TemporarySubsystemManager(to_reset=to_reset):
        instance.y[1, t0].set_value(value(instance.y1_ref))
        calculate_variable_from_constraint(
                instance.x[1, t0],
                instance.mole_frac_balance[1, t0],
                )
        instance.x1_ref = Param(initialize=instance.x[1, t0].value)
    def diff_var_obj_rule(m):
        return (
                m.alpha*sum((m.x[1,i] - m.x1_ref)**2 for i in m.t if i != 1) +
                m.rho*sum((m.u1[i] - m.u1_ref)**2 for i in m.t if i != 1)
                )
    instance.DIFF_VAR_OBJ = Objective(rule=diff_var_obj_rule)
    
    # Deactivate old objective function
    instance.OBJ.deactivate()


def run_optimization(instance):
    timer = HierarchicalTimer()
    import pyomo.opt.base.solvers as solver_module
    solver_module.TIMER = timer
    t_start = time_lib.time()
    timer.start("full-space-solve")
    solver = get_ipopt()
    results = solver.solve(instance,tee=True)
    timer.stop("full-space-solve")
    t_end = time_lib.time()
    
    print(timer)
    
    print("Solve time:", t_end-t_start)
    
    data_dict = {
            str(ComponentUID(var)): var.value
            for var in instance.component_data_objects(Var)
            }
    #import json
    #with open("full_space_data.json", "w") as fp:
    #    json.dump(data_dict, fp)

    # Display some values
    if DISPLAY:
        print(ComponentUID(instance.x[1, :]), ComponentUID(instance.rr[:])) 
        for t in instance.t:
            print(instance.x[1, t].value, instance.rr[t].value)

    # If you have matplotlib you can use the following code to plot the
    # results
    if PLOT:
        t = [] 
        x5 = [] 
        x20 = []

        for i in sorted(instance.t): 
            x5.append(value(instance.x[5,i]))
            x20.append(value(instance.x[20,i]))
            t.append(i)

        import matplotlib.pyplot as plt

        plt.plot(t,x5)
        plt.plot(t,x20)
        plt.show()

    return data_dict


def run_full_space_optimization():
    model = make_model()
    file_dir = os.path.dirname(__file__)
    fname = os.path.join(file_dir, "distill.dat")
    instance = model.create_instance(fname)
    discretize_model(instance)
    add_objective(instance)
    data = run_optimization(instance)
    return data


def main():
    model = make_model()
    instance = model.create_instance('distill.dat')
    discretize_model(instance)
    add_objective(instance)
    data = run_optimization(instance)


if __name__ == "__main__":
    main()
