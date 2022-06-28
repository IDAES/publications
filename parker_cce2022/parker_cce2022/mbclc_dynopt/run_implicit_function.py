from time import time as get_current_time

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
)
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptNLP,
    CyIpoptSolver,
    _cyipopt_status_enum,
)

from idaes.apps.caprese.categorize import (
    VariableCategory as VC,
    ConstraintCategory as CC,
)

from parker_cce2022.mbclc_dynopt.model import (
    get_steady_state_data,
    get_dynamic_model,
)
from parker_cce2022.mbclc_dynopt.series_data import TimeSeriesTuple
from parker_cce2022.mbclc_dynopt.solve_data import SolveData

import scipy.sparse as sps

from pyomo.common.timing import HierarchicalTimer
TIMER = HierarchicalTimer()

import pyomo.repn.plugins.ampl.ampl_ as nl_module
import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_module
import pyomo.contrib.pynumero.interfaces.external_pyomo_model as epm_module
import pyomo.contrib.pynumero.interfaces.nlp_projections as proj_module

nl_module.TIMER = TIMER
cyipopt_module.TIMER = TIMER
epm_module.TIMER = TIMER
proj_module.TIMER = TIMER


"""
This module defines a function for setting up a dynamic optimization
problem instance and solving with the implicit function formulation.
"""


def get_matrix_sparsity_string(matrix):
    N, M = matrix.shape
    nnz = matrix.nnz
    return "%sx%s, %s nonzeros" % (N, M, nnz)


def print_epm_sparsity_information(epm):
    # Matrix information to display:
    # - (Would like external constraints wrt external variables,
    #   as well as its inverse)
    # - External variables wrt input variables
    # - Residual equations wrt input variables
    # - Hessian of Lagrangian contribution
    jac_ex = sps.coo_matrix(epm.evaluate_jacobian_external_variables())
    # This matrix is dense, as it is just a matrix multiplied by
    # a matrix inverse.
    print("external jacobian:  " + get_matrix_sparsity_string(jac_ex))
    jac_eq = sps.coo_matrix(epm.evaluate_jacobian_equality_constraints())
    print("resid jacobian:     " + get_matrix_sparsity_string(jac_eq))
    hlxx, hlxy, hlyy = epm.get_full_space_lagrangian_hessians()
    hess_lag = sps.coo_matrix(
        epm.calculate_reduced_hessian_lagrangian(hlxx, hlxy, hlyy)
    )
    print("Hessian lagrangian: " + get_matrix_sparsity_string(hess_lag))


def run_dynamic_optimization(
        initial_conditions=None,
        setpoint=None,
        scalar_data=None,
        nxfe=10,
        nxcp=1,
        ntfe_per_sample=2,
        ntcp=1,
        sample_width=120.0,
        samples_per_horizon=15,
        ipopt_options=None,
        raise_exception=False,
        differential_bounds=False,
        input_bounds=False,
        use_cyipopt=True,
        ):
    m, var_cat, con_cat = get_dynamic_model(
        nxfe=nxfe,
        nxcp=nxcp,
        ntfe_per_sample=ntfe_per_sample,
        ntcp=ntcp,
        sample_width=sample_width,
        samples_per_horizon=samples_per_horizon,
        initial_data=initial_conditions,
        setpoint=setpoint,
        scalar_data=scalar_data,
    )
    time = m.fs.time
    t0 = time.first()

    reduced_space = pyo.ConcreteModel()

    # Add differential variables, derivatives, inputs, and discretization
    # equations to reduced space model.
    reduced_space.objective = pyo.Reference(m.tracking_objective)

    diff_vars = var_cat[VC.DIFFERENTIAL]
    deriv_vars = var_cat[VC.DERIVATIVE]
    disc_cons = con_cat[CC.DISCRETIZATION]
    diff_cons = con_cat[CC.DIFFERENTIAL]
    alg_vars = var_cat[VC.ALGEBRAIC]
    alg_cons = con_cat[CC.ALGEBRAIC]
    assert len(diff_vars) == len(deriv_vars)
    assert len(diff_vars) == len(disc_cons)
    assert len(diff_vars) == len(diff_cons)
    assert len(alg_vars) == len(alg_cons)

    input_vars = [
        pyo.Reference(m.fs.MB.gas_inlet.flow_mol[:]),
        pyo.Reference(m.fs.MB.solid_inlet.flow_mass[:]),
    ]

    # Add differential variables, derivative variables, discretization
    # equations, and input variables to the reduced space block.
    # Algebraic variables and equations, as well as the differential
    # equations, are handled by the external block.

    @reduced_space.Block(range(len(input_vars)))
    def input(b, i):
        b.var = pyo.Reference(input_vars[i].referent)

    @reduced_space.Block(range(len(diff_vars)))
    def differential(b, i):
        b.state = pyo.Reference(diff_vars[i].referent)
        b.deriv = pyo.Reference(deriv_vars[i].referent)
        b.disc = pyo.Reference(disc_cons[i].referent)

    reduced_space.external_block = ExternalGreyBoxBlock(time)
    ex_block = reduced_space.external_block

    for t in time:
        if t != t0:
            ex_inputs = (
                list(reduced_space.input[:].var[t])
                + list(reduced_space.differential[:].state[t])
                + list(reduced_space.differential[:].deriv[t])
            )
            ex_vars = [var[t] for var in alg_vars]
            ex_cons = [con[t] for con in alg_cons]
            ex_resids = [con[t] for con in diff_cons]

            ex_model = ExternalPyomoModel(
                ex_inputs, ex_vars, ex_resids, ex_cons,
                use_cyipopt=use_cyipopt,
            )
            resid_scaling_factors = [
                m.scaling_factor[var[t]] for var in diff_vars
            ]
            ex_model.set_equality_constraint_scaling_factors(
                resid_scaling_factors
            )
            ex_block[t].set_external_model(ex_model, inputs=ex_inputs)

    # Remove empty external block, which causes an error in PyNLPwGBB
    ex_block[t0].deactivate()

    # Move scaling_factor to reduced_space block
    scaling_factor = m.scaling_factor
    m.del_component(scaling_factor)
    reduced_space.scaling_factor = scaling_factor

    # Move piecewise constant constraints to reduced_space
    pwc_con = m.pwc_con
    m.del_component(pwc_con)
    reduced_space.pwc_con = pwc_con

    if differential_bounds:
        for var in diff_vars:
            for t in time:
                var[t].setlb(0.0)
    if input_bounds:
        sample_points = [i*sample_width for i in range(samples_per_horizon + 1)]
        sample_points = [
            time.at(time.find_nearest_index(t)) for t in sample_points
        ]
        for t in sample_points:
            m.fs.MB.gas_inlet.flow_mol[t].setlb(0.0)
            m.fs.MB.gas_inlet.flow_mol[t].setub(200.0)

    pynlp = PyomoNLPWithGreyBoxBlocks(reduced_space)
    cynlp = CyIpoptNLP(pynlp)

    # Before the solve, I would like to know something about the sparsity
    # structure of (a) a single implicit function and (b) the entire
    # problem seen by the outer optimization algorithm.
    #
    # Just use the last external model we added to the blocks.
    print_epm_sparsity_information(ex_model)

    if ipopt_options is None:
        ipopt_options = {
            "tol": 5e-5,
            "inf_pr_output": "internal",
            "dual_inf_tol": 1e2,
            "constr_viol_tol": 1e2,
            "compl_inf_tol": 1e2,
            "nlp_scaling_method": "user-scaling",
        }

    cyipopt = CyIpoptSolver(cynlp, options=ipopt_options)

    x0 = pynlp.get_primals()

    print("Starting solve")
    TIMER.start("solve")
    t_start = get_current_time()
    try:
        sol, stat = cyipopt.solve(x0=x0, tee=True)
        stat = _cyipopt_status_enum[stat["status_msg"]]
    except (ValueError, RuntimeError) as err:
        # Implicit function solve failed
        print(err)
        stat = "implicit_function_error"
        if raise_exception:
            raise err
    t_end = get_current_time()
    TIMER.stop("solve")

    #options = None
    #solver = pyo.SolverFactory("cyipopt", options=options)
    #solver.solve(reduced_space, tee=True)

    m.fs.MB.gas_inlet.flow_mol.pprint()
    m.fs.MB.solid_inlet.flow_mass.pprint()

    control_values = TimeSeriesTuple(
        {
            # Use strings as keys here for json-serializability
            str(pyo.ComponentUID(m.fs.MB.gas_inlet.flow_mol.referent)): [
                m.fs.MB.gas_inlet.flow_mol[t].value for t in m.fs.time
            ],
            str(pyo.ComponentUID(m.fs.MB.solid_inlet.flow_mass.referent)): [
                m.fs.MB.solid_inlet.flow_mass[t].value for t in m.fs.time
            ],
            str(pyo.ComponentUID(m.tracking_cost[:])): [
                pyo.value(m.tracking_cost[t]) for t in m.fs.time
            ],
        },
        list(m.fs.time),
    )

    print(TIMER)
    time_elapsed = t_end - t_start
    solve_data = SolveData(stat, control_values, time_elapsed)
    return solve_data


if __name__ == "__main__":
    nxfe = 10
    samples_per_horizon = 10
    ic_scalar_data, ic_dae_data = get_steady_state_data(
        nxfe=nxfe,
    )
    sp_input_map = {"fs.MB.solid_inlet.flow_mass[*]": 700.0}
    sp_dof_names = ["fs.MB.gas_inlet.flow_mol[*]"]
    #sp_dof_names = ["fs.MB.solid_inlet.flow_mass[*]"]
    # Note that this key needs to have index [*,0], not [*,0.0]
    # This doesn't make sense to me. It seems like
    # str-> ComponentUID-> component-> slice-> ComponentUID-> str
    # is somehow changing this index from a float to an int...
    # It was, because of how ComponentUID is processing the string index.

    sp_state_list = [("fs.MB.solid_phase.reactions[*,0.0].OC_conv", 0.95)]

    # Why is this name not valid?
    # Becacuse I dereference this variable? Even though it is attached to
    # a model?
    # The reference corresponding to this name appears to be disconnected
    # from any model. Not true. It is just attached to the model somewhere
    # other than this name, as the Port object solid_inlet is not a Block.
    # If ports could add_component like blocks, then this name could
    # be added to the model as a reference-to-reference.
    # The name we would like to use is just syntactic sugar around the
    # underlying reference, which has a somewhat inconvenient name.
    # And we cannot recover the more convenient name from the underlying
    # (reference) variable object as (I think) this object has no idea
    # it's been attached to a port.
    #
    # Maybe find_component_on should raise an error if it attempts to access
    # something that is not a Block? Or if it tries to descend (e.g. to
    # flow_mass) *from* something that is not an instance of block.
    #
    # In order to get the proper "unique" name from something like this, we
    # need to call find_component on the given name, then construct a CUID
    # from the found component.
    # I think this and checking the type in _resolve_cuid are alternatives
    #
    #sp_state_list = [("fs.MB.solid_inlet.flow_mass[*]", 591.4)]
    #
    # This name is valid:
    #sp_state_list = [("fs.MB.solid_phase.properties[*,1.0].flow_mass", 650.0)]

    _, sp_dae_data = get_steady_state_data(
        nxfe=nxfe,
        input_map=sp_input_map,
        to_unfix=sp_dof_names,
        setpoint_list=sp_state_list,
    )

    use_cyipopt = True
    status, input_data, time = run_dynamic_optimization(
        initial_conditions=ic_dae_data,
        setpoint=sp_dae_data,
        scalar_data=ic_scalar_data,
        nxfe=nxfe,
        samples_per_horizon=samples_per_horizon,
        raise_exception=True,
        differential_bounds=False,
        input_bounds=True,
        use_cyipopt=use_cyipopt,
    )
    for name, values in input_data.data.items():
        print(name)
        for val in values:
            print(val)

    print("Optimization time: %s" % time)
