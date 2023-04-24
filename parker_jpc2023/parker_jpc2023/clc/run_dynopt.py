import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.util.subsystems import TemporarySubsystemManager
from idaes.core.util import scaling as iscale
from parker_jpc2023.clc.initialize import initialize_steady
from parker_jpc2023.clc.model import (
    make_model,
    get_state_variable_names,
    preprocess_dynamic_model,
)
from parker_jpc2023.clc.initialize_by_element import (
    initialize_by_time_element
)


def run_dynopt(tfe_per_sample=3, samples_per_horizon=10):
    tfe_width = 60.0
    ntfe = tfe_per_sample * samples_per_horizon
    sample_width = tfe_per_sample * tfe_width
    sample_points = [i*sample_width for i in range(samples_per_horizon+1)]
    each_sim_horizon = ntfe * tfe_width
    horizon = samples_per_horizon * sample_width

    #
    # Create solver and set options
    #
    solver = pyo.SolverFactory("ipopt")
    solver.options["print_user_options"] = "yes"
    # user-scaling might help to some extent
    # Actually makes things worse in dynamic simulation
    #solver.options["nlp_scaling_method"] = "user-scaling"

    # This is necessary to "converge" several solid phase equations:
    # enthalpy balances, mass balances, reaction rate equations.
    solver.options["tol"] = 1e-5
    solver.options["linear_solver"] = "ma57"
    ###

    #
    # Make optimization model
    #
    m = make_model(
        dynamic=True,
        ntfe=ntfe,
        tfe_width=tfe_width,
    )
    m_interface = mpc.DynamicModelInterface(m, m.fs.time)
    ###

    #
    # Use steady state model to get initial conditions
    #
    m_init = make_model()
    m_init_interface = mpc.DynamicModelInterface(m_init, m_init.fs.time)

    # Specify inputs for initial steady state
    # Defaults are about 128, 591.
    init_inputs = mpc.ScalarData({
        "fs.moving_bed.gas_phase.properties[*,0.0].flow_mol": 128.2,
        "fs.moving_bed.solid_phase.properties[*,1.0].flow_mass": 591.4,
    })
    m_init_interface.load_data(init_inputs)
    iscale.calculate_scaling_factors(m_init)

    initialize_steady(m_init)
    solver.solve(m_init, tee=True)

    init_data = m_init_interface.get_data_at_time()
    scalar_data = m_init_interface.get_scalar_variable_data()
    ###

    #
    # Define a disturbance that will be applied.
    #
    disturbance_dict = {"CO2": 0.5, "H2O": 0.0, "CH4": 0.5}
    disturbance = mpc.IntervalData(
        {
            "fs.moving_bed.gas_phase.properties[*,0.0].mole_frac_comp[%s]" % j: [val]
            for j, val in disturbance_dict.items()
        },
        [(0.0, horizon)],
    )
    ###

    #
    # Use steady state model to get target setpoint conditions
    #
    m_setpoint = make_model()
    setpoint_interface = mpc.DynamicModelInterface(m_setpoint, m_setpoint.fs.time)
    setpoint_inputs = mpc.ScalarData({
        "fs.moving_bed.gas_phase.properties[*,0.0].flow_mol": 128.2,
        "fs.moving_bed.solid_phase.properties[*,1.0].flow_mass": 591.4,
    })
    setpoint_interface.load_data(setpoint_inputs)
    iscale.calculate_scaling_factors(m_setpoint)
    setpoint_interface.load_data(disturbance.get_data_at_time(horizon))
    initialize_steady(m_setpoint)
    solver.solve(m_setpoint, tee=True)
    sp_target = {"fs.moving_bed.solid_phase.reactions[*,0.0].OC_conv": 0.95}
    m_setpoint.fs.moving_bed.gas_inlet.flow_mol[:].unfix()
    (
        m_setpoint.penalty_set, m_setpoint.conv_penalty
    ) = setpoint_interface.get_penalty_from_target(sp_target)
    m_setpoint.obj = pyo.Objective(expr=sum(m_setpoint.conv_penalty.values()))
    solver.solve(m_setpoint, tee=True)
    setpoint_data = setpoint_interface.get_data_at_time()
    ###

    # Add objective function to model
    length_domain = m.fs.moving_bed.length_domain
    mbr = m.fs.moving_bed
    setpoint_states = get_state_variable_names(length_domain)
    weight_data = {}
    weight_data.update(
        {mbr.gas_phase.properties[:, x].flow_mol: 0.2 for x in length_domain}
    )
    weight_data.update(
        {mbr.gas_phase.properties[:, x].temperature: 0.1 for x in length_domain}
    )
    weight_data.update({
        mbr.gas_phase.properties[:, x].pressure: 10.0*(1e-5)**2
        for x in length_domain
    })
    weight_data.update({
        mbr.gas_phase.properties[:, x].mole_frac_comp[j]: 200.0
        for x in length_domain for j in ["CH4", "CO2", "H2O"]
    })
    weight_data.update(
        {mbr.solid_phase.properties[:, x].flow_mass: 0.2 for x in length_domain}
    )
    weight_data.update({
        mbr.solid_phase.properties[:, x].temperature: 0.1 for x in length_domain
    })
    weight_data.update({
        mbr.solid_phase.properties[:, x].mass_frac_comp[j]: 200.0
        for x in length_domain for j in ["Fe2O3", "Fe3O4", "Al2O3"]
    })

    m.tracking_var_set, m.tracking_cost = m_interface.get_penalty_from_target(
        setpoint_data,
        variables=setpoint_states,
        weight_data=weight_data,
    )

    #from pyomo.contrib.mpc.modeling.terminal import get_terminal_penalty
    # Terminal equality constraint?
    # I don't think we have the dof for a terminal equality constraint.
    tf = m.fs.time.last()
    m.terminal_con = pyo.Constraint(expr=sum(m.tracking_cost[:, tf]) <= 1e-1)

    #obj_time = sample_points[1:]
    obj_time = m.fs.time
    m.tracking_obj = pyo.Objective(expr=sum(
        m.tracking_cost[i, t]
        for i in m.tracking_var_set for t in obj_time
        if t != m.fs.time.first()
    ))

    # Unfix inputs
    input_variables = [
        m.fs.moving_bed.gas_inlet.flow_mol,
        m.fs.moving_bed.solid_inlet.flow_mass,
    ]
    for var in input_variables:
        for t in m.fs.time:
            if t != m.fs.time.first():
                var[t].unfix()

    # Add piecewise constant constraints
    m.pwc_set, m.pwc_con = m_interface.get_piecewise_constant_constraints(
        input_variables,
        sample_points,
        tolerance=1e-8,
    )

    #
    # Initialize dynamic optimization model to steady state model
    #
    m_interface.load_data(init_data)
    m_interface.load_data(scalar_data)
    ###

    #
    # Load feed-forward disturbance into dynamic optimization model
    #
    m_interface.load_data(disturbance)
    ###

    preprocess_dynamic_model(m)

    input_vardata = [
        var[t] for var in input_variables for t in m.fs.time
        if t != m.fs.time.first()
    ]
    print("Initializing by time element...")
    #with TemporarySubsystemManager(
    #    to_fix=input_vardata,
    #    to_deactivate=[m.pwc_con, m.terminal_con],
    #):
    #    initialize_by_time_element(m, m.fs.time, solver, solve_kwds={"tee": False})
    m.fs.moving_bed.solid_phase.reactions[:, 0.0].OC_conv.setlb(0.89)

    #solver.options["max_iter"] = 20
    res = solver.solve(m, tee=True)

    # Solve
    #from pyomo.contrib.pynumero.algorithms.solvers.callbacks import InfeasibilityCallback
    #callback = InfeasibilityCallback(infeasibility_threshold=1e0)

    #cyipopt = pyo.SolverFactory(
    #    "cyipopt",
    #    options=dict(solver.options),
    #    #intermediate_callback=callback,
    #)
    #cyipopt.solve(m, tee=True)

    #from idaes.core.util.model_statistics import large_residuals_set
    #print("Large residuals:")
    #for con in large_residuals_set(m):
    #    resid = pyo.value(con.body-con.upper)
    #    print(f"  {resid}    {con.name}")

    #pyo.assert_optimal_termination(res)

    from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    igraph = IncidenceGraphInterface(m)

    return m


def main():
    m = run_dynopt()

    for t in m.fs.time:
        tr_cost = pyo.value(sum(m.tracking_cost[:, t]))
        print(f"t = {t}: {tr_cost}")


if __name__ == "__main__":
    main()
