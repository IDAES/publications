import json
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from idaes.core.util import scaling as iscale
from pyomo.common.timing import TicTocTimer, HierarchicalTimer
from parker_jpc2023.clc.initialize import initialize_steady
from parker_jpc2023.clc.model import (
    make_model,
    preprocess_dynamic_model,
)
from parker_jpc2023.clc.initialize_by_element import (
    initialize_by_time_element
)


def main():
    htimer = HierarchicalTimer()

    htimer.start("root")

    # Parameters pertaining to the dynamic optimization model
    tfe_width = 60.0
    tfe_per_sample = 3
    samples_per_horizon = 10
    ntfe = tfe_per_sample * samples_per_horizon
    sample_width = tfe_per_sample * tfe_width
    sample_points = [i*sample_width for i in range(samples_per_horizon+1)]
    horizon = samples_per_horizon * sample_width

    # Parameters pertaining to the simulation
    n_sim = 30
    #ntfe_sim = 10
    ntfe_sim = 4
    sim_horizon = n_sim * sample_width
    tfe_width_sim = sample_width / ntfe_sim

    #
    # Construct a dynamic model for simulation
    #
    htimer.start("construct_model")
    m_sim = make_model(
        dynamic=True,
        ntfe=ntfe_sim,
        tfe_width=tfe_width_sim,
    )
    htimer.stop("construct_model")
    htimer.start("DynamicModelInterface")
    m_sim_interface = mpc.DynamicModelInterface(m_sim, m_sim.fs.time)
    htimer.stop("DynamicModelInterface")
    ###

    #
    # Create a solver object we will use for simulation and optimization
    # solves
    #
    solver = pyo.SolverFactory("ipopt")
    solver.options["print_user_options"] = "yes"
    solver.options["tol"] = 1e-5
    solver.options["linear_solver"] = "ma57"
    ###

    #
    # Get data for initial steady state
    #
    htimer.start("construct_model")
    m_init = make_model()
    htimer.stop("construct_model")
    htimer.start("DynamicModelInterface")
    m_init_interface = mpc.DynamicModelInterface(m_init, m_init.fs.time)
    htimer.stop("DynamicModelInterface")
    # Specify inputs for initial steady state
    init_inputs = mpc.ScalarData({
        #"fs.moving_bed.gas_phase.properties[*,0.0].flow_mol": 128.2,
        #"fs.moving_bed.solid_phase.properties[*,1.0].flow_mass": 591.4,
    })
    m_init_interface.load_data(init_inputs)
    iscale.calculate_scaling_factors(m_init)

    htimer.start("solve_steady")
    initialize_steady(m_init)
    solver.solve(m_init, tee=True)
    htimer.stop("solve_steady")

    init_data = m_init_interface.get_data_at_time()
    scalar_data = m_init_interface.get_scalar_variable_data()
    with open("initial_conditions.json", "w") as f:
        json.dump(init_data.to_serializable(), f)
    ###

    #
    # Define a disturbance that will be applied.
    #
    disturbance_dict = {"CO2": 0.5, "H2O": 0.0, "CH4": 0.5}
    disturbance_name = "fs.moving_bed.gas_phase.properties[*,0.0].mole_frac_comp[%s]"
    disturbance = mpc.IntervalData(
        {
            disturbance_name % j: [val]
            for j, val in disturbance_dict.items()
        },
        [(0.0, sim_horizon)],
    )
    ###

    #
    # Get data for the target setpoint
    #
    htimer.start("construct_model")
    m_setpoint = make_model()
    htimer.stop("construct_model")
    htimer.start("DynamicModelInterface")
    setpoint_interface = mpc.DynamicModelInterface(
        m_setpoint, m_setpoint.fs.time
    )
    htimer.stop("DynamicModelInterface")
    setpoint_inputs = mpc.ScalarData({
        #"fs.moving_bed.gas_phase.properties[*,0.0].flow_mol": 128.2,
        #"fs.moving_bed.solid_phase.properties[*,1.0].flow_mass": 591.4,
    })
    setpoint_interface.load_data(setpoint_inputs)
    iscale.calculate_scaling_factors(m_setpoint)
    setpoint_interface.load_data(disturbance.get_data_at_time(sim_horizon))
    htimer.start("solve_steady")
    initialize_steady(m_setpoint)
    solver.solve(m_setpoint, tee=True)
    htimer.stop("solve_steady")
    sp_target = {"fs.moving_bed.solid_phase.reactions[*,0.0].OC_conv": 0.95}
    m_setpoint.fs.moving_bed.gas_inlet.flow_mol[:].unfix()
    (
        m_setpoint.penalty_set, m_setpoint.conv_penalty
    ) = setpoint_interface.get_penalty_from_target(sp_target)
    m_setpoint.obj = pyo.Objective(expr=sum(m_setpoint.conv_penalty.values()))
    htimer.start("solve_steady")
    solver.solve(m_setpoint, tee=True)
    htimer.stop("solve_steady")
    setpoint_data = setpoint_interface.get_data_at_time()
    with open("setpoint.json", "w") as f:
        json.dump(setpoint_data.to_serializable(), f)
    ###

    #
    # Make controller's optimization model
    #
    htimer.start("construct_model")
    m_opt = make_model(
        dynamic=True,
        ntfe=ntfe,
        tfe_width=tfe_width,
    )
    htimer.stop("construct_model")
    htimer.start("DynamicModelInterface")
    m_opt_interface = mpc.DynamicModelInterface(m_opt, m_opt.fs.time)
    htimer.stop("DynamicModelInterface")
    ###

    #
    # Add objective function and terminal constraint to optimization model
    #
    length_domain = m_opt.fs.moving_bed.length_domain
    mbr = m_opt.fs.moving_bed
    weight_data = {}
    setpoint_states = []
    gprop = m_opt.fs.moving_bed.gas_phase.properties
    sprop = m_opt.fs.moving_bed.solid_phase.properties
    for x in length_domain:
        weight_data[gprop[:, x].flow_mol] = 0.2
        weight_data[gprop[:, x].temperature] = 0.1
        weight_data[gprop[:, x].pressure] = 10.0*(1e-5)**2
        weight_data[gprop[:, x].mole_frac_comp["CH4"]] = 200.0
        weight_data[gprop[:, x].mole_frac_comp["CO2"]] = 200.0
        weight_data[gprop[:, x].mole_frac_comp["H2O"]] = 200.0

        weight_data[sprop[:, x].flow_mass] = 0.2
        weight_data[sprop[:, x].temperature] = 0.1
        weight_data[sprop[:, x].mass_frac_comp["Fe2O3"]] = 200.0
        weight_data[sprop[:, x].mass_frac_comp["Fe3O4"]] = 200.0
        weight_data[sprop[:, x].mass_frac_comp["Al2O3"]] = 200.0

        if x != length_domain.first():
            setpoint_states.append(gprop[:, x].flow_mol)
            setpoint_states.append(gprop[:, x].temperature)
            setpoint_states.append(gprop[:, x].pressure)
            setpoint_states.append(gprop[:, x].mole_frac_comp["CH4"])
            setpoint_states.append(gprop[:, x].mole_frac_comp["CO2"])
            setpoint_states.append(gprop[:, x].mole_frac_comp["H2O"])
        if x != length_domain.last():
            setpoint_states.append(sprop[:, x].flow_mass)
            setpoint_states.append(sprop[:, x].temperature)
            setpoint_states.append(sprop[:, x].mass_frac_comp["Fe2O3"])
            setpoint_states.append(sprop[:, x].mass_frac_comp["Fe3O4"])
            setpoint_states.append(sprop[:, x].mass_frac_comp["Al2O3"])

    (
        m_opt.tracking_var_set, m_opt.tracking_cost
    ) = m_opt_interface.get_penalty_from_target(
        setpoint_data,
        variables=setpoint_states,
        weight_data=weight_data,
    )

    @m_opt.Expression(m_opt.fs.time)
    def total_tracking_cost(m_opt, t):
        return sum(m_opt.tracking_cost[:, t])

    print("N. tracking variables:", len(m_opt.tracking_var_set))
    obj_time = m_opt.fs.time
    m_opt.tracking_obj = pyo.Objective(expr=sum(
        m_opt.tracking_cost[i, t]
        for i in m_opt.tracking_var_set for t in obj_time
        if t != m_opt.fs.time.first()
    ))
    # Add terminal penalty
    tf = m_opt.fs.time.last()
    m_opt.terminal_con = pyo.Constraint(
        expr=sum(m_opt.tracking_cost[:, tf]) <= 1e-1
    )
    ###

    (
        m_sim.tracking_var_set, m_sim.tracking_cost
    ) = m_sim_interface.get_penalty_from_target(
        setpoint_data,
        variables=setpoint_states,
        weight_data=weight_data,
    )
    @m_sim.Expression(m_sim.fs.time)
    def total_tracking_cost(m_sim, t):
        return sum(m_sim.tracking_cost[:, t])
    # Re-construct interface with tracking_cost present
    htimer.start("DynamicModelInterface")
    m_sim_interface = mpc.DynamicModelInterface(m_sim, m_sim.fs.time)
    htimer.stop("DynamicModelInterface")

    #
    # Unfix inputs and constrain them to be piecewise constant on sample periods
    #
    input_variables = [
        m_opt.fs.moving_bed.gas_inlet.flow_mol,
        m_opt.fs.moving_bed.solid_inlet.flow_mass,
    ]
    for var in input_variables:
        for t in m_opt.fs.time:
            if t != m_opt.fs.time.first():
                var[t].unfix()
    (
        m_opt.pwc_set, m_opt.pwc_con
    ) = m_opt_interface.get_piecewise_constant_constraints(
        input_variables,
        sample_points,
        tolerance=1e-8,
    )
    ###

    #
    # Initialize simulation and optimization models with initial steady state.
    # Note that this initializes variables at all time points.
    #
    m_sim_interface.load_data(init_data)
    m_sim_interface.load_data(scalar_data)
    m_opt_interface.load_data(init_data)
    m_opt_interface.load_data(scalar_data)
    ###

    #
    # Load feed-forward disturbance into dynamic optimization model
    #
    m_opt_interface.load_data(disturbance)
    m_sim_interface.load_data(disturbance)
    ###

    preprocess_dynamic_model(m_opt)
    preprocess_dynamic_model(m_sim)

    #
    # Set lower bound on oxygen carrier conversion
    #
    m_opt.fs.moving_bed.solid_phase.reactions[:, 0.0].OC_conv.setlb(0.89)
    ###

    # Re-construct m_opt_interface after tracking_cost has been defined.
    htimer.start("DynamicModelInterface")
    m_opt_interface = mpc.DynamicModelInterface(m_opt, m_opt.fs.time)
    htimer.stop("DynamicModelInterface")

    # TODO: Data structures to hold control inputs.
    # Is this necessary? Can I just get control inputs from the simulation
    # data? Why not...
    sim_data = m_sim_interface.get_data_at_time([0.0], include_expr=True)

    # Initialize data structure to hold control inputs
    control_inputs = m_opt_interface.get_data_at_time([0.0])
    control_inputs = control_inputs.extract_variables(input_variables)
    control_inputs = mpc.data.convert.series_to_interval(control_inputs)

    # Initialize data structure to hold tracking cost from controller
    tracking_cost_data = m_opt_interface.get_data_at_time([0.0], include_expr=True)
    tracking_cost_data = tracking_cost_data.extract_variables(
        [m_sim.tracking_cost[i, :] for i in m_sim.tracking_var_set]
    )
    # Note that this could be anything relating to a single optimization solve.
    # E.g. solve time, etc.
    tracking_cost_tf_data = mpc.data.TimeSeriesData(
        {
            "tracking_cost_tf": [0.0],
            "solve_time": [0.0],
        },
        [0.0],
    )

    non_initial_model_time = list(m_sim.fs.time)[1:]
    controller_sample = [
        t for t in m_opt.fs.time if t > m_opt.fs.time.first()
        and t <= sample_width + m_opt.fs.time.first()
    ]
    timer = TicTocTimer()
    for i in range(n_sim):
        print("***************************")
        print(f"BEGINNING NMPC ITERATION {i+1}")
        print("***************************")
        sim_t0 = i * sample_width
        sim_tf = (i + 1) * sample_width

        #
        # Solve controller optimization problem
        #
        htimer.start("solve_dynamic_optimization")
        timer.tic()
        from pyomo.util.subsystems import TemporarySubsystemManager
        to_deactivate = [m_opt.pwc_con, m_opt.terminal_con]
        input_vardata = [data for var in input_variables for data in var.values()]
        if i >= 2:
            with TemporarySubsystemManager(
                to_fix=input_vardata,
                to_deactivate=to_deactivate,
            ):
                m_opt.fs.moving_bed.solid_phase.reactions[:, 0.0].OC_conv.setlb(None)
                initialize_by_time_element(
                    m_opt, m_opt.fs.time, solver, solve_kwds={"tee": False}
                )
                m_opt.fs.moving_bed.solid_phase.reactions[:, 0.0].OC_conv.setlb(0.89)
        #from idaes.core.util.model_statistics import large_residuals_set
        #print("Large residuals")
        #for con in large_residuals_set(m_opt):
        #    resid = pyo.value(con.body-con.upper)
        #    print(f"  {resid}  {con.name}")
        try:
            res = solver.solve(m_opt, tee=True)
        except ValueError:
            # Pyomo raises ValueError if Ipopt converges with a restoration
            # failure, which happens commonly at primal/dual-feasible points
            # in NMPC.
            continue
        #from idaes.core.util.model_statistics import large_residuals_set
        #print("Large residuals")
        #for con in large_residuals_set(m_opt):
        #    resid = pyo.value(con.body-con.upper)
        #    print(f"  {resid}  {con.name}")
        #m_opt.fs.moving_bed.solid_phase.reactions[:, 0.0].OC_conv.pprint()
        #pyo.assert_optimal_termination(res)
        ###
        solve_time = timer.toc()
        htimer.stop("solve_dynamic_optimization")

        #
        # Extract inputs from optimization model
        #
        input_data = m_opt_interface.get_data_at_time(
            sample_width + m_opt.fs.time.first()
        )
        input_data = input_data.extract_variables(input_variables)
        ###

        # Load inputs
        #sim_time = [sim_t0 + t - m_sim.fs.time.first() for t in m_sim.fs.time]
        #local_inputs = mpc.data.convert.interval_to_series(
        #    input_sequence, time_points=sim_time
        #)
        #local_inputs.shift_time_points(m_sim.fs.time.first() - sim_t0)
        m_sim_interface.load_data(input_data, tolerance=1e-6)

        # Solve simulation model
        htimer.start("simulate_plant")
        initialize_by_time_element(
            m_sim, m_sim.fs.time, solver, solve_kwds={"tee": False}
        )
        res = solver.solve(m_sim, tee=True)
        htimer.stop("simulate_plant")
        pyo.assert_optimal_termination(res)

        #
        # Extend series of control inputs
        #
        # Get SeriesData for control inputs
        new_control_inputs = m_opt_interface.get_data_at_time(
            [m_opt.fs.time.first(), m_opt.fs.time.first() + sample_width]
        )
        new_control_inputs = mpc.data.convert.series_to_interval(
            new_control_inputs
        )
        new_control_inputs.shift_time_points(sim_t0 - m_opt.fs.time.first())
        control_inputs.concatenate(new_control_inputs)
        ###

        #
        # Extend tracking cost series
        #
        new_tracking_cost_data = m_sim_interface.get_data_at_time(
            non_initial_model_time, include_expr=True
        )
        new_tracking_cost_data = new_tracking_cost_data.extract_variables(
            [m_sim.tracking_cost[i, :] for i in m_sim.tracking_var_set]
        )
        new_tracking_cost_data.shift_time_points(sim_t0 - m_sim.fs.time.first())
        tracking_cost_data.concatenate(new_tracking_cost_data)
        tracking_cost_tf = sum(
            pyo.value(m_opt.tracking_cost[:, m_opt.fs.time.last()])
        )
        new_tracking_cost_tf_data = mpc.data.TimeSeriesData(
            {
                "tracking_cost_tf": [tracking_cost_tf],
                "solve_time": [solve_time],
            },
            [sim_tf],
        )
        tracking_cost_tf_data.concatenate(new_tracking_cost_tf_data)
        ###

        # Extract data to serializable data structure
        model_data = m_sim_interface.get_data_at_time(
            non_initial_model_time, include_expr=True
        )
        model_data.shift_time_points(sim_t0 - m_sim.fs.time.first())
        sim_data.concatenate(model_data)

        # Cycle initial conditions in simulation model
        tf_data = m_sim_interface.get_data_at_time(m_sim.fs.time.last())
        m_sim_interface.load_data(tf_data)

        # Cycle controller model (initialize to the previous solve)
        m_opt_interface.shift_values_by_time(sample_width)
        # Both of the following branches set the initial conditions to the plant
        # variable values.
        if i <= 10:
            # Initialize to result of previous solve (don't override anything
            # other than initial condition)
            m_opt_interface.load_data(tf_data, time_points=m_opt.fs.time.first())
        else:
            # Initialize to initial condition
            m_opt_interface.load_data(tf_data)

    with open("plant_data.json", "w") as f:
        json.dump(sim_data.to_serializable(), f)
    with open("control_inputs.json", "w") as f:
        json.dump(control_inputs.to_serializable(), f)
    with open("tracking_cost.json", "w") as f:
        json.dump(tracking_cost_data.to_serializable(), f)
    with open("aux_data.json", "w") as f:
        json.dump(tracking_cost_tf_data.to_serializable(), f)

    htimer.stop("root")
    print(htimer)


if __name__ == "__main__":
    main()
