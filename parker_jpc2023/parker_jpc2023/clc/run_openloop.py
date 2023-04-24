import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from idaes.core.util import scaling as iscale
from parker_jpc2023.clc.initialize import initialize_steady
from parker_jpc2023.clc.model import make_model
from parker_jpc2023.clc.plot import (
    _plot_time_indexed_variables,
    _step_time_indexed_variables,
)


def run_openloop(n_sim=10, ntfe=10):
    tfe_width = 12.0
    each_sim_horizon = ntfe * tfe_width

    m = make_model(
        dynamic=True,
        ntfe=ntfe,
        tfe_width=tfe_width,
    )
    m_interface = mpc.DynamicModelInterface(m, m.fs.time)
    m_init = make_model()
    m_init_interface = mpc.DynamicModelInterface(m_init, m_init.fs.time)

    # Specify inputs for initial steady state
    init_inputs = mpc.ScalarData({
        "fs.moving_bed.gas_phase.properties[*,0.0].flow_mol": 120.0,
        "fs.moving_bed.solid_phase.properties[*,1.0].flow_mass": 550.0,
    })
    m_init_interface.load_data(init_inputs)
    iscale.calculate_scaling_factors(m_init)

    initialize_steady(m_init)

    solver = pyo.SolverFactory("ipopt")
    solver.options["print_user_options"] = "yes"
    # user-scaling might help to some extent
    # Actually makes things worse in dynamic simulation
    #solver.options["nlp_scaling_method"] = "user-scaling"

    # This is necessary to "converge" several solid phase equations:
    # enthalpy balances, mass balances, reaction rate equations.
    solver.options["tol"] = 1e-5
    solver.solve(m_init, tee=True)

    init_data = m_init_interface.get_data_at_time()
    scalar_data = m_init_interface.get_scalar_variable_data()
    m_interface.load_data(init_data)
    m_interface.load_data(scalar_data)

    input_sequence = mpc.IntervalData(
        {
            "fs.moving_bed.gas_phase.properties[*,0.0].flow_mol": [
                120.0 + i*10.0 for i in range(n_sim)
            ],
            "fs.moving_bed.solid_phase.properties[*,1.0].flow_mass": [
                550.0 + 10.0*i for i in range(n_sim)
            ],
        },
        [(i*each_sim_horizon, (i+1)*each_sim_horizon) for i in range(n_sim)],
    )

    sim_data = m_interface.get_data_at_time([0.0])

    non_initial_model_time = list(m.fs.time)[1:]
    for i in range(n_sim):
        sim_t0 = i*each_sim_horizon
        sim_tf = (i+1)*each_sim_horizon

        # Load inputs
        sim_time = [sim_t0 + t - m.fs.time.first() for t in m.fs.time]
        local_inputs = mpc.data.convert.interval_to_series(
            input_sequence, time_points=sim_time
        )
        local_inputs.shift_time_points(m.fs.time.first() - sim_t0)
        m_interface.load_data(local_inputs, tolerance=1e-6)

        # Solve
        res = solver.solve(m, tee=True)
        pyo.assert_optimal_termination(res)

        # Extract data to serializable data structure
        model_data = m_interface.get_data_at_time(non_initial_model_time)
        model_data.shift_time_points(sim_t0 - m.fs.time.first())
        sim_data.concatenate(model_data)

        # Cycle initial conditions
        tf_data = m_interface.get_data_at_time(m.fs.time.last())
        m_interface.load_data(tf_data)

    return sim_data


def main():
    sim_data = run_openloop()

    outlet_temperatures = [
        #"fs.moving_bed.gas_outlet.temperature[*]",
        #"fs.moving_bed.solid_outlet.temperature[*]",
        "fs.moving_bed.gas_phase.properties[*,1.0].temperature",
        "fs.moving_bed.solid_phase.properties[*,0.0].temperature",
    ]
    _plot_time_indexed_variables(sim_data, outlet_temperatures, show=True)

    outlet_compositions = [
        "fs.moving_bed.gas_phase.properties[*,1.0].mole_frac_comp[CO2]",
        "fs.moving_bed.solid_phase.properties[*,0.0].mass_frac_comp[Fe2O3]",
    ]
    _plot_time_indexed_variables(sim_data, outlet_compositions, show=True)


if __name__ == "__main__":
    main()
