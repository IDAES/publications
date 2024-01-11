import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import create_instance

# Create piecewise-constant input sequence, model, and DynamicModelInterface
input_sequence = mpc.data.convert.series_to_interval(mpc.TimeSeriesData(
    {"flow_in[*]": [0.1, 1.0, 0.5, 1.3, 1.0, 0.3]},
    [0.0, 2.0, 4.0, 6.0, 8.0, 15.0],
))
model_horizon = 1.0
m = create_instance(horizon=model_horizon, ntfe=10)
dynamic_interface = mpc.DynamicModelInterface(m, m.time)

# Initialize data structure to hold rolling horizon simulation results
sim_data = dynamic_interface.get_data_at_time([0.0])

solver = pyo.SolverFactory("ipopt")
non_initial_model_time = list(m.time)[1:]

for i in range(10):
    sim_t0 = i * model_horizon
    sim_time = [sim_t0 + t for t in m.time]

    # Load inputs into model
    new_inputs = mpc.data.convert.interval_to_series(
        input_sequence, time_points=sim_time
    )
    new_inputs.shift_time_points(m.time.first() - sim_t0)
    dynamic_interface.load_data(new_inputs, tolerance=1e-6)

    # Simulate model
    res = solver.solve(m, tee=True)

    # Extract data from solved model
    m_data = dynamic_interface.get_data_at_time(non_initial_model_time)
    m_data.shift_time_points(sim_t0 - m.time.first())
    sim_data.concatenate(m_data)

    # Re-initialize model (including initial conditions) to values at final time
    tf_data = dynamic_interface.get_data_at_time(m.time.last())
    dynamic_interface.load_data(tf_data)
