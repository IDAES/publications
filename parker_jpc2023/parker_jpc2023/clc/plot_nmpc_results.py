import json
import matplotlib.pyplot as plt
import pyomo.contrib.mpc as mpc


CONTROLLER_START_TIME = 720.0
POINTS_PER_SAMPLE = 4
EXTENSION = ".pdf"


plt.rcParams["font.size"] = 16


def get_plant_series_to_plot():
    with open("initial_conditions.json", "r") as f:
        init_data = json.load(f)
    with open("setpoint.json", "r") as f:
        setpoint_data = json.load(f)
    with open("plant_data.json", "r") as f:
        plant_data = json.load(f)

    init_data = mpc.ScalarData(init_data)
    setpoint_data = mpc.ScalarData(setpoint_data)
    plant_data = mpc.TimeSeriesData(*plant_data)

    mpc_time = plant_data.get_time_points()
    dt = mpc_time[1] - mpc_time[0]
    n_pre_mpc_points = round(CONTROLLER_START_TIME/dt)
    pre_mpc_points = [dt*i for i in range(n_pre_mpc_points)]

    # Want to define for all keys
    pre_mpc_data = mpc.TimeSeriesData(
        {
            name: [value for t in pre_mpc_points]
            for name, value in init_data.get_data().items()
        },
        pre_mpc_points,
    )
    pre_mpc_tracking_cost = mpc.TimeSeriesData(
        {"total_tracking_cost[*]": [0.0 for t in pre_mpc_points]}, pre_mpc_points
    )
    tracking_cost = plant_data.extract_variables(["total_tracking_cost[*]"])
    tracking_cost.shift_time_points(CONTROLLER_START_TIME)
    pre_mpc_tracking_cost.concatenate(tracking_cost)
    tracking_cost = pre_mpc_tracking_cost
    # Need to extract only the variables that exist at steady state
    # (i.e. no derivatives)
    plant_data = plant_data.extract_variables(
        list(init_data.get_data().keys())
    )
    plant_data.shift_time_points(CONTROLLER_START_TIME)
    pre_mpc_data.concatenate(plant_data)
    plant_data = pre_mpc_data

    time_points = plant_data.get_time_points()

    setpoint_series = mpc.TimeSeriesData(
        {
            name: [
                init_data.get_data_from_key(name) if t < CONTROLLER_START_TIME
                else setpoint_data.get_data_from_key(name)
                for t in time_points
            ]
            for name in init_data.get_data().keys()
        },
        time_points,
    )

    return plant_data, setpoint_series, tracking_cost


def get_control_inputs_to_plot():
    with open("initial_conditions.json", "r") as f:
        init_data = json.load(f)
    with open("setpoint.json", "r") as f:
        setpoint_data = json.load(f)
    with open("control_inputs.json", "r") as f:
        control_inputs = json.load(f)

    init_data = mpc.ScalarData(init_data)
    setpoint_data = mpc.ScalarData(setpoint_data)
    control_inputs = mpc.IntervalData(*control_inputs)
    control_inputs = mpc.data.convert.interval_to_series(control_inputs)

    sample_points = control_inputs.get_time_points()
    dt = sample_points[1] - sample_points[0]
    n_pre_mpc_points = round(CONTROLLER_START_TIME/dt)
    pre_mpc_points = [dt*i for i in range(n_pre_mpc_points)]

    # Want to define for all keys
    pre_mpc_data = mpc.TimeSeriesData(
        {
            name: [init_data.get_data_from_key(name) for t in pre_mpc_points]
            for name in control_inputs.get_data().keys()
        },
        pre_mpc_points,
    )
    # Need to extract only the variables that exist at steady state
    # (i.e. no derivatives)
    control_inputs.shift_time_points(CONTROLLER_START_TIME)
    pre_mpc_data.concatenate(control_inputs)
    control_inputs = pre_mpc_data

    time_points = control_inputs.get_time_points()

    setpoint_series = mpc.TimeSeriesData(
        {
            name: [
                init_data.get_data_from_key(name) if t < CONTROLLER_START_TIME
                else setpoint_data.get_data_from_key(name)
                for t in time_points
            ]
            for name in control_inputs.get_data().keys()
        },
        time_points,
    )

    return control_inputs, setpoint_series


def plot_component_fractions(plant_data, setpoint_series):
    time_points = plant_data.get_time_points()
    state_names = [
        "fs.moving_bed.gas_phase.properties[*,1.0].mole_frac_comp[CO2]",
        "fs.moving_bed.gas_phase.properties[*,1.0].mole_frac_comp[CH4]",
        "fs.moving_bed.gas_phase.properties[*,1.0].mole_frac_comp[H2O]",
        "fs.moving_bed.solid_phase.properties[*,0.0].mass_frac_comp[Fe2O3]",
        "fs.moving_bed.solid_phase.properties[*,0.0].mass_frac_comp[Fe3O4]",
        "fs.moving_bed.solid_phase.properties[*,0.0].mass_frac_comp[Al2O3]",
    ]
    state_labels = ["CO2", "CH4", "H2O", "Fe2O3", "Fe3O4", "Al2O3"]
    state_colors = [
        "tab:blue", "tab:green", "tab:orange",
        "tab:red", "tab:purple", "tab:brown",
    ]
    fig, ax = plt.subplots()
    for i, name in enumerate(state_names):
        label = state_labels[i]
        color = state_colors[i]
        ax.step(
            time_points,
            setpoint_series.get_data_from_key(name),
            where="post",
            linestyle="--",
            color=color,
        )

        ax.plot(
            time_points,
            plant_data.get_data_from_key(name),
            color=color,
            marker=".",
            markevery=POINTS_PER_SAMPLE,
            label=label,
        )

    ax.set_title("Outlet component fractions")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fractional composition")
    ax.legend()

    fig.savefig("component_fractions" + EXTENSION)


def plot_outlet_pressure(plant_data, setpoint_series):
    time_points = plant_data.get_time_points()
    state_names = ["fs.moving_bed.gas_phase.properties[*,1.0].pressure"]
    state_colors = [
        "tab:blue", "tab:green", "tab:orange",
        "tab:red", "tab:purple", "tab:brown",
    ]
    fig, ax = plt.subplots()
    for i, name in enumerate(state_names):
        color = state_colors[i]
        ax.step(
            time_points,
            # Scale pressure to have units of bar
            [val/1e5 for val in setpoint_series.get_data_from_key(name)],
            where="post",
            linestyle="--",
            color=color,
        )

        ax.plot(
            time_points,
            # Scale pressure to have units of bar
            [val/1e5 for val in plant_data.get_data_from_key(name)],
            marker=".",
            markevery=POINTS_PER_SAMPLE,
            color=color,
        )

    ax.set_title("Outlet pressure")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (bar)")
    fig.tight_layout()

    fig.savefig("pressure" + EXTENSION)


def plot_outlet_conversion(plant_data, setpoint_series):
    time_points = plant_data.get_time_points()
    state_names = ["fs.moving_bed.solid_phase.reactions[*,0.0].OC_conv"]
    state_colors = [
        "tab:blue", "tab:green", "tab:orange",
        "tab:red", "tab:purple", "tab:brown",
    ]
    fig, ax = plt.subplots()
    for i, name in enumerate(state_names):
        color = state_colors[i]
        ax.step(
            time_points,
            setpoint_series.get_data_from_key(name),
            where="post",
            linestyle="--",
            color=color,
        )

        ax.plot(
            time_points,
            plant_data.get_data_from_key(name),
            marker=".",
            markevery=POINTS_PER_SAMPLE,
            color=color,
        )

    ax.set_title("Outlet solid conversion")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fractional conversion")
    fig.tight_layout()

    fig.savefig("conversion" + EXTENSION)


def plot_outlet_temperatures(plant_data, setpoint_series):
    # TODO: If I want to include this in paper, I need to plot with
    # broken y-axis.
    time_points = plant_data.get_time_points()
    state_names = [
        "fs.moving_bed.gas_phase.properties[*,1.0].temperature",
        "fs.moving_bed.solid_phase.properties[*,0.0].temperature",
    ]
    state_labels = ["Gas temperature", "Solid temperature"]
    state_colors = ["tab:blue", "tab:orange"]
    fig, ax = plt.subplots()
    for i, name in enumerate(state_names):
        color = state_colors[i]
        label = state_labels[i]
        ax.step(
            time_points,
            setpoint_series.get_data_from_key(name),
            where="post",
            linestyle="--",
            color=color,
        )

        ax.plot(
            time_points,
            plant_data.get_data_from_key(name),
            marker=".",
            markevery=POINTS_PER_SAMPLE,
            color=color,
            label=label,
        )

    ax.set_title("Outlet temperatures")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    fig.tight_layout()

    fig.savefig("temperature" + EXTENSION)


def plot_outlet_flow_rates(plant_data, setpoint_series):
    time_points = plant_data.get_time_points()
    state_names = [
        "fs.moving_bed.gas_phase.properties[*,1.0].flow_mol",
        "fs.moving_bed.solid_phase.properties[*,0.0].flow_mass",
    ]
    state_labels = ["Gas flow rate", "Solid flow rate"]
    state_colors = ["tab:blue", "tab:orange"]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    for i, name in enumerate(state_names):
        ax = axes[i]
        color = state_colors[i]
        label = state_labels[i]
        ax.step(
            time_points,
            setpoint_series.get_data_from_key(name),
            where="post",
            linestyle="--",
            color=color,
        )

        ax.plot(
            time_points,
            plant_data.get_data_from_key(name),
            marker=".",
            markevery=POINTS_PER_SAMPLE,
            color=color,
            label=label,
        )

    ax1.set_title("Outlet flow rates")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Gas flow rate (mol/s)")
    ax2.set_ylabel("Solid flow rate (kg/s)")
    ax1.set_ylim((350, 750))
    ax2.set_ylim((470, 590))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2)
    fig.tight_layout()

    fig.savefig("outlet_flow" + EXTENSION)


def plot_inlet_flow_rates(control_inputs, setpoint_series):
    time_points = control_inputs.get_time_points()
    state_names = [
        "fs.moving_bed._flow_mol_gas_inlet_ref[*]",
        "fs.moving_bed._flow_mass_solid_inlet_ref[*]",
        #"fs.moving_bed.gas_phase.properties[*,0.0].flow_mol",
        #"fs.moving_bed.solid_phase.properties[*,1.0].flow_mass",
    ]
    state_labels = ["Gas flow rate", "Solid flow rate"]
    state_colors = ["tab:blue", "tab:orange"]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    for i, name in enumerate(state_names):
        ax = axes[i]
        color = state_colors[i]
        label = state_labels[i]
        ax.step(
            time_points,
            setpoint_series.get_data_from_key(name),
            where="post",
            linestyle="--",
            color=color,
        )

        line = ax.step(
            time_points,
            control_inputs.get_data_from_key(name),
            where="pre",
            #marker=".",
            color=color,
            label=label,
        )

    ax1.set_title("Inlet flow rates")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Gas flow rate (mol/s)")
    ax2.set_ylabel("Solid flow rate (kg/s)")
    ax1.set_ylim((125, 450))
    ax2.set_ylim((480, 600))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2)

    fig.tight_layout()

    fig.savefig("inlet_flow" + EXTENSION)


def plot_tracking_cost(tracking_cost):
    time_points = tracking_cost.get_time_points()
    setpoint_series = mpc.TimeSeriesData(
        {"total_tracking_cost[*]": [0.0 for t in time_points]}, time_points
    )
    state_names = ["total_tracking_cost[*]"]
    state_colors = [
        "tab:blue", "tab:green", "tab:orange",
        "tab:red", "tab:purple", "tab:brown",
    ]
    fig, ax = plt.subplots()
    for i, name in enumerate(state_names):
        color = state_colors[i]
        ax.step(
            time_points,
            setpoint_series.get_data_from_key(name),
            where="post",
            linestyle="--",
            color=color,
        )

        ax.plot(
            time_points,
            tracking_cost.get_data_from_key(name),
            marker=".",
            markevery=POINTS_PER_SAMPLE,
            color=color,
        )

    ax.set_title("Total tracking cost")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking cost (dimensionless)")
    fig.tight_layout()

    fig.savefig("tracking_cost" + EXTENSION)


def plot_tracking_cost_tf():
    with open("aux_data.json", "r") as f:
        tracking_cost_tf = json.load(f)
    tracking_cost_tf = tracking_cost_tf[0]["tracking_cost_tf"][1:]
    nmpc_iterations = list(range(1, len(tracking_cost_tf)+1))

    fig, ax = plt.subplots()
    ax.plot(nmpc_iterations, tracking_cost_tf, marker=".")
    ax.set_title("Predicted terminal tracking cost")
    ax.set_xlabel("NMPC iteration")
    ax.set_ylabel("Tracking cost (dimensionless)")
    ax.set_ylim((0.0, 0.12))
    fig.tight_layout()

    fig.savefig("tracking_cost_tf" + EXTENSION)


def plot_solve_times():
    with open("aux_data.json", "r") as f:
        solve_times = json.load(f)
    solve_times = solve_times[0]["solve_time"][1:]
    nmpc_iterations = list(range(1, len(solve_times)+1))

    fig, ax = plt.subplots()
    ax.plot(nmpc_iterations, solve_times, marker=".")
    ax.set_title("NLP solve times")
    ax.set_xlabel("NMPC iteration")
    ax.set_ylabel("Solve time (s)")
    ax.set_ylim((0.0, 60.0))
    fig.tight_layout()

    fig.savefig("solve_time" + EXTENSION)


def main():
    plant_data, setpoint_series, tracking_cost = get_plant_series_to_plot()
    plot_component_fractions(plant_data, setpoint_series)
    plot_outlet_pressure(plant_data, setpoint_series)
    plot_outlet_conversion(plant_data, setpoint_series)
    plot_outlet_temperatures(plant_data, setpoint_series)
    plot_outlet_flow_rates(plant_data, setpoint_series)
    plot_tracking_cost(tracking_cost)

    control_inputs, setpoint_series = get_control_inputs_to_plot()
    plot_inlet_flow_rates(control_inputs, setpoint_series)

    plot_tracking_cost_tf()
    plot_solve_times()


if __name__ == "__main__":
    main()
