import json
import matplotlib.pyplot as plt

PLANT_FNAME = "nmpc_plant_data.json"
IC_FNAME = "ic_data.json"
SETPOINT_FNAME = "setpoint_data.json"

# Here, "state" is just anything we plot as continuous-time,
# as opposed to an input, which is piecewise-constant.
# NOTE: These names depend on our knowledge of the boundary points.
STATE_NAMES = [
    "fs.MB.gas_phase.properties[*,1.0].flow_mol",
    "fs.MB.gas_phase.properties[*,1.0].temperature",
    "fs.MB.gas_phase.properties[*,1.0].pressure",
    "fs.MB.gas_phase.properties[*,1.0].mole_frac_comp[CH4]",
    "fs.MB.gas_phase.properties[*,1.0].mole_frac_comp[CO2]",
    "fs.MB.gas_phase.properties[*,1.0].mole_frac_comp[H2O]",
    "fs.MB.solid_phase.properties[*,0.0].flow_mass",
    "fs.MB.solid_phase.properties[*,0.0].temperature",
    "fs.MB.solid_phase.properties[*,0.0].mass_frac_comp[Fe2O3]",
    "fs.MB.solid_phase.properties[*,0.0].mass_frac_comp[Fe3O4]",
    "fs.MB.solid_phase.properties[*,0.0].mass_frac_comp[Al2O3]",
    "fs.MB.solid_phase.reactions[*,0.0].OC_conv",
    "tracking_cost",
]

STATE_LABELS = [
    "Gas",
    "Gas",
    "Pressure",
    "CH4",
    "CO2",
    "H2O",
    "Solid",
    "Solid",
    "Fe2O3",
    "Fe3O4",
    "Al2O3",
    "Conversion",
]

INPUT_NAMES = [
    "fs.MB.gas_phase.properties[*,0.0].flow_mol",
    "fs.MB.solid_phase.properties[*,1.0].flow_mass",
]

DELTA_T = 60.0
PREPEND_STATES = 10
PREPEND_INPUTS = 10


def main():
    with open(PLANT_FNAME, "r") as fp:
        plant_data = json.load(fp)
    with open(IC_FNAME, "r") as fp:
        ic_data = json.load(fp)
    with open(SETPOINT_FNAME, "r") as fp:
        setpoint_data = json.load(fp)

    time, plant_data = plant_data
    n_time = len(time)
    # time: [list of time points]
    # plant_data:
    # {str(cuid): [list of values]}
    state_trajectories = {name: plant_data[name] for name in STATE_NAMES}
    state_setpoints = {
        # Dont have a setpoint for tracking_cost... But we know target is zero
        name: [setpoint_data[name]]*n_time
        if name in setpoint_data else [0.0]*n_time for name in STATE_NAMES
    }
    input_trajectories = {name: plant_data[name] for name in INPUT_NAMES}
    input_setpoints = {name: [setpoint_data[name]]*n_time for name in INPUT_NAMES}

    state_ic = {
        name: ic_data[name]
        if name != "tracking_cost" else 0.0
        for name in STATE_NAMES
    }
    input_ic = {name: ic_data[name] for name in INPUT_NAMES}
    for name in STATE_NAMES:
        state_trajectories[name] = (
            [state_ic[name]]*PREPEND_STATES
            + state_trajectories[name]
        )
        state_setpoints[name] = (
            [state_ic[name]]*PREPEND_STATES
            + state_setpoints[name]
        )
    for name in INPUT_NAMES:
        input_trajectories[name] = (
            [input_ic[name]]*PREPEND_INPUTS
            + input_trajectories[name]
        )
        input_setpoints[name] = (
            [input_ic[name]]*PREPEND_INPUTS
            + input_setpoints[name]
        )

    t_end = time[-1]
    time.extend([t_end + i*DELTA_T for i in range(1, PREPEND_STATES+1)])

    name_label_map = dict(zip(STATE_NAMES, STATE_LABELS))

    # Plot state trajectories
    state_subsets = [
        # These state subsets are commented because they are not
        # that interesting, and it is convenient to only show
        # four state trajectory plots.
        #[
        #    "fs.MB.gas_phase.properties[*,1.0].flow_mol",
        #    "fs.MB.solid_phase.properties[*,0.0].flow_mass",
        #],
        #[
        #    "fs.MB.gas_phase.properties[*,1.0].temperature",
        #    "fs.MB.solid_phase.properties[*,0.0].temperature",
        #],
        [
            "fs.MB.gas_phase.properties[*,1.0].mole_frac_comp[CH4]",
            "fs.MB.gas_phase.properties[*,1.0].mole_frac_comp[CO2]",
            "fs.MB.gas_phase.properties[*,1.0].mole_frac_comp[H2O]",
            "fs.MB.solid_phase.properties[*,0.0].mass_frac_comp[Fe2O3]",
            "fs.MB.solid_phase.properties[*,0.0].mass_frac_comp[Fe3O4]",
            "fs.MB.solid_phase.properties[*,0.0].mass_frac_comp[Al2O3]",
        ],
        [
            "fs.MB.gas_phase.properties[*,1.0].pressure",
        ],
        [
            "fs.MB.solid_phase.reactions[*,0.0].OC_conv",
        ],
        [
            "tracking_cost",
        ],
    ]

    subset_titles = [
        "Outlet component fractions",
        "Outlet pressure",
        "Outlet solid conversion",
        "Overall tracking cost",
    ]

    subset_ylabels = [
        "Fractional composition",
        "Pressure (bar)",
        "Fractional conversion",
        "Weighted tracking cost",
    ]

    plt.rcParams.update({"font.size": 16})
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    marker_list = ["o", "v", "^", "s", "D", "P", "*"]

    # Plot state trajectories
    for i, subset in enumerate(state_subsets):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(subset_ylabels[i])
        # TODO: Set y label (units vary by subset)
        for j, name in enumerate(subset):
            label = name_label_map[name] if i == 0 else None
            ax.plot(
                time,
                state_trajectories[name],
                linewidth=2,
                label=label,
                color=color_list[j],
                marker=marker_list[j],
                markevery=3,
                markersize=5,
            )
            ax.step(
                time,
                state_setpoints[name],
                linewidth=1,
                linestyle="--",
                color=color_list[j],
            )
            if i == 0:
                ax.legend()
        ax.set_title(subset_titles[i])
        fig.tight_layout()
        fig.savefig("nmpc_state%s.png" % i, transparent=True)

    # Plot input piecewise-constant trajectories
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    ax_labels = ["Gas flow rate (mol/s)", "Solid flow rate (kg/s)"]
    ylims = [(120, 450), (480, 600)]
    ax1.set_xlabel("Time (s)")
    line_labels = ["Gas flow rate", "Solid flow rate"]

    color_list = ["blue", "orange"]
    
    lines = []
    for i, name in enumerate(INPUT_NAMES):
        inputs = input_trajectories[name]
        setpoint = input_setpoints[name]
        # TODO: Label
        ax = axes[i]
        ax.set_ylim(ylims[i])
        ax.set_ylabel(ax_labels[i])
        label = line_labels[i]
        line = ax.step(
            time,
            inputs,
            #marker=marker_list[i],
            linewidth=2,
            #markersize=10,
            color=color_list[i],
            label=label,
        )
        lines.extend(line)
        ax.step(
            time,
            setpoint,
            linewidth=1,
            linestyle="--",
            color=color_list[i],
        )
    #fig.legend(loc="upper left")
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")
    fig.tight_layout()
    fig.savefig("nmpc_inputs.png", transparent=True)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
