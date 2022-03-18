import json
import itertools
import matplotlib.pyplot as plt

VARIABLE_NAMES = [
    "fs.MB.gas_phase.properties[*,*].flow_mol",
    "fs.MB.gas_phase.properties[*,*].temperature",
    "fs.MB.gas_phase.properties[*,*].pressure",
    "fs.MB.gas_phase.properties[*,*].mole_frac_comp[CH4]",
    "fs.MB.gas_phase.properties[*,*].mole_frac_comp[CO2]",
    "fs.MB.gas_phase.properties[*,*].mole_frac_comp[H2O]",
    "fs.MB.solid_phase.properties[*,*].flow_mass",
    "fs.MB.solid_phase.properties[*,*].temperature",
    "fs.MB.solid_phase.properties[*,*].mass_frac_comp[Fe2O3]",
    "fs.MB.solid_phase.properties[*,*].mass_frac_comp[Fe3O4]",
    "fs.MB.solid_phase.properties[*,*].mass_frac_comp[Al2O3]",
]

# Labels for plotting
VARIABLE_LABELS = [
    "Gas flow",
    "Gas temp.",
    "Pressure",
    "CH4 frac.",
    "CO2 frac.",
    "H2O frac.",
    "Solid flow",
    "Solid temp.",
    "Fe2O3 frac.",
    "Fe3O4 frac.",
    "Al2O3 frac.",
]

FILES_DISC_MAP = {
    # These are the files for my original, simple simulation
    #"300s_error_2_4_8_16_32.json": [2, 4, 8, 16],
    #"300s_error_32_64.json": [32],
    #"300s_error_64_128.json": [64],

    # These are the files for the more complicated simulation with
    # perturbed inputs.
    "600s_inputs_2_4_8_16_32_64.json": [2, 4, 8, 16, 32],
    "600s_inputs_64_128.json": [64],
}

N_DISCRETIZATION_POINTS = [2, 4, 8, 16, 32, 64]


def main():
    variable_errors = {name: {} for name in VARIABLE_NAMES}
    for fname, disc_list in FILES_DISC_MAP.items():
        with open(fname, "r") as fp:
            data = json.load(fp)
        for varname in VARIABLE_NAMES:
            for nfe, error in zip(disc_list, data[varname]):
                variable_errors[varname][nfe] = error

    variable_errors = {
        name: [error[i] for i in N_DISCRETIZATION_POINTS]
        for name, error in variable_errors.items()
    }

    name_label_map = dict(zip(VARIABLE_NAMES, VARIABLE_LABELS))
    disc_list = N_DISCRETIZATION_POINTS

    # Subsets of states to plot on same axes
    plot_subsets = [
        [
            "fs.MB.gas_phase.properties[*,*].flow_mol",
            "fs.MB.solid_phase.properties[*,*].flow_mass",
        ],
        [
            "fs.MB.gas_phase.properties[*,*].temperature",
            "fs.MB.solid_phase.properties[*,*].temperature",
        ],
        [
            "fs.MB.gas_phase.properties[*,*].pressure",
        ],
        [
            "fs.MB.gas_phase.properties[*,*].mole_frac_comp[CH4]",
            "fs.MB.gas_phase.properties[*,*].mole_frac_comp[CO2]",
            "fs.MB.gas_phase.properties[*,*].mole_frac_comp[H2O]",
            "fs.MB.solid_phase.properties[*,*].mass_frac_comp[Fe2O3]",
            "fs.MB.solid_phase.properties[*,*].mass_frac_comp[Fe3O4]",
            "fs.MB.solid_phase.properties[*,*].mass_frac_comp[Al2O3]",
        ],
    ]

    color_list = [(0.0 + i*0.05,)*3 for i in range(1, 8)]
    marker_list = ["o", "v", "^", "s", "D", "P", "*"]
    #color_marker_list = list(itertools.product(color_list, marker_list))
    color_marker_list = list(zip(color_list, marker_list))
    plt.rcParams.update({"font.size": 16})

    for i, subset in enumerate(plot_subsets):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel("Discretization points")
        ax.set_ylabel("Integrated error")
        # Error decreases approximately linearly on a log-log scale,
        # but axis labeling is nicer on a linear scale because the data
        # doesn't vary over many orders of magnitude.
        #ax.set_yscale("log")
        #ax.set_xscale("log")
        for j, name in enumerate(subset):
            label = name_label_map[name]
            color, marker = color_marker_list[j]
            ax.plot(
                disc_list,
                variable_errors[name],
                label=label,
                linewidth=2,
                # Do my figures need to be greyscale?
                #color=color,
                marker=marker,
                markersize=10,
            )
        ax.legend()
        fig.tight_layout()
        fig.savefig("disc_error_%s.png" % i, transparent=True)


if __name__ == "__main__":
    main()
