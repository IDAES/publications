import itertools
import matplotlib.pyplot as plt
"""
This script plots a piecewise constant sequence of inputs used in the
simulation for which we validate stability.
"""

INPUT_NAMES = [
    "fs.MB.gas_inlet.flow_mol[*]",
    "fs.MB.solid_inlet.flow_mass[*]",
]

# Legend labels for plotting
INPUT_LABELS = [
    "Gas flow rate",
    "Solid flow rate",
]

INPUT_VALUES = [
    [100.0, 128.2, 200.0],
    [500.0, 591.4, 700.0],
]

PREPEND_INPUTS = [
    # for 0 and 60 s
    [128.2, 128.2],
    [591.4, 591.4],
]

TIME_INTERVAL = 60.0


def main():
    input_sequence = {name: [] for name in INPUT_NAMES}

    for name, inputs in zip(INPUT_NAMES, PREPEND_INPUTS):
        input_sequence[name].extend(inputs)

    product_list = list(itertools.product(*INPUT_VALUES))
    for inputs in product_list:
        for name, value in zip(INPUT_NAMES, inputs):
            input_sequence[name].append(value)

    for name in INPUT_NAMES:
        print(name)
        for value in input_sequence[name]:
            print(value)

    n_values = len(next(iter(input_sequence.values())))
    time_values = [TIME_INTERVAL*i for i in range(n_values)]

    name_label_map = dict(zip(INPUT_NAMES, INPUT_LABELS))

    plt.rcParams.update({"font.size": 16})
    marker_list = ["o", "v", "^", "s", "D", "P", "*"]
    color_list = ["blue", "orange"]


    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    ax_labels = ["Gas flow rate (mol/s)", "Solid flow rate (kg/s)"]
    ylims = [(80, 400), (300, 720)]
    ax1.set_xlabel("Time (s)")

    for i, name in enumerate(INPUT_NAMES):
        inputs = input_sequence[name]
        label = name_label_map[name]
        ax = axes[i]
        ax.set_ylim(ylims[i])
        ax.set_ylabel(ax_labels[i])
        ax.step(
            time_values,
            inputs,
            label=label,
            marker=marker_list[i],
            linewidth=2,
            markersize=10,
            color=color_list[i],
        )
    fig.legend(loc="upper left")
    fig.tight_layout()
    #fig.show()
    fig.savefig("input_sequence.png", transparent=True)


if __name__ == "__main__":
    main()
