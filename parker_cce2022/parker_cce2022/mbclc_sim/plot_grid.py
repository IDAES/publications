import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

"""
Open question: Should I have made these functions in my existing
plot_data.py file?

"""

FNAME = "param_sweep.json"

def get_results(fname):
    with open(fname, "r") as fp:
        results = json.load(fp)
    return results


def display_number_problems(results, key):
    n_problems = len(results)
    successful = [res for res in results if res[1][key][0] == "optimal"]
    n_successful = len(successful)
    print(key)
    print("%s/%s solved\n" % (n_successful, n_problems))


def display_average_solvetimes(results, key):
    nfes = sorted(set(nfe for (_, nfe), _ in results))
    temps = sorted(set(temp for (temp, _), _ in results))
    results_dict = dict((tuple(key), val) for key, val in results)

    successful_by_nfe = [
        [
            results_dict[temp, nfe][key][1]
            for temp in temps
            if results_dict[temp, nfe][key][0] == "optimal"
        ]
        for nfe in nfes
    ]
    sum_by_nfe = [sum(times) for times in successful_by_nfe]
    ave_by_nfe = [
        time/len(times) if len(times) > 0 else float('nan')
        for time, times in zip(sum_by_nfe, successful_by_nfe)
    ]
    print(key)
    print("NFE\tAverage solve time")
    for nfe, time in zip(nfes, ave_by_nfe):
        print("%s\t%s" % (nfe, time))
    print()


def display_average_solved_by_both(results):
    nfes = sorted(set(nfe for (_, nfe), _ in results))
    temps = sorted(set(temp for (temp, _), _ in results))
    results_dict = dict((tuple(key), val) for key, val in results)
    successful = [
        (
            results_dict[temp, nfe]["full"][1],
            results_dict[temp, nfe]["reduced"][1],
        )
        for temp in temps for nfe in nfes
        if (results_dict[temp, nfe]["full"][0] == "optimal"
            and results_dict[temp, nfe]["reduced"][0] == "optimal")
    ]
    n_both_solved = len(successful)
    total_full = sum(time for time, _ in successful)
    total_reduced = sum(time for _, time in successful)
    average_full = total_full/n_both_solved \
            if n_both_solved > 0 else float('nan')
    average_reduced = total_reduced/n_both_solved \
            if n_both_solved > 0 else float('nan')

    print("Number solved by both: %s" % n_both_solved)
    print("Average full: %s" % average_full)
    print("Average reduced: %s" % average_reduced)


def display_error_solved_by_both(results, tolerance=1e-2):
    nfes = sorted(set(nfe for (_, nfe), _ in results))
    temps = sorted(set(temp for (temp, _), _ in results))
    results_dict = dict((tuple(key), val) for key, val in results)
    successful = [
        ((temp, nfe), results_dict[temp, nfe]["error"])
        for temp in temps for nfe in nfes
        if (results_dict[temp, nfe]["full"][0] == "optimal"
            and results_dict[temp, nfe]["reduced"][0] == "optimal")
    ]
    n_both_solved = len(successful)
    print("Error in solves that were both successful:")
    for params, error in successful:
        print(params, error)

    same_sol = [(_, error) for _, error in successful if error <= tolerance]
    n_same_sol = len(same_sol)

    print(
        "Problems converged to same solution within tolerance: %s/%s"
        % (n_same_sol, n_both_solved)
    )


def plot_results(ax, results, key):
    # These results are "flattened" in the sense that they
    # are a 1-d array, each with a tuple index.
    # I want to get them back into the "matrix form."
    # There must be an "order-preserving" way to do this...
    # The problem is that order is not necessarily the same
    # "between rows." We need to know something about the data
    # to "stack rows." Otherwise the best we can do is sort...
    gas_temps = sorted(set(temp for (temp, _), _ in results))
    solid_temps = sorted(set(temp for (_, temp), _ in results), reverse=True)
    results_dict = dict((tuple(key), val) for key, val in results)
    result_array = np.array([[
        1 if results_dict[g_temp, s_temp][key][0] == "optimal" else 0
        for g_temp in gas_temps
    ] for s_temp in solid_temps])

    cmap = ListedColormap([
        (0.1, 0.1, 0.1),
        (0.8, 0.8, 0.8),
    ])

    ax.matshow(result_array, cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("Gas temperature (K)")
    ax.set_ylabel("Solid temperature (K)")

    x_ticks = [i for i in range(len(gas_temps)) if not i % 2]
    x_tick_labels = [str(round(gas_temps[i])) for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.xaxis.set_ticks_position("bottom")

    y_ticks = list(range(len(solid_temps)))
    y_tick_labels = [str(temp) for temp in solid_temps]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    if key == "reduced":
        ax.set_title("Implicit function")
    elif key == "full":
        ax.set_title("Full space")
    else:
        raise RuntimeError()

    ax.set_xticks(np.arange(-0.5, len(gas_temps), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(solid_temps), 1), minor=True)

    ax.grid(which="minor", linestyle="-", linewidth=1.5)
    ax.tick_params(length=0)


if __name__ == "__main__":
    results = get_results(FNAME)

    plt.rcParams.update({"font.size": 18})
    figure = plt.figure()
    ax_full = figure.add_subplot(121)

    plot_results(ax_full, results, "full")

    ax_ifcn = figure.add_subplot(122)

    plot_results(ax_ifcn, results, "reduced")

    cmap = ListedColormap([
        (0.1, 0.1, 0.1),
        (0.8, 0.8, 0.8),
    ])
    colors = [cmap(0), cmap(1)]
    labels = ["Unsuccessful", "Successful"]
    patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, labels)
    ]

    figure.legend(
        handles=patches,
        #loc=5,
    )

    figure.set_size_inches(11, 7)
    figure.tight_layout()

    #figure.set_figheight(2.4)
    #figure.set_figwidth(16.0)

    figure.savefig("simulation_grid.pdf", transparent=True)
    display_average_solvetimes(results, "full")
    display_average_solvetimes(results, "reduced")
    display_number_problems(results, "full")
    display_number_problems(results, "reduced")
    display_average_solved_by_both(results)
    display_error_solved_by_both(results)
