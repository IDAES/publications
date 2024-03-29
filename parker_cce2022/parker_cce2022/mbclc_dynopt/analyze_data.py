import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

solid_flow_name = "fs.MB.solid_phase.properties[*,1.0].flow_mass"
conversion_name = "fs.MB.solid_phase.reactions[*,0.0].OC_conv"

def plot_results_on_axes(ax, results, wrong_solutions):

    flow_rates = list(sorted(set(
        inputs[solid_flow_name] for inputs, _, _ in results
    )))
    conversions = list(sorted(set(
        inputs[conversion_name] for inputs, _, _ in results
    ), reverse=True))

    status_dict = {
        (inputs[solid_flow_name], inputs[conversion_name]): status
        for inputs, _, (status, _, _) in results
    }
    wrong_sol_dict = {
        (inputs[solid_flow_name], inputs[conversion_name]): wrong_solutions[i]
        for i, (inputs, _, _) in enumerate(results)
    }

    result_array = np.array([
        [
            1 if (
                # We converge successfully...
                status_dict[flow, conv] == "optimal"
                or status_dict[flow, conv] == "Solve_Succeeded"
                # to the right solution.
            ) and not wrong_sol_dict[flow, conv]
            else 0
            for flow in flow_rates]
        for conv in conversions
    ])

    cmap = ListedColormap([(0.1, 0.1, 0.1), (0.8, 0.8, 0.8)])
    ax.matshow(result_array, cmap=cmap, vmin=0, vmax=1)

    ax.set_xlabel("Solid flow rate (kg/s)")
    ax.set_ylabel("Solid conversion")

    x_ticks = [i for i in range(len(flow_rates)) if not i % 2]
    x_tick_labels = [str(round(flow_rates[i])) for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.xaxis.set_ticks_position("bottom")

    y_ticks = list(range(len(conversions)))
    y_tick_labels = ["%1.2f" % round(conv, 2) for conv in conversions]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

    ax.grid(which="minor", linestyle="-", linewidth=1.5)
    ax.tick_params(length=0)

    return ax


def analyze_results(
    fullspace_fname,
    implicit_fname,
    out_fname=None,
    show=False,
    save=False,
    display_all_solvetimes=False,
):
    with open(fullspace_fname, "r") as fp:
        fullspace_data = json.load(fp)

    with open(implicit_fname, "r") as fp:
        implicit_data = json.load(fp)

    fullspace_metadata, fullspace_results = fullspace_data
    implicit_metadata, implicit_results = implicit_data

    print("Metadata:")
    print(implicit_metadata)
    print()

    implicit_metadata["ipopt_options"].pop("inf_pr_output")
    assert fullspace_metadata == implicit_metadata

    n_total = len(fullspace_results)
    assert n_total == len(implicit_results)

    fullspace_results_solved = []
    for res in fullspace_results:
        inputs, setpoint, (status, values, time) = res
        if status == "optimal":
            fullspace_results_solved.append(res)

    implicit_results_solved = []
    for res in implicit_results:
        inputs, setpoint, (status, values, time) = res
        if status == "Solve_Succeeded":
            implicit_results_solved.append(res)

    implicit_results_not_solved = []
    for res in implicit_results:
        inputs, setpoint, (status, values, time) = res
        if status != "Solve_Succeeded":
            implicit_results_not_solved.append(res)

    # The results, for each formulation, that were successfully converged
    # by both formulations.
    fullspace_both_solved = []
    implicit_both_solved = []
    for res_full, res_ifcn in zip(fullspace_results, implicit_results):
        _, _, (stat_full, _, _) = res_full
        _, _, (stat_ifcn, _, _) = res_ifcn
        if stat_full == "optimal" and stat_ifcn == "Solve_Succeeded":
            fullspace_both_solved.append(res_full)
            implicit_both_solved.append(res_ifcn)

    n_implicit_faster = 0
    for res_full, res_ifcn in zip(fullspace_both_solved, implicit_both_solved):
        _, _, (_, _, time_full) = res_full
        _, _, (_, _, time_ifcn) = res_ifcn
        if time_ifcn < time_full:
            n_implicit_faster += 1

    # Get problem instances that converge to same solution.
    n_same_solution = 0
    full_wrong_solution = []
    ifcn_wrong_solution = []
    for res_full, res_ifcn in zip(fullspace_results, implicit_results):
        params_full, _, (stat_full, inputs_full, _) = res_full
        params_ifcn, _, (stat_ifcn, inputs_ifcn, _) = res_ifcn
        if stat_full != "optimal" or stat_ifcn != "Solve_Succeeded":
            # At least one of the formulations did not converge.
            # Vacuously, we don't say that either reached the wrong solution.
            full_wrong_solution.append(False)
            ifcn_wrong_solution.append(False)
            # We will not do any comparison.
            continue
        # Same input variables
        assert set(inputs_full[0].keys()) == set(inputs_ifcn[0].keys())
        # Same time points
        assert inputs_full[1] == inputs_ifcn[1]
        # Same parameters in sweep
        assert params_full == params_ifcn
        same_solution = True
        for key in inputs_full[0].keys():
            full_data = inputs_full[0][key]
            ifcn_data = inputs_ifcn[0][key]
            differences = [
                # Use a relative difference based on the larger value
                abs(val1 - val2)/max((val1, val2))
                # But if both values are small (e.g. tracking cost),
                # just use the absolute difference.
                if max((val1, val2)) >= 1.0 else abs(val1 - val2)
                for val1, val2 in zip(full_data, ifcn_data)
            ]
            # If any control input has a significant difference at any point
            # in time, we say the problems did not converge to the same solution.
            if any(diff >= 0.01 for diff in differences):
                print(
                    "Problems with parameters %s did not converge to same solution"
                    % params_full
                )
                same_solution = False
                tr_cost_key = "tracking_cost[*]"
                full_cost = sum(inputs_full[0][tr_cost_key])
                ifcn_cost = sum(inputs_ifcn[0][tr_cost_key])
                # Use min here so we can say "the _ formulation is X% worse"
                margin = abs(full_cost - ifcn_cost)/min((full_cost, ifcn_cost))
                obj_comparison_tol = 0.05
                if margin < obj_comparison_tol:
                    print(
                        "The two solutions are equally good within a tolerance of %s"
                        % obj_comparison_tol
                    )
                    full_wrong_solution.append(False)
                    ifcn_wrong_solution.append(False)
                elif full_cost < ifcn_cost:
                    factor = ifcn_cost / full_cost
                    print(
                        "Full-space has a better solution by a factor of %s"
                        % factor
                    )
                    full_wrong_solution.append(False)
                    ifcn_wrong_solution.append(True)
                elif ifcn_cost < full_cost:
                    factor = full_cost / ifcn_cost
                    print(
                        "Implicit has a better solution by a factor of %s"
                        % factor
                    )
                    full_wrong_solution.append(True)
                    ifcn_wrong_solution.append(False)
                break
        if same_solution:
            full_wrong_solution.append(False)
            ifcn_wrong_solution.append(False)
            n_same_solution += 1
    print("Number converged to same solution: %s" % n_same_solution)

    #
    # Now reclassify, only accepting instances that converge to
    # the correct solution.
    #
    fullspace_results_solved = []
    for i, res in enumerate(fullspace_results):
        inputs, setpoint, (status, values, time) = res
        if status == "optimal" and not full_wrong_solution[i]:
            fullspace_results_solved.append(res)

    implicit_results_solved = []
    for i, res in enumerate(implicit_results):
        inputs, setpoint, (status, values, time) = res
        if status == "Solve_Succeeded" and not ifcn_wrong_solution[i]:
            implicit_results_solved.append(res)

    implicit_results_not_solved = []
    for i, res in enumerate(implicit_results):
        inputs, setpoint, (status, values, time) = res
        # "Not solved" should include instances that converge to the wrong
        # solutions.
        if status != "Solve_Succeeded" or ifcn_wrong_solution[i]:
            implicit_results_not_solved.append(res)

    # The results, for each formulation, that were successfully converged
    # by both formulations.
    fullspace_both_solved = []
    implicit_both_solved = []
    for i, (res_full, res_ifcn) in enumerate(
        zip(fullspace_results, implicit_results)
    ):
        _, _, (stat_full, _, _) = res_full
        _, _, (stat_ifcn, _, _) = res_ifcn
        if (
            stat_full == "optimal" and stat_ifcn == "Solve_Succeeded"
            and not ifcn_wrong_solution[i] and not full_wrong_solution[i]
        ):
            fullspace_both_solved.append(res_full)
            implicit_both_solved.append(res_ifcn)

    n_implicit_faster = 0
    for res_full, res_ifcn in zip(fullspace_both_solved, implicit_both_solved):
        _, _, (_, _, time_full) = res_full
        _, _, (_, _, time_ifcn) = res_ifcn
        if time_ifcn < time_full:
            n_implicit_faster += 1
    ###

    n_converged_fullspace = len(fullspace_results_solved)
    n_converged_implicit = len(implicit_results_solved)

    fullspace_solvetimes = [
        time for _, _, (_, _, time) in fullspace_results_solved
    ]
    # Sometimes may be useful to see the solve times of each instance
    # that converges.
    if display_all_solvetimes:
        print("fullspace times")
        for params, _, (_, _, time) in fullspace_results_solved:
            print(params, time)
    implicit_solvetimes = [
        time for _, _, (_, _, time) in implicit_results_solved
    ]
    if display_all_solvetimes:
        print("implicit times")
        for params, _, (_, _, time) in implicit_results_solved:
            print(params, time)
    ave_time_fullspace = sum(fullspace_solvetimes)/len(fullspace_solvetimes)
    ave_time_implicit = sum(implicit_solvetimes)/len(implicit_solvetimes)

    print(
        "Implicit function converged %s/%s problems"
        % (n_converged_implicit, n_total)
    )
    print(
        "Full space converged %s/%s problems"
        % (n_converged_fullspace, n_total)
    )

    print(
        "Implicit function converged in an average of %s seconds"
        % ave_time_implicit
    )
    print(
        "Full space converged in an average of %s seconds"
        % ave_time_fullspace
    )
    print(
        "The implicit function formulation converges faster in %s instances\n"
        "out of the %s that both formulations converge."
        % (n_implicit_faster, len(fullspace_both_solved))
    )

    print("\nProblems not converged by the implicit function formulation are:")
    for res in implicit_results_not_solved:
        inputs, _, (status, _, _) = res
        print(
            "  %s, %s: %s"
            % (inputs[solid_flow_name], inputs[conversion_name], status)
        )

    plt.rcParams.update({"font.size": 18})
    figure = plt.figure()
    ax_full = figure.add_subplot(121)
    ax_ifcn = figure.add_subplot(122)

    plot_results_on_axes(ax_full, fullspace_results, full_wrong_solution)
    plot_results_on_axes(ax_ifcn, implicit_results, ifcn_wrong_solution)

    ax_full.set_title("Full space")
    ax_ifcn.set_title("Implicit function")

    # TODO: Make cmap global?
    cmap = ListedColormap([(0.1, 0.1, 0.1), (0.8, 0.8, 0.8)])
    colors = [cmap(0), cmap(1)]
    labels = ["Unsuccessful", "Successful"]
    patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, labels)
    ]
    figure.legend(
        handles=patches,
        loc=1,
    )

    figure.set_size_inches(11, 7)
    figure.tight_layout()

    if show:
        plt.show()
    if save:
        if out_fname is not None:
            figure.savefig(out_fname, transparent=True)
        else:
            out_fname = "sweep.pdf"


def main():
    display_all_solvetimes = False
    print("################################")
    print("# Equality-constrained results #")
    print("################################")
    fullspace_fname = "full_space_sweep_eqcon.json"
    implicit_fname = "implicit_sweep_eqcon.json"
    out_fname = "eqcon_sweep.pdf"
    analyze_results(
        fullspace_fname,
        implicit_fname,
        out_fname=out_fname,
        show=True,
        save=False,
        display_all_solvetimes=display_all_solvetimes,
    )

    print()
    print("#############################")
    print("# Bound-constrained results #")
    print("#############################")
    fullspace_fname = "full_space_sweep_boundcon.json"
    implicit_fname = "implicit_sweep_boundcon.json"
    out_fname = "boundcon_sweep.pdf"
    analyze_results(
        fullspace_fname,
        implicit_fname,
        out_fname=out_fname,
        show=True,
        save=False,
        display_all_solvetimes=display_all_solvetimes,
    )


if __name__ == "__main__":
    main()
