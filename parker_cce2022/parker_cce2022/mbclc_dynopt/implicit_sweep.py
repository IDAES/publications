import itertools

import pyomo.environ as pyo
from parker_cce2022.mbclc_dynopt.run_implicit_function import (
    run_dynamic_optimization,
)
from parker_cce2022.mbclc_dynopt.model import (
    get_steady_state_data,
    get_dynamic_model,
)
from parker_cce2022.mbclc_dynopt.solve_data import SolveData
from parker_cce2022.mbclc_dynopt.sweep_data import (
    SweepDataContainer,
    SweepData,
)

import json

default_ipopt_options = {
    "tol": 5e-5,
    "inf_pr_output": "internal",
    "dual_inf_tol": 1e2,
    "constr_viol_tol": 1e2,
    "compl_inf_tol": 1e2,
    "nlp_scaling_method": "user-scaling",
    "max_cpu_time": 300.0,
    "max_iter": 1000,
}

def run_parameter_sweep(
        solid_inlet_flow_list,
        conversion_list,
        nxfe=10,
        samples_per_horizon=10,
        ipopt_options=None,
        input_bounds=False,
        differential_bounds=False,
        ):
    #
    # Solve for initial steady state outside of the loop
    #
    ic_scalar_data, ic_dae_data = get_steady_state_data(nxfe)

    if ipopt_options is None:
        ipopt_options = default_ipopt_options
    else:
        # If a key is not provided but has a default,
        # use the default value.
        for key, val in default_ipopt_options.items():
            if key not in ipopt_options:
                ipopt_options[key] = val

    metadata = {
        "nxfe": nxfe,
        "samples_per_horizon": samples_per_horizon,
        "ipopt_options": ipopt_options,
        "input_bounds": input_bounds,
        "differential_bounds": differential_bounds,
    }
    results = []
    sweep_data_container = SweepDataContainer(metadata, results)

    for i, (flow, conv) in enumerate(itertools.product(
            solid_inlet_flow_list, conversion_list
            )):
        print(i)
        sp_input_map = {
            "fs.MB.solid_phase.properties[*,1.0].flow_mass": flow,
        }
        sp_dof_names = [
            "fs.MB.gas_phase.properties[*,0.0].flow_mol"
        ]
        # Why is this data structure a list of tuples again?
        sp_state_list = [
            ("fs.MB.solid_phase.reactions[*,0.0].OC_conv", conv)
        ]

        try:
            _, sp_dae_data = get_steady_state_data(
                nxfe=nxfe,
                input_map=sp_input_map,
                to_unfix=sp_dof_names,
                setpoint_list=sp_state_list,
                tee=False,
            )
        except ValueError as err:
            print(err)
            print()
            continue

        #
        # Get small data structure to represent setpoint
        # sp_data maps names (strings) to values
        #
        sp_data = {
            # This is a hack to get around inconsistency of string
            # representation
            name: sp_dae_data[str(pyo.ComponentUID(name))]
            for name in sp_dof_names
        }
        sp_data.update(sp_input_map)
        sp_data.update(dict(sp_state_list))
        for key, val in sp_data.items():
            print(key, val)
        print()

        solve_data = run_dynamic_optimization(
            initial_conditions=ic_dae_data,
            setpoint=sp_dae_data,
            scalar_data=ic_scalar_data,
            nxfe=nxfe,
            samples_per_horizon=samples_per_horizon,
            ipopt_options=ipopt_options,
            input_bounds=input_bounds,
            differential_bounds=differential_bounds,
        )

        inputs = {
            "fs.MB.solid_phase.reactions[*,0.0].OC_conv": conv,
            "fs.MB.solid_phase.properties[*,1.0].flow_mass": flow,
        }
        setpoint = sp_data
        solve = SolveData(
            # solve_data.status should be a string here.
            solve_data.status,
            solve_data.values,
            solve_data.time,
        )
        sweep_data = SweepData(inputs, setpoint, solve)
        sweep_data_container.results.append(sweep_data)

    return sweep_data_container


if __name__ == "__main__":
    nxfe = 10
    samples_per_horizon = 10

    #solid_inlet_flow_list = [500.0 + (i+1)*20.0 for i in range(10)]
    #conversion_list = [0.88 + i*0.01 for i in range(10)]
    #solid_inlet_flow_list = [700.0]
    #conversion_list = [0.95]
    solid_inlet_flow_list = [660.0 + i*20.0 for i in range(10)]
    conversion_list = [0.90 + 0.01*i for i in range(10)]
    #solid_inlet_flow_list = [700.0]
    #conversion_list = [0.96, 0.97]

    ipopt_options = {"tol": 4e-4}
    data_container = run_parameter_sweep(
        solid_inlet_flow_list,
        conversion_list,
        nxfe=nxfe,
        samples_per_horizon=samples_per_horizon,
        ipopt_options=ipopt_options,
        input_bounds=True,
    )
    metadata = data_container.metadata
    results = data_container.results

    print("Implicit function parameter sweep")
    print(
        "nxfe = %s, samples_per_horizon = %s"
        % (metadata["nxfe"], metadata["samples_per_horizon"])
    )
    print("-----------------------------------")
    for sweep_data in results:
        input_values = list(sweep_data.inputs.values())
        status = sweep_data.solve.status
        solve_time = sweep_data.solve.time
        print(
            input_values[0],
            input_values[1],
            status,
            solve_time,
        )

    n_total = len(results)
    converged_results = [
        res for res in results if res.solve.status == "Solve_Succeeded"
    ]
    n_converged = len(converged_results)
    print("\nConverged %s/%s problems" % (n_converged, n_total))

    with open("implicit_function.json", "w") as fp:
        json.dump(data_container, fp)
