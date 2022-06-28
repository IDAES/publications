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

from parker_cce2022.mbclc_dynopt.implicit_sweep import (
    run_parameter_sweep,
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


def main():
    nxfe = 10
    samples_per_horizon = 10

    solid_inlet_flow_list = [660.0 + i*20.0 for i in range(10)]
    conversion_list = [0.90 + 0.01*i for i in range(10)]

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

    with open("implicit_sweep_boundcon.json", "w") as fp:
        json.dump(data_container, fp)


if __name__ == "__main__":
    main()
