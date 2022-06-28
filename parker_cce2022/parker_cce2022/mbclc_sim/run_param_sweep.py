import itertools
import json

from parker_cce2022.mbclc_sim.config import get_temperature_list, get_nxfe_list
from parker_cce2022.mbclc_sim.full_space import solve_full_space
from parker_cce2022.mbclc_sim.reduced_space import solve_reduced_space

from parker_cce2022.common.serialize.data_from_model import (
        get_structured_variables_from_model,
        )
from parker_cce2022.common.serialize.arithmetic import (
        subtract_variable_data,
        abs_variable_data,
        max_variable_data,
        )

"""
The purpose of this script is to compute the difference between models
resulting from full and reduced space solves.
"""

def main():
    #temperature_list = get_temperature_list()
    #nxfe_list = get_nxfe_list()

    gas_temp_list = [
        500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400
    ]
    solid_temp_list = [
        500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400
    ]

    #gas_temp_list = [800, 1000]
    #solid_temp_list = [800, 1000]

    full_solve_status = {}
    reduced_solve_status = {}
    full_solve_times = {}
    reduced_solve_times = {}
    errors = {}
    for gas_temp, solid_temp in itertools.product(
            gas_temp_list,
            solid_temp_list,
            ):
        res_full, m_full = solve_full_space(T=gas_temp, solid_temp=solid_temp)
        res_reduced, m_reduced = solve_reduced_space(
            T=gas_temp,
            solid_temp=solid_temp,
        )

        gas_length_full = m_full.fs.MB.gas_phase.length_domain
        solid_length_full = m_full.fs.MB.solid_phase.length_domain
        x0 = gas_length_full.first()
        x1 = gas_length_full.next(x0)
        sets = (gas_length_full, solid_length_full)
        indices = (x1, x1)
        data_full = get_structured_variables_from_model(m_full, sets, indices)
        data_full = data_full["model"]

        gas_length_reduced = m_reduced.fs.MB.gas_phase.length_domain
        solid_length_reduced = m_reduced.fs.MB.solid_phase.length_domain
        x0 = gas_length_reduced.first()
        x1 = gas_length_reduced.next(x0)
        sets = (gas_length_reduced, solid_length_reduced)
        indices = (x1, x1)
        data_reduced = get_structured_variables_from_model(m_reduced, sets, indices)
        data_reduced = data_reduced["model"]

        delta = [subtract_variable_data(data1, data2) for data1, data2 in
                zip(data_full, data_reduced)]
        abs_delta = [abs_variable_data(data) for data in delta]

        abs_values = [abs_variable_data(data) for data in data_full]

        max_delta = [max_variable_data(data) for data in abs_delta]
        max_values = [max_variable_data(data) for data in abs_values]
        state_errors = [
                (name, data1[name]/data2[name])
                if data2[name] != 0 else (name, 0.0)
                for data1, data2 in zip(max_delta, max_values)
                for name in data1
                ]

        state_errors = dict(state_errors)
        average_error = sum(state_errors.values())/len(state_errors)

        # Extract required values from results structure
        full_solve_status[gas_temp, solid_temp] = res_full.solver.termination_condition
        reduced_solve_status[gas_temp, solid_temp] = res_reduced.solver.termination_condition

        full_solve_times[gas_temp, solid_temp] = res_full.solver.time
        reduced_solve_times[gas_temp, solid_temp] = res_reduced.solver.wallclock_time

        errors[gas_temp, solid_temp] = average_error

    # TODO: Serialize this data
    data_json = []
    print("Parameters\tFull\t\t\tReduced\t\tError")
    for gas_temp, solid_temp in itertools.product(gas_temp_list, solid_temp_list):
        print("%s\t\t%s\t%1.6f\t%s\t%1.6f\t\t%1.6f" % (
            (gas_temp, solid_temp),
            full_solve_status[gas_temp, solid_temp],
            full_solve_times[gas_temp, solid_temp],
            reduced_solve_status[gas_temp, solid_temp],
            reduced_solve_times[gas_temp, solid_temp],
            errors[gas_temp, solid_temp],
            ))

        data_json.append((
            (gas_temp, solid_temp),
            {
                "full": (
                    str(full_solve_status[gas_temp, solid_temp]),
                    full_solve_times[gas_temp, solid_temp],
                    ),
                "reduced": (
                    str(reduced_solve_status[gas_temp, solid_temp]),
                    reduced_solve_times[gas_temp, solid_temp],
                    ),
                "error": errors[gas_temp, solid_temp],
                }
            ))

    with open("param_sweep.json", "w") as fp:
        json.dump(data_json, fp)


if __name__ == "__main__":
    main()
