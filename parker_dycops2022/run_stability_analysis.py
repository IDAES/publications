import itertools
import json
import pyomo.environ as pyo

from pyomo.common.collections import ComponentSet
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    _generate_variables_in_constraints,
)
from pyomo.dae.flatten import flatten_components_along_sets

from workspace.mbclc.model import (
    make_square_dynamic_model,
    make_square_model,
)
from workspace.mbclc.initialize import (
    set_default_design_vars,
    set_default_inlet_conditions,
    initialize_steady,
    initialize_dynamic_from_steady,
)
from workspace.mbclc.plot import (
    plot_outlet_states_over_time,
)
from workspace.common.initialize import (
    initialize_by_time_element,
    get_subsystems_along_time,
)
from workspace.common.dynamic_data import (
    load_inputs_into_model,
)
from workspace.common.serialize.data_from_model import (
    get_structured_variables_from_model,
)
from workspace.common.serialize.interpolate import interpolate_data_onto_sets
from workspace.common.serialize.integrate import integrate_variable_data
from workspace.common.serialize.arithmetic import (
    subtract_variable_data,
    sum_variable_data,
    multiply_variable_data,
    concatenate_data_along_set,
)
from idaes.apps.caprese.categorize import (
    VariableCategory as VC,
    ConstraintCategory as CC,
)
from workspace.mbclc.results.dycops2022.model import (
    get_model_for_simulation,
)

from workspace.common.timing import HierarchicalTimer

TIMER = HierarchicalTimer()
import workspace.common.initialize as initialize_module
initialize_module.TIMER = TIMER


def simulate_and_get_data(
        input_dict=None,
        model_kwds=None,
        solve_kwds=None,
        input_dict_list=None,
        n_samples=1,
        ):
    """
    Note that this function simulates a model many times
    to get the full horizon simulation we are interested in.
    """
    if input_dict is None:
        input_dict = {}
    if model_kwds is None:
        model_kwds = {}
    if solve_kwds is None:
        solve_kwds = {}
    if input_dict_list is None:
        # TODO: Remove input_dict argument
        input_dict_list = [{}]*n_samples

    # Outside of loop: make_model
    with TIMER.context("make model"):
        # Expect this to be expensive, but not frequent
        m, var_cat, con_cat = get_model_for_simulation(**model_kwds)
    time = m.fs.time
    gas_length = m.fs.MB.gas_phase.length_domain
    solid_length = m.fs.MB.solid_phase.length_domain
    sets = (time, gas_length, solid_length)
    indices = tuple(next(iter(s)) for s in sets)
    time_space_flatten_vars = flatten_components_along_sets(
        m, sets, pyo.Var, indices
    )

    # Recover the "flattened vars" here so we don't have to flatten
    # again in the initialize_by_time_element function
    scalar_vars = []
    dae_vars = []
    scalar_cons = []
    dae_cons = []
    for varlist in var_cat.values():
        if varlist:
            var = next(iter(varlist))
            if var.is_indexed():
                # Assume all variables in this list are indexed
                dae_vars.extend(varlist)
            else:
                scalar_vars.extend(varlist)
    for conlist in con_cat.values():
        if conlist:
            con = next(iter(conlist))
            if con.is_indexed():
                dae_cons.extend(conlist)
            else:
                scalar_cons.extend(conlist)
    flatten_vars = (scalar_vars, dae_vars)
    flatten_cons = (scalar_cons, dae_cons)
    # Should be able to move get_subsystems_along_time out of the loop,
    # to right here.
    subsystems = get_subsystems_along_time(m, time, flatten_vars, flatten_cons)

    total_data = None
    for i in range(n_samples):
        input_dict = input_dict_list[i]
        with TIMER.context("load inputs"):
            # Expect this not to register
            load_inputs_into_model(m, time, input_dict)

        solver = pyo.SolverFactory("ipopt")
        with TIMER.context("simulate"):
            # Expect this to be expensive
            res_list = initialize_by_time_element(
                m,
                time,
                solver=solver,
                solve_kwds=solve_kwds,
                skip_partition=True,
                flatten_vars=flatten_vars,
                flatten_cons=flatten_cons,
                time_subsystems=subsystems,
            )
        with TIMER.context("get data"):
            # Expect this to be expensive
            data = get_structured_variables_from_model(
                m,
                sets,
                flatten_vars=time_space_flatten_vars,
            )

        if total_data is None:
            total_data = data
        else:
            with TIMER.context("concatenate"):
                # Don't expect this to be expensive
                total_data = {
                    "model": [
                        concatenate_data_along_set(
                            subset1,
                            subset2,
                            time,
                        )
                        for subset1, subset2 in zip(
                            total_data["model"], data["model"]
                        )
                    ],
                }

        # "Loop" model - set initial conditions from final values
        t0 = time.first()
        tf = time.last()
        with TIMER.context("reinitialize"):
            # Don't expect this to be expensive
            for varlist in var_cat.values():
                for var in varlist:
                    var_tf = var[tf]
                    if var.is_indexed():
                        for t in time:
                            var[t].set_value(var_tf.value)

    return total_data


def main():
    #horizon = 600.0
    #ntfe = 40
    horizon = 15.0
    ntfe = 2
    # This coded as a default in model_kwds, but is never
    # used. TODO: remove

    # The number of times we repeat this simulation of length horizon
    # TODO: "samples" is a bad name here. Really this is the number of
    # times we repeat the simulation of our model (with above horizon)
    n_samples = 40
    t1 = 0.0
    # TODO: Should really use a more complicated simulation that
    # varies inputs as well, here.
    ch4_cuid = "fs.MB.gas_inlet.mole_frac_comp[*,CH4]"
    co2_cuid = "fs.MB.gas_inlet.mole_frac_comp[*,CO2]"
    h2o_cuid = "fs.MB.gas_inlet.mole_frac_comp[*,H2O]"
    disturbance_dict = {
        ch4_cuid: {(t1, horizon): 0.5},
        co2_cuid: {(t1, horizon): 0.5},
        h2o_cuid: {(t1, horizon): 0.0},
    }

    PERTURB_INPUTS = True
    # Set up perturbed input values we want to apply
    gas_input_values = [100.0, 128.2, 200.0]
    solid_input_values = [500.0, 591.4, 700.0]
    input_list = list(itertools.product(gas_input_values, solid_input_values))
    n_input_applications = 10
    repetitions_per_input = n_samples//n_input_applications \
        if n_samples > n_input_applications else 1

    # Set up list of input dicts to use for each sample
    input_dict_list = [
        dict(disturbance_dict)
        for _ in range(max(n_input_applications, n_samples))
        # This is still okay. If n_input_applications is greater, we just
        # create a list that is longer than needed. If n_samples is
        # greater, we create an input dict for each "sample," which is what
        # is needed.
    ]
    if PERTURB_INPUTS:
        for i in range(1, n_input_applications):
            # There are 10 samples and 9 input combinations.
            # On the first sample, we leave the inputs unperturbed.
            gas_flow, solid_flow = input_list[i-1]
            input_dict = {
                "fs.MB.gas_inlet.flow_mol[*]": {(0, horizon): gas_flow},
                "fs.MB.solid_inlet.flow_mass[*]": {(0, horizon): solid_flow},
            }
            for j in range(repetitions_per_input):
                idx = repetitions_per_input * i + j
                input_dict_list[idx].update(input_dict)

    model_kwds = {"horizon": horizon, "ntfe": ntfe}
    time_disc_list = [
        #{"ntfe": 2},
        #{"ntfe": 4},
        #{"ntfe": 8},
        {"ntfe": 16},
        {"ntfe": 32},
        #{"ntfe": 64},
        #{"ntfe": 128},
    ]
    space_disc_list = [
        #{"nxfe": 2},
        #{"nxfe": 4},
        #{"nxfe": 8},
        #{"nxfe": 16},
        #{"nxfe": 32},
        {"nxfe": 64},
        {"nxfe": 128},
    ]
    data_list = []
    solve_kwds = {"tee": True}
    for time_disc, space_disc in zip(time_disc_list, space_disc_list):
        # I had an earlier bug where I mistyped these above...
        assert "ntfe" not in space_disc and "nxfe" in space_disc
        assert "nxfe" not in time_disc and "ntfe" in time_disc

        model_kwds.update(time_disc)
        model_kwds.update(space_disc)
        data = simulate_and_get_data(
            input_dict_list=input_dict_list,
            #input_dict=input_dict,
            model_kwds=dict(model_kwds),
            solve_kwds=solve_kwds,
            n_samples=n_samples,
        )
        data_list.append(data)

        # Code to debug
        #fname = "%s_%ss_%sdisc.json" % (
        #    n_samples, int(horizon), time_disc["ntfe"]
        #)
        #with open(fname, "w") as fp:
        #    json.dump(data, fp)

    # These are the variables we actually care about
    PDE_VARIABLE_NAMES = [
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

    pde_errors = {name: [] for name in PDE_VARIABLE_NAMES}

    # Once we have the data at every discretization point (which we
    # may compute and serialize in a separate script/function),
    # loop through and interpolate/get difference between each pair.
    n_disc = len(time_disc_list)
    error_data = []
    for i in range(1, n_disc):
        data1 = data_list[i-1]
        data2 = data_list[i]
        # Interpolate data1 (coarse) onto sets used in data2 (fine)
        set_dict = {}
        for subset in data2["model"]:
            # Subset of variables that are indexed by these sets:
            sets = [] if subset["sets"] is None else subset["sets"]
            indices = [] if subset["indices"] is None else subset["indices"]
            set_dict.update(dict(zip(sets, indices)))
        # This line leads us with "nan"s in the interpolated data if any
        # of the variable values are None.
        # TODO: Should try to make sure that no variable values are None
        interp_data = interpolate_data_onto_sets(data1["model"], set_dict)

        for subset1, subset2 in zip(interp_data, data2["model"]):
            # Assuming subsets are in same order...
            assert subset1["sets"] == subset2["sets"]
            assert subset1["indices"] == subset2["indices"]

        difference = [
            subtract_variable_data(subset1, subset2)
            for subset1, subset2 in zip(interp_data, data2["model"])
        ]
        squared = [
            multiply_variable_data(subset, subset) for subset in difference
        ]
        integral_squared_error = [
            integrate_variable_data(subset) for subset in squared
        ]

        error = integral_squared_error

        # TODO: Better name. This is just the error in variables we care about
        pde_error = {}
        for subset in error:
            for name in PDE_VARIABLE_NAMES:
                if name in subset:
                    if name in pde_error:
                        raise RuntimeError(
                            "The same name shouldn't appear twice"
                        )
                    pde_error[name] = subset[name]
                    pde_errors[name].append(subset[name])

        error_data.append(error)

    for name, errors in pde_errors.items():
        print(name, errors)
    fname = "_".join(["error"] + [str(d["ntfe"]) for d in time_disc_list])
    with open(fname, "w") as fp:
        json.dump(pde_errors, fp)


if __name__ == "__main__":
    with TIMER.context("main"):
        main()
    print(TIMER)
