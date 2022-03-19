import json
import pyomo.environ as pyo
import pyomo.dae as dae

from pyomo.dae.flatten import flatten_dae_components
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.common.collections import ComponentSet
#from pyomo.common.timing import HierarchicalTimer
from common.timing import HierarchicalTimer
from common.initialize import initialize_by_time_element

import mbclc.model as model_module
from mbclc.model import (
    make_square_dynamic_model,
    make_square_model,
    add_constraints_for_missing_variables,
)
import mbclc.initialize as initialize_module
from mbclc.initialize import (
    set_default_design_vars,
    set_default_inlet_conditions,
    initialize_steady,
    initialize_dynamic_from_steady,
)
from mbclc.plot import (
    plot_outlet_states_over_time,
    plot_inputs_over_time,
    plot_outlet_data,
)
from common.dynamic_data import (
    load_inputs_into_model,
    get_inputs_at_time,
    interval_data_from_time_series,
    initialize_time_series_data,
    extend_time_series_data,
)
import model as dynopt_module
from model import (
    get_steady_state_model,
    get_data_from_steady_model,
    get_model_for_dynamic_optimization,
    initialize_dynamic,
    get_state_variable_names,
    get_tracking_cost_expressions,
    get_nmpc_plant_model,
)
from scaling import (
    get_max_values_from_steady,
)
# TODO: import scaling factors and variance
from deserialize import (
    get_variance_data,
    get_scaling_factor_data,
    get_variance_of_time_slices,
    get_scaling_of_time_slices,
)


TIMER = HierarchicalTimer()
initialize_module.TIMER = TIMER
model_module.TIMER = TIMER
dynopt_module.TIMER = TIMER


def main():
    """
    """
    nxfe = 10 # Default = 10
    # NOTE: Default inputs: (128.2, 591.4)
    ic_model_params = {"nxfe": nxfe}
    ic_inputs = {
        #"fs.MB.gas_phase.properties[*,0.0].flow_mol": 120.0,
        #"fs.MB.solid_phase.properties[*,1.0].flow_mass": 550.0,
    }
    m_ic = get_steady_state_model(
        ic_inputs,
        solve_kwds={"tee": True},
        model_params=ic_model_params,
    )
    time = m_ic.fs.time
    scalar_data, dae_data = get_data_from_steady_model(m_ic, time)
    # TODO: get steady state data (scalar and dae both necessary)
    x0 = 0.0
    x1 = 1.0

    # NOTE: Decreasing size of model (horizon, tfe_width, and nxfe)
    # while developing NMPC workflow. Defaults: 900, 15, 10
    n_nmpc_samples = 30 # For now. My target should probably be around 20-30
    horizon = 1800
    tfe_width = 60
    sample_width = 180
    sample_points = [
        # Calculate sample points first with integer arithmetic
        # to avoid roundoff error
        float(sample_width*i) for i in range(0, horizon//sample_width + 1)
    ]
    nmpc_sample_points = [
        float(sample_width*i) for i in range(n_nmpc_samples)
        # This is the "real time" at which NMPC inputs are applied
        # for each solve of the optimal control problem.
    ]
    horizon = float(horizon)
    tfe_width = float(tfe_width)
    model_params = {
        "horizon": horizon,
        "tfe_width": tfe_width,
        "ntcp": 1,
        "nxfe": nxfe,
    }

    # These are approximately the default values:
    #disturbance_dict = {"CO2": 0.03, "H2O": 0.0, "CH4": 0.97}
    disturbance_dict = {"CO2": 0.5, "H2O": 0.0, "CH4": 0.5}
    disturbance = dict(
        (
            "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[%s]" % (x0, j),
            {(0.0, horizon): val},
        )
        for j, val in disturbance_dict.items()
    )

    # Create solver here as it is needed to solve for the setpoint
    solver = pyo.SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["max_cpu_time"] = 900

    #
    # Get setpoint data
    #
    sp_inputs = get_inputs_at_time(disturbance, horizon)
    #sp_inputs.update({
    #    "fs.MB.gas_phase.properties[*,0.0].flow_mol": 272.8,
    #    "fs.MB.solid_phase.properties[*,1.0].flow_mass": 591.4,
    #})
    sp_model_params = {"nxfe": nxfe}
    m_sp = get_steady_state_model(
        sp_inputs,
        solve_kwds={"tee": True},
        model_params=sp_model_params,
    )
    time = m_sp.fs.time
    space = m_sp.fs.MB.gas_phase.length_domain
    t0 = time.first()
    # Solve optimization problem for setpoint
    sp_objective_states = [
        "fs.MB.solid_phase.reactions[*,%s].OC_conv" % x0,
    ]
    sp_target = {
        "fs.MB.solid_phase.reactions[*,%s].OC_conv" % x0: 0.95,
    }
    m_sp.fs.MB.gas_inlet.flow_mol[:].unfix()
    m_sp.setpoint_expr = get_tracking_cost_expressions(
        sp_objective_states, time, sp_target
    )
    m_sp.objective = pyo.Objective(expr=m_sp.setpoint_expr[t0])
    solver.solve(m_sp, tee=True)
    scalar_vars, dae_vars = flatten_dae_components(m_sp, time, pyo.Var)
    setpoint = {
        str(pyo.ComponentUID(var.referent)): var[t0].value
        for var in dae_vars
    }
    ###

    max_data = get_max_values_from_steady(m_sp)
    variance_data = get_variance_of_time_slices(m_sp, time, space)
    weight_data = None
    #weight_data = {
    #    name: 1.0/s if s != 0 else 1.0 for name, s in variance_data.items()
    #    #name: 1/w if w != 0 else 1.0 for name, w in max_data.items()
    #    # Note: 1/w**2 does not converge with states in objective...
    #}
    objective_states = get_state_variable_names(space)

    with open("ic_data.json", "w") as fp:
        json.dump(dae_data, fp)
    with open("setpoint_data.json", "w") as fp:
        json.dump(setpoint, fp)

    flattened_vars = [None, None]
    m = get_model_for_dynamic_optimization(
        sample_points=sample_points,
        parameter_perturbation=disturbance,
        model_params=model_params,
        ic_scalar_data=scalar_data,
        ic_dae_data=dae_data,
        setpoint_data=setpoint,
        objective_weights=weight_data,
        objective_states=objective_states,

        # this argument is a huge hack to get the flattened
        # vars without having to do a bit more work.
        flatten_out=flattened_vars,
    )
    add_constraints_for_missing_variables(m)
    time = m.fs.time
    t0 = time.first()
    scalar_vars, dae_vars = flattened_vars
    initialize_dynamic(m, dae_vars)

    #TODO:
    # Outside the loop:
    # - make plant model
    # - initialize data structure for plant data to initial condition
    #   (one data structure for states, one for controls)
    # Inside the loop:
    # - initialize controller model (with bounds fixed)
    # - solve control problem
    # - extract first control input from controller model,
    #   send to plant and data structure for inputs
    # - simulate plant
    # - extend plant state data structure with results of simulation
    # - update controller initial conditions with final value from plant
    # - update plant initial conditions with final value from plant

    plant_model_params = {
        "horizon": sample_width,
        "tfe_width": tfe_width,
        "ntcp": 1,
        "nxfe": nxfe,
    }
    m_plant = get_nmpc_plant_model(
        parameter_perturbation=disturbance,
        model_params=plant_model_params,
        ic_scalar_data=scalar_data,
        ic_dae_data=dae_data,
        setpoint_data=setpoint,
        objective_weights=weight_data,
        objective_states=objective_states,

        # this argument is a huge hack to get the flattened
        # vars without having to do a bit more work.
        flatten_out=flattened_vars,
    )
    add_constraints_for_missing_variables(m_plant)
    plant_time = m_plant.fs.time

    input_names = [
        "fs.MB.gas_phase.properties[*,0.0].flow_mol",
        "fs.MB.solid_phase.properties[*,1.0].flow_mass",
    ]
    applied_inputs = (
        [t0],
        {name: [m.find_component(name)[t0].value] for name in input_names},
    )
    # TODO: Initialize a planned_inputs sequence

    plant_scalar_vars, plant_dae_vars = flattened_vars
    plant_variables = list(plant_dae_vars)
    plant_variables.append(m_plant.tracking_cost)
    plant_data = initialize_time_series_data(plant_variables, plant_time, t0=t0)
    controller_dae_vars = [
        m.find_component(var.referent) for var in plant_dae_vars
    ]
    controller_variables = list(controller_dae_vars)
    controller_variables.append(m.tracking_cost)

    # Assuming plant and controller have same initial conditions at this point
    for i in range(n_nmpc_samples):
        current_time = nmpc_sample_points[i]
        #
        # Initialze controller
        #
        input_vardata = (
            [m.fs.MB.gas_inlet.flow_mol[t] for t in time if t != t0]
            + [m.fs.MB.solid_inlet.flow_mass[t] for t in time if t != t0]
        )
        # Want an unbounded conversion for simulation.
        m.fs.MB.solid_phase.reactions[:,0.0].OC_conv.setlb(None)
        with TemporarySubsystemManager(
                to_fix=input_vardata,
                to_deactivate=[m.piecewise_constant_constraint],
                ):
            print("Initializing controller by time element...")
            with TIMER.context("elem-init-controller"):
                initialize_by_time_element(m, time, solver)
        m.fs.MB.solid_phase.reactions[:,0.0].OC_conv.setlb(0.89)
        ###

        #
        # Solve controller model
        #
        print("Starting dynamic optimization solve...")
        with TIMER.context("solve dynamic"):
            solver.solve(m, tee=True)
        ###

        #
        # Extract inputs from controller model
        #
        controller_inputs = (
            list(sample_points),
            {
                name: [m.find_component(name)[ts].value for ts in sample_points]
                for name in input_names
            },
        )
        # We have two important input sequences to keep track of. One is the
        # past inputs, actually applied to the plant. The other is the planned
        # inputs.
        # TODO, here:
        # (a) Extract the first input from controller_inputs, use it to extend
        #     the sequence of applied inputs
        # (b) Apply offset to this controller_inputs sequence, rename it
        #     planned_inputs
        # Both of these have an offset applied
        # (c) Should happen first: extract first input from controller_inputs,
        #     apply to plant

        # This is the extracted first input
        plant_inputs = (
            [t0, sample_width],
            {
                name: [values[0], values[1]]
                for name, values in controller_inputs[1].items()
            },
        )

        # Add this extracted first input to the sequence of applied inputs
        applied_inputs[0].append(current_time + sample_width)
        for name, values in plant_inputs[1].items():
            applied_inputs[1][name].append(values[1])

        # Planned_inputs. These will be used in the case we cannot solve
        # a dynamic optimization problem.
        planned_inputs = (
            [t + current_time for t in controller_inputs[0]],
            controller_inputs[1],
        )

        #
        # Sent inputs into plant model
        #
        plant_inputs = interval_data_from_time_series(plant_inputs)
        load_inputs_into_model(m_plant, time, plant_inputs)
        ###

        #
        # Simulate plant model
        #
        print("Initializing plant by time element...")
        with TIMER.context("elem-init-plant"):
            initialize_by_time_element(m_plant, plant_time, solver)
            solver.solve(m_plant, tee=True)
        # Record data
        plant_data = extend_time_series_data(
            plant_data,
            plant_variables,
            plant_time,
            offset=current_time,
        )
        ###

        pyo.Reference(m.fs.MB.solid_phase.properties[:, 0.0].temperature).pprint()
        # indiscriminately shift every time-indexed variable in the
        # model backwards one sample. Inputs, disturbances, everything...
        seen = set()
        for var in controller_dae_vars:
            if id(var[t0]) in seen:
                continue
            else:
                # We need to make sure we don't do this twice for the same
                # vardata. Note that we can encounter the same vardata multiple
                # times due to references.
                seen.add(id(var[t0]))
            for t in time:
                ts = t + sample_width
                idx = time.find_nearest_index(ts, tolerance=1e-8)
                if idx is None:
                    # ts is outside the controller's horizon
                    var[t].set_value(var[time.last()].value)
                else:
                    ts = time.at(idx)
                    var[t].set_value(var[ts].value)
        pyo.Reference(m.fs.MB.solid_phase.properties[:, 0.0].temperature).pprint()

        #
        # Re-initialize plant and controller to new initial conditions
        #
        tf = sample_width
        for i, var in enumerate(plant_dae_vars):
            final_value = var[tf].value
            for t in plant_time:
                var[t].set_value(final_value)
            controller_var = controller_dae_vars[i]
            controller_var[t0].set_value(final_value)
        ###

    plant_fname = "nmpc_plant_data.json"
    with open(plant_fname, "w") as fp:
        json.dump(plant_data, fp)
    # Note that this is not actually all the data I need.
    # I also need the setpoint data.

    input_fname = "nmpc_input_data.json"
    with open(input_fname, "w") as fp:
        json.dump(applied_inputs, fp)

    # Don't plot states in this script. This will be handled by a
    # separate plotting script.
    #extra_states = [
    #    "fs.MB.solid_phase.reactions[*,0.0].OC_conv",
    #    "tracking_cost",
    #]
    #plot_outlet_data(
    #    plant_data,
    #    show=False,
    #    prefix="nmpc_",
    #    extra_states=extra_states,
    #)

    # TODO: Plot inputs over time (from applied_inputs sequence)

    #extra_states = [
    #    pyo.Reference(m.fs.MB.solid_phase.reactions[:,0.0].OC_conv),
    #]
    #plot_outlet_states_over_time(m, show=False, extra_states=extra_states)
    #plot_inputs_over_time(m, input_names, show=False)
    #print(m.tracking_cost.name)
    #for t in m.fs.time:
    #    print(t, pyo.value(m.tracking_cost[t]))
    #print()


if __name__ == "__main__":
    with TIMER.context("main"):
        main()
    print(TIMER)
