"""
Base for deterministic and robust MEA flowsheet optimization workflows.

Originally, and by default, this module provides support for
optimization under a proxy objective; however, the methods
herein provide suffficient customizability to support
other optimization objectives.
"""


from collections import namedtuple
from collections.abc import Iterable
import datetime
import itertools
import logging
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.base.var_like_expression import VarLikeExpressionData
from idaes.core.util.model_statistics import degrees_of_freedom
import pyomo.environ as pyo
from pyomo.common.collections import Bunch, ComponentMap, ComponentSet
from pyomo.common.log import LogStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import capture_output
import pyomo.contrib.pyros as pyros
from pyomo.core.base import VarData, ParamData, ExpressionData, ObjectiveData
from pyomo.opt import SolverResults, TerminationCondition

from confidence_ellipsoid import (
    get_conf_lvl_plot_label,
    get_pyros_ellipsoidal_set,
)
from flowsheets.combined_flowsheet_withcosting_CCS import MEACombinedFlowsheet
from flowsheets.mea_properties import switch_liquid_to_parmest_params
from heatmap import (
    evaluate_design_heatmap,
    plot_capture_target_sensitivity,
    plot_confidence_level_sensitivity,
    plot_heatmap_scatter_results,
    plot_heatmap_feasibility_results,
    plot_heatmap_feasibility_target_response,
    process_heatmap_results,
    plot_quantity_distributions,
    HeatmapQuantityInfo,
)
from model_utils import (
    get_cons_with_component,
    get_exprs_with_component,
    get_uniform_grid,
    get_unused_state_vars,
    get_active_inequality_constraints,
    get_active_var_bounds,
    get_state_vars_in_active_cons,
    get_vars_with_close_bounds,
    SolverWithBackup,
    SolverWithScaling,
    get_co2_capture_str,
    substitute_uncertain_params,
    strip_state_var_bounds,
    log_main_dependency_module_info,
    time_code,
    float_to_str,
    get_solver,
)
from plotting_utils import (
    annotate_heatmap,
    DEFAULT_MPL_RC_PARAMS,
    heatmap,
    set_nonoverlapping_fixed_xticks,
    wrap_quantity_str,
)


default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)
default_logger.addHandler(logging.StreamHandler())

# default solver for simulation and optimization models
default_solver = "ipopt"
default_solver_options = {
    'bound_push': 1e-4,
    'nlp_scaling_method': 'user-scaling',
    'linear_solver': 'ma57',
    'OF_ma57_automatic_scaling': 'yes',
    'max_iter': 1000,
    'constr_viol_tol': 1e-8,
    'halt_on_ampl_error': 'no',
    "mu_init": 1e-2,
}


def build_square_flowsheet_model(
        axial_coordinate_grid=None,
        progress_logger=None,
        solve_square_model=True,
        initialization_solver=None,
        initialization_solver_options=None,
        load_from_json=None,
        save_to_json=None,
        solver=None,
        solver_options=None,
        flowsheet_process_block_constructor=None,
        ):
    """
    Build square flowsheet model.
    """
    # resolve arguments
    if axial_coordinate_grid is None:
        axial_coordinate_grid = get_uniform_grid(nfe=40)
    if progress_logger is None:
        progress_logger = default_logger

    if flowsheet_process_block_constructor is None:
        flowsheet_process_block_constructor = MEACombinedFlowsheet

    progress_logger.info("Building square flowsheet model")

    # Build flowsheet
    m = pyo.ConcreteModel()
    m.fs = flowsheet_process_block_constructor(
        time_set=[0],
        absorber_finite_element_set=axial_coordinate_grid,
        stripper_finite_element_set=axial_coordinate_grid,
    )

    switch_liquid_to_parmest_params(
        m.fs.stripper_section.liquid_properties, ions=True
    )
    switch_liquid_to_parmest_params(
        m.fs.stripper_section.liquid_properties_no_ions, ions=False
    )
    switch_liquid_to_parmest_params(
        m.fs.absorber_section.liquid_properties, ions=True
    )
    switch_liquid_to_parmest_params(
        m.fs.absorber_section.liquid_properties_no_ions, ions=False
    )

    # initialize square model
    # Set Initial guesses for inlets to sub-flowsheets
    m.fs.absorber_section.makeup_mixer.h2o_makeup.flow_mol.fix(1300)  # mol/sec
    iscale.calculate_scaling_factors(m.fs)

    if initialization_solver is None:
        initialization_solver_options = {
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-8,
            'halt_on_ampl_error': 'no',
        }
    m.fs.initialize_build(
        outlvl=idaeslog.DEBUG,
        optarg=initialization_solver_options,
        load_from=load_from_json,
        save_to=save_to_json,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ SIMULATION ~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve flowsheet, to a better initial point for optimization
    # Also add lean-rich heat exchanger, condenser and reboiler area
    # calculation constraints
    progress_logger.info("Increasing column diameters...")

    m.fs.absorber_section.absorber.diameter_column.fix(14)
    m.fs.stripper_section.stripper.diameter_column.fix(6.5)

    # fix column volume expressions
    m.fs.stripper_section.volume_column_withheads.set_value(
        math.pi * (
            m.fs.stripper_section.stripper.diameter_column ** 2
            * m.fs.stripper_section.stripper.length_column
        ) / 4
        + math.pi * m.fs.stripper_section.stripper.diameter_column ** 3 / 6
    )
    m.fs.absorber_section.volume_column_withheads.set_value(
        math.pi * (
            m.fs.absorber_section.absorber.diameter_column ** 2
            * m.fs.absorber_section.absorber.length_column
        ) / 4
        + math.pi * m.fs.absorber_section.absorber.diameter_column ** 3 / 6
    )

    if solver is None and solver_options is None:
        solver = None
        solver_options = {
            'bound_push': 1e-4,
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 1000,
            'constr_viol_tol': 1e-8,
            'halt_on_ampl_error': 'no',
            'mu_init': 1e-2,
        }
    solver_obj = get_solver(solver, solver_options)

    results = solver_obj.solve(m, tee=True)
    pyo.assert_optimal_termination(results)

    progress_logger.info("Adding heat exchanger components")
    m.fs.stripper_section.add_condenser_reboiler_performance_equations()

    assert degrees_of_freedom(m) == 0
    results = solver_obj.solve(m, tee=True)
    pyo.assert_optimal_termination(results)

    print("\n-------- Absorber Simulation Results --------")
    m.fs.absorber_section.print_column_design_parameters()

    print("\n-------- Stripper Simulation Results --------")
    m.fs.stripper_section.print_column_design_parameters()

    return m


def get_first_stage_vars(model):
    """
    Return list of first-stage variables of deterministic
    flowsheet model.
    """
    return [
        model.fs.absorber_section.absorber.length_column,
        model.fs.absorber_section.absorber.diameter_column,
        model.fs.stripper_section.stripper.length_column,
        model.fs.stripper_section.stripper.diameter_column,
        model.fs.absorber_section.lean_rich_heat_exchanger.area,
        model.fs.absorber_section.rich_solvent_pump.max_work,
        model.fs.stripper_section.condenser.area,
        model.fs.stripper_section.reboiler.area,
        model.fs.condenser_cw_temperature_outlet_excess_temp,
    ]


def get_second_stage_vars(model):
    """
    Return list of second-stage variables of deterministic
    flowsheet model.
    """
    return [
        model.fs.mea_recirculation_rate[0],
        model.fs.stripper_section.reboiler.steam_flow_mol[0],
        model.fs.stripper_section.condenser.cooling_flow_mol[0],
    ]


def get_dof_vars(model):
    """
    Get all DOF variables of the deterministic flowsheet model.
    """
    return get_first_stage_vars(model) + get_second_stage_vars(model)


def get_uncertain_params(model):
    """
    Get uncertain parameters of the flowsheet model.
    """
    return [
        model.fs.k_eq_b_bicarbonate,
        model.fs.k_eq_b_carbamate,
        model.fs.lwm_coeff_1,
        model.fs.lwm_coeff_2,
        model.fs.lwm_coeff_3,
        model.fs.lwm_coeff_4,
    ]


def get_uncertain_param_to_fixed_var_map(model):
    """
    Get mapping from uncertain params to list of
    fixed Vars they represent.
    """
    # maps for substituting mutable vars for fixed params
    absorber_blk = model.fs.absorber_section
    stripper_blk = model.fs.stripper_section
    section_blocks = [absorber_blk, stripper_blk]

    param_to_var_map = ComponentMap(
        (param, [])
        for param in get_uncertain_params(model)
    )
    for blk in section_blocks:
        param_to_var_map.setdefault(model.fs.k_eq_b_bicarbonate, []).append(
            blk.liquid_properties.reaction_bicarbonate.k_eq_coeff_2,
        )
        param_to_var_map.setdefault(model.fs.k_eq_b_carbamate, []).append(
            blk.liquid_properties.reaction_carbamate.k_eq_coeff_2,
        )
        param_to_var_map.setdefault(model.fs.lwm_coeff_1, []).append(
            blk.liquid_properties.CO2.lwm_coeff_1,
        )
        param_to_var_map.setdefault(model.fs.lwm_coeff_2, []).append(
            blk.liquid_properties.CO2.lwm_coeff_2,
        )
        param_to_var_map.setdefault(model.fs.lwm_coeff_3, []).append(
            blk.liquid_properties.CO2.lwm_coeff_3,
        )
        param_to_var_map.setdefault(model.fs.lwm_coeff_4, []).append(
            blk.liquid_properties.CO2.lwm_coeff_4,
        )

    return param_to_var_map


def get_fixed_var_to_uncertain_param_map(model):
    """
    Get mapping from fixed Vars representing uncertain
    quantities to corresponding mutable Params representing
    those quantities.
    """
    param_to_varlist_map = get_uncertain_param_to_fixed_var_map(model)
    var_to_param_map = ComponentMap()

    for param, varlist in param_to_varlist_map.items():
        var_to_param_map.update((var, param) for var in varlist)

    return var_to_param_map


def add_proxy_objective_components(model, progress_logger):
    """
    Add proxy objective components to flowsheet model.
    """
    # break down objective to facilitate analysis of contributions
    # of the various capital and operating cost terms
    # after optimization
    model.fs.steam_price = pyo.Param(initialize=1e-1, mutable=True)
    model.fs.absorber_capex_cost = pyo.Expression(
        expr=model.fs.absorber_section.volume_column_withheads / 8
    )
    model.fs.rich_pump_capex_cost = pyo.Expression(
        expr=model.fs.absorber_section.rich_solvent_pump.max_work / 2e2,
    )
    model.fs.lean_pump_capex_cost = pyo.Expression(
        expr=model.fs.absorber_section.lean_solvent_pump.max_work / 2e2,
    )
    model.fs.stripper_capex_cost = pyo.Expression(
        expr=model.fs.stripper_section.volume_column_withheads
    )
    model.fs.lean_rich_hex_capex_cost = pyo.Expression(
        expr=model.fs.absorber_section.lean_rich_heat_exchanger.area / 20
    )
    model.fs.reboiler_capex_cost = pyo.Expression(
        expr=model.fs.stripper_section.reboiler.area / 30
    )
    model.fs.condenser_capex_cost = pyo.Expression(
        expr=model.fs.stripper_section.condenser.area / 10
    )

    model.fs.rich_pump_opex_cost = pyo.Expression(
        expr=(
            model.fs.absorber_section.rich_solvent_pump.control_volume.work[0]
            / 3e2
        ),
    )
    model.fs.lean_pump_opex_cost = pyo.Expression(
        expr=(
            model.fs.absorber_section.lean_solvent_pump.control_volume.work[0]
            / 1e4
        ),
    )
    model.fs.mea_recirculation_opex_cost = pyo.Expression(
        expr=model.fs.mea_recirculation_rate[0] / 15
    )
    model.fs.h2o_makeup_opex_cost = pyo.Expression(
        expr=model.fs.makeup.flow_mol[0] / 10,
    )
    model.fs.reboiler_steam_opex_cost = pyo.Expression(
        expr=(
            model.fs.stripper_section.reboiler.steam_flow_mol[0]
            * model.fs.steam_price
        ),
    )
    model.fs.condenser_cw_opex_cost = pyo.Expression(
        expr=model.fs.stripper_section.condenser.cooling_flow_mol[0] * 3e-3
    )
    model.fs.condenser_h2o_vapor_mole_frac_opex_cost = pyo.Expression(
        expr=(
            model.fs.stripper_section.condenser
            .vapor_phase.properties_out[0].mole_frac_comp["H2O"]
        ) * 1e3,
    )

    model.fs.condenser_cw_temperature_outlet_opex_cost = pyo.Expression(
        expr=2 * model.fs.condenser_cw_temperature_outlet_excess_temp
    )

    # allow easier tracking of capex and opex cost terms
    capex_cost_terms = [
        model.fs.absorber_capex_cost,
        model.fs.rich_pump_capex_cost,
        model.fs.stripper_capex_cost,
        model.fs.lean_rich_hex_capex_cost,
        model.fs.reboiler_capex_cost,
        model.fs.condenser_capex_cost,
    ]
    opex_cost_terms = [
        model.fs.mea_recirculation_opex_cost,
        model.fs.h2o_makeup_opex_cost,
        model.fs.rich_pump_opex_cost,
        model.fs.reboiler_steam_opex_cost,
        model.fs.condenser_cw_opex_cost,
        model.fs.condenser_h2o_vapor_mole_frac_opex_cost,
        model.fs.condenser_cw_temperature_outlet_opex_cost,
    ]
    model.fs.capex_cost_terms = pyo.Expression(
        [expr.local_name for expr in capex_cost_terms],
        initialize={expr.local_name: expr / 1000 for expr in capex_cost_terms},
    )
    model.fs.opex_cost_terms = pyo.Expression(
        [expr.local_name for expr in opex_cost_terms],
        initialize={expr.local_name: expr / 1000 for expr in opex_cost_terms},
    )

    model.fs.capex_cost = pyo.Expression(
        expr=sum(model.fs.capex_cost_terms.values()),
    )
    model.fs.opex_cost = pyo.Expression(
        expr=sum(model.fs.opex_cost_terms.values()),
    )
    model.fs.obj = pyo.Objective(
        expr=model.fs.capex_cost + model.fs.opex_cost,
        sense=pyo.minimize,
    )

    logstream = LogStream(logging.DEBUG, progress_logger)
    with capture_output(logstream):
        progress_logger.debug("PROXY COST TERMS AT SIMULATION SOLUTION:")
        model.fs.obj.display(ostream=logstream)
        model.fs.capex_cost.display(ostream=logstream)
        model.fs.opex_cost.display(ostream=logstream)
        model.fs.capex_cost_terms.display(ostream=logstream)
        model.fs.opex_cost_terms.display(ostream=logstream)

        progress_logger.debug("CAPEX breakdown at initial point")
        for key, val in model.fs.capex_cost_terms.items():
            progress_logger.debug(
                f" {key:50s} = "
                f"{pyo.value(100 * val / model.fs.capex_cost):.2f}% "
            )
        progress_logger.debug("OPEX breakdown at initial point")
        for key, val in model.fs.opex_cost_terms.items():
            progress_logger.debug(
                f" {key:50s} = "
                f"{pyo.value(100 * val / model.fs.opex_cost):.2f}% "
            )


def add_deterministic_optimization_components(
        model,
        co2_capture_target,
        progress_logger=None,
        ):
    """
    Add components for extending square model to deterministic
    optimization model.

    Parameters
    ----------
    model : ConcreteModel
        Square flowsheet model, with feasible solution loaded.
    co2_capture_target : float or 'simulation'
        CO2 capture target. If 'simulation' is specified,
        then the capture target is set to the value of the CO2
        capture rate at the solution to the input square model.
    progress_logger : logging.Logger, optional
        Progress logger.
    """
    if progress_logger is None:
        progress_logger = default_logger
    fs = model.fs

    # add bounds on reboiler and condenser utility flows
    fs.stripper_section.reboiler.steam_flow_mol[0].setlb(0)
    fs.stripper_section.condenser.cooling_flow_mol[0].setlb(0)

    # bounds for column length/diameter ratio
    fs.absorber_section.absorber.HDratio_lower_bound = pyo.Constraint(
        expr=(
            1.2*fs.absorber_section.absorber.diameter_column
            <= fs.absorber_section.absorber.length_column
        ),
    )
    fs.absorber_section.absorber.HDratio_upper_bound = pyo.Constraint(
        expr=(
            30*fs.absorber_section.absorber.diameter_column
            >= fs.absorber_section.absorber.length_column
        ),
    )
    fs.stripper_section.stripper.HDratio_lower_bound = pyo.Constraint(
        expr=(
            1.2*fs.stripper_section.stripper.diameter_column
            <= fs.stripper_section.stripper.length_column
        ),
    )
    fs.stripper_section.stripper.HDratio_upper_bound = pyo.Constraint(
        expr=(
            30*fs.stripper_section.stripper.diameter_column
            >= fs.stripper_section.stripper.length_column
        ),
    )

    add_flood_fraction_constraints(
        column_blk=fs.absorber_section.absorber,
        flood_lb=0.5,
        flood_ub=0.8,
    )
    add_flood_fraction_constraints(
        column_blk=fs.stripper_section.stripper,
        flood_lb=0.5,
        flood_ub=0.8,
    )

    # pump constraints
    pump_blocks = [
        fs.absorber_section.rich_solvent_pump,
        fs.absorber_section.lean_solvent_pump,
    ]
    for pump_blk in pump_blocks:
        pump_work_var = pump_blk.control_volume.work[0]
        pump_blk.max_work = pyo.Var(
            initialize=pyo.value(pump_work_var),
            bounds=pump_work_var.bounds,
        )
        pump_blk.max_work_con = pyo.Constraint(
            expr=pump_work_var - pump_blk.max_work <= 0
        )
    fs.absorber_section.lean_solvent_pump.max_work.fix(0)

    # to facilitate model re-use, we make the
    # co2 capture target a mutable parameter
    if co2_capture_target == "simulation":
        co2_capture_target = pyo.value(
            fs.absorber_section.absorber.co2_capture[0]
        )
        progress_logger.debug(
            f"Simulation capture target ({co2_capture_target}%) "
            "specified via `co2_capture_target='simulation'."
        )

    fs.co2_capture_target = pyo.Param(
        initialize=co2_capture_target,
        mutable=True,
        within=pyo.Reals,
    )
    fs.CO2_lower_bound = pyo.Constraint(
        expr=(
            fs.absorber_section.absorber.co2_capture[0]
            >= fs.co2_capture_target
        ),
    )
    iscale.constraint_scaling_transform(
      fs.CO2_lower_bound, 1e-2
    )

    # process-side outlet temperatures are state variables
    # in deterministic model.
    # for now, the utility-side outlet temperatures
    # remain fixed
    fs.stripper_section.reboiler.bottoms.temperature.unfix()
    fs.stripper_section.condenser.reflux.temperature.unfix()

    # Still want this one fixed bc no good way
    # to choose MEA/water ratio otherwise
    fs.h2o_mea_ratio.fix()

    # CW outlet temperatures in excess of 40 C are to be penalized
    # in the proxy objective
    condenser_cw_outlet_temp_var = (
        model.fs.stripper_section.condenser.cooling_water_temperature_outlet[0]
    )
    model.fs.condenser_cw_temperature_outlet_excess_temp = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=max(
            0, pyo.value(condenser_cw_outlet_temp_var - (40 + 273.15))
        ),
    )
    model.fs.condenser_cw_temperature_penalty_constraint = pyo.Constraint(
        expr=(
            model.fs.condenser_cw_temperature_outlet_excess_temp
            - (condenser_cw_outlet_temp_var - (40 + 273.15))
            >= 0
        )
    )

    # unfix DOF variables.
    # log if they were unfixed in square model
    dof_vars = get_dof_vars(model)
    for var in dof_vars:
        if not var.fixed:
            progress_logger.debug(
                f"DOF variable {var.name!r} unfixed in square model "
                f"used for initialization."
            )
        var.unfix()

    # unfix the stripper HEX utility outlet temperatures
    model.fs.stripper_section.reboiler.condensate_temperature_outlet.unfix()
    model.fs.stripper_section.reboiler.condensate_temperature_outlet.setlb(
        105 + 273.15
    )
    model.fs.stripper_section.reboiler.condensate_temperature_outlet.setub(
        151 + 273.15
    )
    fs.stripper_section.condenser.cooling_water_temperature_outlet.unfix()

    condenser = model.fs.stripper_section.condenser

    # Algorithmic bound to avoid lower quality
    # local optima with condenser area 0
    condenser.area.setlb(10)

    add_proxy_objective_components(model, progress_logger)


def add_flood_fraction_constraints(column_blk, flood_lb=0.5, flood_ub=0.8):
    """
    Add flooding fraction constraints.
    """
    @column_blk.Constraint(
        column_blk.parent_block().time,
        column_blk.vapor_phase.length_domain,
    )
    def LB_flood_ratio(blk, t, x):
        if x == column_blk.vapor_phase.length_domain.first():
            return pyo.Constraint.Skip
        return blk.flood_fraction[t, x] >= flood_lb

    @column_blk.Constraint(
        column_blk.parent_block().time,
        column_blk.vapor_phase.length_domain,
    )
    def UB_flood_ratio(blk, t, x):
        if x == column_blk.vapor_phase.length_domain.first():
            return pyo.Constraint.Skip
        return blk.flood_fraction[t, x] <= flood_ub


def _activate_flooding_constraints(indexed_flood_con, keep_cons_at_idxs=None):
    """
    Activate desired members of indexed column
    flooding fraction (lower/upper) bound constraints,
    and deactivate the others.

    Parameters
    ----------
    indexed_flood_con : IndexedConstraint
        Indexed lower or upper bound flooding constraint upon
        which to act.
    keep_lb_flooding_at_idxs : None or iterable of int, optional
        Integer indices of ``list(indexed_flood_con.keys())``
        corresponding to members of ``indexed_flood_con`` to
        be activated. All other members of ``indexed_flood_con``
        are deactivated.
    """
    # if this assertion fails, may need to update this method
    assert indexed_flood_con.index_set().first() == (0, 0)

    if keep_cons_at_idxs is None:
        keep_cons_at_idxs = set(
            idx for idx, (time, pos)
            in enumerate(indexed_flood_con.index_set())
        )
    else:
        keep_cons_at_idxs = set(keep_cons_at_idxs)

    for idx, (time, pos) in enumerate(indexed_flood_con.index_set()):
        if (time, pos) in indexed_flood_con.keys():
            # some indices (such as 0, 0) may have been skipped
            if idx in keep_cons_at_idxs:
                indexed_flood_con[time, pos].activate()
            else:
                indexed_flood_con[time, pos].deactivate()


def activate_flooding_constraints(
        column_blk,
        keep_lb_flooding_at_idxs=None,
        keep_ub_flooding_at_idxs=None,
        ):
    """
    Activate desired flooding constraints of column block,
    and deactivate all other flooding constraints.
    """
    _activate_flooding_constraints(
        indexed_flood_con=column_blk.LB_flood_ratio,
        keep_cons_at_idxs=keep_lb_flooding_at_idxs,
    )
    _activate_flooding_constraints(
        indexed_flood_con=column_blk.UB_flood_ratio,
        keep_cons_at_idxs=keep_ub_flooding_at_idxs,
    )


def build_deterministic_flowsheet_model(
        co2_capture_target,
        *square_model_args,
        progress_logger=None,
        keep_state_var_bounds=True,
        det_init_solver=None,
        det_init_solver_options=None,
        keep_absorber_lb_flooding_at_idxs=None,
        keep_absorber_ub_flooding_at_idxs=None,
        keep_stripper_lb_flooding_at_idxs=None,
        keep_stripper_ub_flooding_at_idxs=None,
        flowsheet_model_data_type=None,
        flowsheet_model_data_kwargs=None,
        **square_model_kwargs,
        ):
    """
    Build and initialize a deterministic flowsheet model instance.

    Returns
    -------
    ConcreteModel
        Flowsheet model.
    """
    if progress_logger is None:
        progress_logger = default_logger
    if flowsheet_model_data_type is None:
        flowsheet_model_data_type = FlowsheetModelData
    if flowsheet_model_data_kwargs is None:
        flowsheet_model_data_kwargs = dict()

    square_model_builder = (
        flowsheet_model_data_type.build_square_flowsheet_model
    )

    timer = HierarchicalTimer()
    with time_code(timer, "main"):
        progress_logger.info("Building initial square model...")
        with time_code(timer, square_model_builder.__name__):
            model = square_model_builder(
                *square_model_args,
                progress_logger=progress_logger,
                **square_model_kwargs,
            )

        progress_logger.info("Adding generic optimization components...")
        with time_code(
                timer, add_deterministic_optimization_components.__name__
                ):
            add_deterministic_optimization_components(
                model,
                co2_capture_target=co2_capture_target,
                progress_logger=progress_logger,
            )

        progress_logger.info("Ascertaining 'unused' state variables...")
        with time_code(timer, get_unused_state_vars.__name__):
            unused_vars = get_unused_state_vars(
                model=model,
                dof_vars=get_dof_vars(model),
                include_fixed=False,
            )
        if unused_vars:
            unused_vars_str = "\n ".join(
                f"{var.name!r}" for var in unused_vars
            )
            progress_logger.warning(
                f"Found {len(unused_vars)} unused, unfixed variables in "
                f"square model:\n {unused_vars_str} "
                f"\nFixing for now. Check square model for structural issues "
                "with IDAES diagnostics toolbox."
            )
        for var in unused_vars:
            var.fix()

        vars_with_close_bounds = get_vars_with_close_bounds(
            model, tol=1e-4
        )
        if vars_with_close_bounds:
            progress_logger.warning(
                "The following variables have bounds which are "
                f"equal up to absolute tolerance {1e-4:e}"
            )
            progress_logger.warning(
                "\n ".join(var.name for var in vars_with_close_bounds)
            )

        # removing state var bounds may be helpful for PyROS
        if not keep_state_var_bounds:
            with time_code(timer, strip_state_var_bounds.__name__):
                strip_state_var_bounds(
                    model=model,
                    dof_vars=get_dof_vars(model),
                    vars_in_active_cons_only=False,
                    include_fixed=False,
                )

        # Some Vars that are uninvolved in the NLP are uninitialized
        # However, this causes problems for the scaled model back
        # propagation, so we now initialize them to placeholder value
        for var in model.component_data_objects(pyo.Var):
            if var.value is None:
                var.set_value(0)

        # switch on/off flooding constraints as desired
        progress_logger.info("Treating flooding constraints...")
        with time_code(timer, activate_flooding_constraints.__name__):
            activate_flooding_constraints(
                model.fs.absorber_section.absorber,
                keep_lb_flooding_at_idxs=keep_absorber_lb_flooding_at_idxs,
                keep_ub_flooding_at_idxs=keep_absorber_ub_flooding_at_idxs,
            )
        with time_code(timer, activate_flooding_constraints.__name__):
            activate_flooding_constraints(
                model.fs.stripper_section.stripper,
                keep_lb_flooding_at_idxs=keep_stripper_lb_flooding_at_idxs,
                keep_ub_flooding_at_idxs=keep_stripper_ub_flooding_at_idxs,
            )

        # add class-specific configurable modifications to model
        # (e.g. modification of objective, a new objective, etc.)
        # NOTE: proxy objective and associated components
        #       have already been added and activated
        progress_logger.info("Performing type-specific model adjustments")
        with time_code(
                timer, flowsheet_model_data_type.tune_flowsheet_model.__name__,
                ):
            flowsheet_model_data_type.tune_flowsheet_model(model)

        progress_logger.info("Substituting param components...")
        with time_code(
                timer, declare_and_substitute_uncertain_params.__name__,
                ):
            declare_and_substitute_uncertain_params(model, progress_logger)

    progress_logger.info("All done building deterministic flowsheet model.")
    progress_logger.info(f"Timing stats:\n{timer}")
    progress_logger.debug("Initial point stats: ")
    initial_model_data = flowsheet_model_data_type(
        original_model=model,
        **flowsheet_model_data_kwargs,
    )
    initial_res = initial_model_data.create_flowsheet_results_object(
        solver_results=None,
    )
    progress_logger.debug(
        initial_res
        .to_series(include_detailed_stream_results=False)
        .to_string()
    )

    return model


class FlowsheetModelData:
    """
    Flowsheet model data.

    Facilitate management of deterministic flowsheet
    model and its scaled counterpart.
    """
    def __init__(self, original_model, scaled_model=None):
        """Initialize self (see class docstring).

        """
        self.original_model = original_model
        if scaled_model is None:
            self.scaled_model = self._create_scaled_model()

    @staticmethod
    def build_square_flowsheet_model(*args, **kwargs):
        """
        Build square flowsheet model.
        """
        return build_square_flowsheet_model(*args, **kwargs)

    def clone(self):
        """
        Create a clone (deepcopy) of self.
        """
        return self.__class__(
            original_model=self.original_model.clone(),
            scaled_model=None,
        )

    def _create_scaled_model(self):
        """
        Create scaled model based on scaling factor suffixes
        of `self.original_model`.
        """
        transformation = pyo.TransformationFactory("core.scale_model")

        # NOTE: scaling factor suffixes of the scaled model
        #       are deactivated automatically.
        #       the suffixes of the original remain active.
        return transformation.create_using(
            self.original_model,
            rename=False,
        )

    def _pre_solve(self, solver, *args, load_solutions=True, **kwargs):
        """
        'Pre-solve' the model by invoking a solver
        on the model with the condenser area fixed.
        """
        # condenser area may have been scaled
        orig_name_to_scaled_comp_map = {
            orig_name: scaled_comp
            for scaled_comp, orig_name in
            self.scaled_model.scaled_component_to_original_name_map.items()
        }
        scaled_condenser_area_var = orig_name_to_scaled_comp_map.get(
            self.original_model.fs.stripper_section.condenser.area.name,
            self.scaled_model.fs.stripper_section.condenser.area,
        )
        scaled_condenser_area_var.fix()

        res = solver.solve(
            self.scaled_model,
            *args,
            load_solutions=load_solutions,
            **kwargs,
        )
        if load_solutions:
            self._load_solution(res)

        scaled_condenser_area_var.unfix()

        return res

    @staticmethod
    def create_heatmap_solver_wrapper(solver, logger=None):
        """
        Create solver wrapper for heatmap workflows.

        Parameters
        ----------
        solver : Pyomo solver object
            Solver to be wrapped. This is usually an instance
            of `SolverWithBackup`.
        logger : logging.Logger or None, optional
            Logger with which to equip the wrapper.

        Returns
        -------
        SolverWithScaling
            Solver wrapper.
        """
        return SolverWithScaling(solver, logger)

    heatmap_initialization_func = None

    def log_proxy_objective_breakdown(self, logger, level=logging.DEBUG):
        """
        Log breakdown of proxy costs.
        """
        fs_blk = self.original_model.fs
        total_proxy_cost = pyo.value(fs_blk.obj)
        logger.log(level, "Proxy cost breakdown:")
        logger.log(level, f"Total proxy cost : {total_proxy_cost:.2f}")

        # CAPEX breakdown
        for ctype in ["capex", "opex"]:
            ctype_cost_expr = fs_blk.find_component(f"{ctype}_cost")
            total_ctype_cost = pyo.value(ctype_cost_expr)
            ctype_over_total = total_ctype_cost / total_proxy_cost
            logger.log(
                level=level,
                msg=(
                    f"  Proxy {ctype.upper()} cost : "
                    f"{pyo.value(total_ctype_cost):6.2f} "
                    f"({100 * ctype_over_total:.2f}% total)"
                ),
            )
            ctype_cost_terms = fs_blk.find_component(f"{ctype}_cost_terms")
            for key, term_expr in ctype_cost_terms.items():
                cost_term_val = pyo.value(term_expr)
                cost_term_over_ctype = cost_term_val / total_ctype_cost
                cost_term_over_proxy = cost_term_val / total_proxy_cost
                logger.log(
                    level=level,
                    msg=(
                        f"    {key:50s} : {cost_term_val:6.2f} "
                        f"({100 * cost_term_over_ctype:6.2f}% {ctype.upper()}"
                        f", {100 * cost_term_over_proxy:6.2f}% total)"
                    ),
                )

    def log_objective_breakdown(self, logger, level=logging.DEBUG):
        """
        Log breakdown of model objective.
        """
        return self.log_proxy_objective_breakdown(logger, level=logging.DEBUG)

    def solve_deterministic(
            self,
            solver,
            logger=None,
            keep_model_in_results=True,
            *args,
            **kwargs,
            ):
        """
        Solve deterministic model contained in self.

        If the termination condition is acceptable,
        load the solution returned by the optimizer
        and log a breakdown of the proxy objective terms.

        Returns
        -------
        MEAFlowsheetResults
            Solve results.
        """
        # skip pre-solve
        preopt_res = SolverResults()
        preopt_res.solver.termination_condition = (
            TerminationCondition.other
        )
        preopt_res.solver.message = "Presolve bypassed."
        setattr(
            preopt_res.solver,
            SolverWithBackup.TOTAL_WALL_TIME_ATTR,
            0,
        )

        res = solver.solve(
            self.scaled_model,
            load_solutions=False,
            *args,
            **kwargs,
        )

        if pyo.check_optimal_termination(res):
            logger.info(
                "Successfully solve deterministic flowsheet model to "
                "an optimality status. "
                f"Solver termination:\n {res.solver}"
            )
            self._load_solution(res)
        else:
            logger.warning(
                "Could not solve deterministic flowsheet model to "
                "an optimality status. "
                f"Solver termination:\n {res.solver}"
                "Any solution returned by the optimizer will not be loaded."
            )

        return self.create_flowsheet_results_object(
            solver_results=res,
            presolver_results=preopt_res,
            keep_model=keep_model_in_results,
        )

    @staticmethod
    def get_model_first_stage_variables(model):
        """Get first-stage variables of a model."""
        return list(get_first_stage_vars(model))

    @staticmethod
    def get_model_second_stage_variables(model):
        """Get second-stage variables of a model."""
        return list(get_second_stage_vars(model))

    @classmethod
    def get_model_dof_variables(cls, model):
        """
        Get DOF variables of a model.
        """
        return (
            cls.get_model_first_stage_variables(model)
            + cls.get_model_second_stage_variables(model)
        )

    @staticmethod
    def get_model_uncertain_params(model):
        """Get uncertain parameters of a model."""
        return list(get_uncertain_params(model))

    def get_first_stage_variables(self):
        """Get first-stage variables of the original model."""
        return self.get_model_first_stage_variables(self.original_model)

    def get_second_stage_variables(self):
        """Get second-stage variables of the original model."""
        return self.get_model_second_stage_variables(self.original_model)

    def get_dof_variables(self):
        """Get DOF variables of the original model."""
        return self.get_model_dof_variables(self.original_model)

    def get_uncertain_params(self):
        """Get uncertain parameters variables of the original model."""
        return self.get_model_uncertain_params(self.original_model)

    def get_scaled_model_components(self, original_model_components):
        """
        Get scaled model components.

        Yields
        ------
        Component or _ComponentData
            Scaled model component.
        """
        orig_name_to_scaled_comp_map = {
            orig_name: scaled_comp
            for scaled_comp, orig_name in
            self.scaled_model.scaled_component_to_original_name_map.items()
        }
        for comp in original_model_components:
            yield orig_name_to_scaled_comp_map.get(
                comp.name,
                self.scaled_model.find_component(comp.name),
            )

    def get_scaled_first_stage_variables(self):
        """
        Get first-stage variables of scaled model.
        """
        return list(self.get_scaled_model_components(
            self.get_first_stage_variables()
        ))

    def get_scaled_second_stage_variables(self):
        """
        Get second-stage variables of scaled model.
        """
        return list(self.get_scaled_model_components(
            self.get_second_stage_variables()
        ))

    def get_scaled_dof_variables(self):
        """
        Get DOF variables of scaled model.
        """
        return list(self.get_scaled_model_components(self.get_dof_variables()))

    def get_scaled_uncertain_params(self):
        """
        Get uncertain parameters of scaled model.
        """
        return list(self.get_scaled_model_components(
            self.get_uncertain_params()
        ))

    def get_scaled_ellipsoidal_set(self, ellipsoidal_set):
        """
        Get scaled PyROS ellipsoidal set.
        """
        return ellipsoidal_set

    @staticmethod
    def add_pyros_separation_priorities(
            model,
            prioritize_absorber_lb_flood_idxs=None,
            prioritize_absorber_ub_flood_idxs=None,
            prioritize_stripper_lb_flood_idxs=None,
            prioritize_stripper_ub_flood_idxs=None,
            ):
        """
        Add PyROS performance constraint separation priority order
        specifications to original model of `self`.
        """
        sep_priority = model.pyros_separation_priority = pyo.Suffix(
            direction=pyo.Suffix.LOCAL,
            datatype=None,
        )

        # adjust priorities for flooding constraints
        if prioritize_absorber_lb_flood_idxs is None:
            prioritize_absorber_lb_flood_idxs = []
        if prioritize_absorber_ub_flood_idxs is None:
            prioritize_absorber_ub_flood_idxs = []
        if prioritize_stripper_lb_flood_idxs is None:
            prioritize_stripper_lb_flood_idxs = []
        if prioritize_stripper_ub_flood_idxs is None:
            prioritize_stripper_ub_flood_idxs = []

        column_blks = [
            model.fs.absorber_section.absorber,
            model.fs.stripper_section.stripper,
        ]
        flood_priority_lists = [
            (
                list(prioritize_absorber_lb_flood_idxs),
                list(prioritize_absorber_ub_flood_idxs),
            ),
            (
                list(prioritize_stripper_lb_flood_idxs),
                list(prioritize_stripper_ub_flood_idxs),
            ),
        ]
        flood_priority_zip = zip(column_blks, flood_priority_lists)
        for col_blk, (lb_idxs, ub_idxs) in flood_priority_zip:
            lb_flood_cons = col_blk.LB_flood_ratio
            ub_flood_cons = col_blk.UB_flood_ratio
            for lbidx, (lbt, lbx) in enumerate(lb_flood_cons.index_set()):
                if lbidx in lb_idxs:
                    sep_priority[lb_flood_cons[lbt, lbx]] = 1
                elif lbidx > 0:
                    sep_priority[lb_flood_cons[lbt, lbx]] = None
            for ubidx, (ubt, ubx) in enumerate(ub_flood_cons.index_set()):
                if ubidx in ub_idxs:
                    sep_priority[ub_flood_cons[ubt, ubx]] = 1
                elif ubidx > 0:
                    sep_priority[ub_flood_cons[ubt, ubx]] = None

        sep_priority[
            model.fs.stripper_section.reboiler.condensate_temperature_outlet[0]
        ] = 1
        sep_priority[
            model.fs.absorber_section.rich_solvent_pump.max_work_con
        ] = 1
        sep_priority[
            model.fs.absorber_section.lean_solvent_pump.max_work_con
        ] = None
        sep_priority[model.fs.CO2_lower_bound] = 1

        sep_priority[
            model.fs.condenser_cw_temperature_penalty_constraint
        ] = None

        cond = model.fs.stripper_section.condenser
        reb = model.fs.stripper_section.reboiler
        state_vars_to_prioritize = ComponentSet((
            reb.condensate_temperature_outlet[0],
            cond.DeltaT_cold_end[0],
            cond.DeltaT_hot_end[0],
            reb.steam_flow_mol[0],
            cond.cooling_flow_mol[0],
        ))
        for var in get_state_vars_in_active_cons(model, get_dof_vars(model)):
            if var not in state_vars_to_prioritize:
                sep_priority[var] = None

        # prioritize the utility stream flow rates
        sep_priority[reb.condensate_temperature_outlet[0]] = 1
        sep_priority[cond.DeltaT_cold_end[0]] = 1
        sep_priority[cond.DeltaT_hot_end[0]] = 1
        sep_priority[reb.steam_flow_mol[0]] = 1
        sep_priority[cond.cooling_flow_mol[0]] = 1

    def solve_pyros(
            self,
            uncertainty_set,
            progress_logger,
            keep_model_in_results=True,
            **kwargs,
            ):
        """
        Solve RO model with PyROS and load solution.
        """
        first_stage_vars = self.get_scaled_first_stage_variables()
        second_stage_vars = self.get_scaled_second_stage_variables()
        uncertain_params = self.get_scaled_uncertain_params()
        final_uncertainty_set = self.get_scaled_ellipsoidal_set(
            uncertainty_set
        )

        pyros_solver = pyros.PyROS()
        pyros_res = pyros_solver.solve(
            model=self.scaled_model,
            first_stage_variables=first_stage_vars,
            second_stage_variables=second_stage_vars,
            uncertain_params=uncertain_params,
            uncertainty_set=final_uncertainty_set,
            progress_logger=progress_logger,
            load_solution=True,
            **kwargs,
        )
        pyo.TransformationFactory("core.scale_model").propagate_solution(
            scaled_model=self.scaled_model,
            original_model=self.original_model,
        )

        return self.create_flowsheet_results_object(
            solver_results=None,
            presolver_results=None,
            pyros_solver_results=pyros_res,
            keep_model=keep_model_in_results,
        )

    def _load_solution(self, results):
        """
        Load solution from solver results to `self.scaled_model`,
        then propagate (inverse scale) the solution and load to
        the original model.

        Parameters
        ----------
        results : pyomo.opt.SolverResults
            Results object returned by solver invoked on
            `self.scaled_model`.
        """
        self.scaled_model.solutions.load_from(results)
        pyo.TransformationFactory("core.scale_model").propagate_solution(
            scaled_model=self.scaled_model,
            original_model=self.original_model,
        )

    def adjust_flooding_requirements(self):
        """
        Adjust flooding fraction constraints of original
        and scaled models.
        """
        ...

    def adjust_co2_capture_target(self, co2_capture_target):
        """
        Adjust co2 capture target of original and scaled models.
        """
        self.original_model.fs.co2_capture_target.set_value(co2_capture_target)

        # should be same param object since params not scaled
        self.scaled_model.fs.co2_capture_target.set_value(co2_capture_target)

    def create_flowsheet_results_object(self, *args, **kwargs):
        """
        Create flowsheet results object from the original model
        contained in self.
        """
        results_type = self.get_flowsheet_results_type()
        return results_type(model_data=self, *args, **kwargs)

    @staticmethod
    def tune_flowsheet_model(fs_model):
        """
        Make type-specific modifications to flowsheet model.

        This should involve the addition and/or modification of
        components, such as an Objective.
        """
        pass

    @staticmethod
    def get_flowsheet_results_type():
        """
        Return matching flowsheet solve results type.
        """
        return MEAFlowsheetResults

    def get_heatmap_expressions_to_evaluate(self):
        """
        Get expressions from `self.original_model` to include
        in evaluation of heatmap solutions.
        """
        return get_heatmap_expressions_to_evaluate(self)


def adjust_co2_capture_target(model, co2_capture_target):
    """
    Adjust model CO2 capture target.
    """
    model.fs.co2_capture_target.set_value(co2_capture_target)


class MEAFlowsheetResults:
    """
    Container for MEA flowsheet results.
    """

    def __init__(
            self,
            model_data,
            solver_results,
            presolver_results=None,
            pyros_solver_results=None,
            keep_model=False,
            ):
        """Initialize self (see class docstring).

        """
        model = model_data.original_model
        self._setup_absorber_column_results(model)
        self._setup_stripper_column_results(model)
        self._setup_lean_rich_hex_results(model)
        self._setup_rich_solvent_pump_results(model)
        self._setup_lean_solvent_pump_results(model)
        self._setup_stripper_reboiler_results(model)
        self._setup_stripper_condenser_results(model)
        self._setup_flowsheet_obj_results(model)
        self._setup_solver_termination_results(solver_results)
        self._setup_presolver_termination_results(presolver_results)
        self._setup_pyros_solver_termination_results(pyros_solver_results)
        self._setup_recirc_stream_results(model)
        self._setup_solvent_loading_results(model)
        self._setup_solvent_temperature_results(model)

        self._setup_detailed_stream_results(model)

        # track active constraints and variables near bounds
        # note: these are not serialized
        active_lb_cons, active_ub_cons = get_active_inequality_constraints(
            model, tol=1e-4
        )
        active_lb_vars, active_ub_vars = get_active_var_bounds(
            model, tol=1e-4
        )
        self.active_lb_con_names = tuple(con.name for con in active_lb_cons)
        self.active_ub_con_names = tuple(con.name for con in active_ub_cons)
        self.active_lb_var_names = tuple(var.name for var in active_lb_vars)
        self.active_ub_var_names = tuple(var.name for var in active_ub_vars)

        self.solver_results = solver_results
        self.presolver_results = presolver_results
        self.pyros_solver_results = pyros_solver_results
        self.model_data = model_data if keep_model else None

    @staticmethod
    def _get_solver_results_bunch(solver_results):
        """
        Get Bunch summarizing main attributes of solver results object.

        Returns
        -------
        Bunch
            With attributes:

            - total_solve_time : float
                Total solver wall time, in seconds.
                If `solver_results` is None, then this attribute
                is set to None.
            - termination_condition : str or TerminationCondition
                Solver termination condition.
                If `solver_results` is None, then this attribute
                is set to None.
            - message : str
                Solver message.
                If `solver_results` is None, then this attribute
                is set to 'Solver not used'.
        """
        if solver_results is None:
            return Bunch(
                total_solve_time=None,
                termination_condition=None,
                message="Solver not used.",
            )
        else:
            return Bunch(
                total_solve_time=getattr(
                    solver_results.solver,
                    SolverWithBackup.TOTAL_WALL_TIME_ATTR,
                ),
                termination_condition=(
                    solver_results.solver.termination_condition
                ),
                message=solver_results.solver.message,
            )

    def _setup_solver_termination_results(self, solver_results):
        """
        Set up solver termination results.
        """
        self.solver_termination = self._get_solver_results_bunch(
            solver_results
        )

    def _setup_pyros_solver_termination_results(self, solver_results):
        """
        Set up pyros solver termination results.

        This sets up a new Bunch-type attribute
        `pyros_solver_termination` with the following attributes:

        - total_solve_time : float
            Total solver wall time, in seconds.
            If `solver_results` is None, then this attribute
            is set to None.
        - iterations : int
            Number of iterations elapsed.
            If `solver_results` is None, then this attribute
            is set to None.
        - pyros_termination_condition : pyrosTerminationCondition
            PyROS solver termination condition.
            If `solver_results` is None, then this attribute
            is set to None.
        - final_objective_value : float
            Final objective value reported in the results object.
            If `solver_results` is None, then this attribute
            is set to None.
        """
        if solver_results is None:
            self.pyros_solver_termination = Bunch(
                total_solve_time=None,
                iterations=None,
                pyros_termination_condition=None,
                final_objective_value=None,
                ellipsoid_num_std=None,
                ellipsoid_gaussian_conf_lvl=None,
            )
        else:
            is_uncertainty_set_ellipsoid = isinstance(
                solver_results.config.uncertainty_set,
                pyros.EllipsoidalSet,
            )
            assert is_uncertainty_set_ellipsoid
            from confidence_ellipsoid import calc_conf_lvl

            self.pyros_solver_termination = Bunch(
                total_solve_time=solver_results.time,
                iterations=solver_results.iterations,
                pyros_termination_condition=(
                    solver_results.pyros_termination_condition
                ),
                final_objective_value=solver_results.final_objective_value,
                ellipsoid_num_std=(
                    solver_results.config.uncertainty_set.scale ** 0.5
                ),
                ellipsoid_gaussian_conf_lvl=100 * calc_conf_lvl(
                    r=solver_results.config.uncertainty_set.scale ** 0.5,
                    n=solver_results.config.uncertainty_set.dim,
                ),
            )

    def _setup_presolver_termination_results(self, solver_results):
        """
        Set up pre-solver termination results
        """
        self.presolver_termination = self._get_solver_results_bunch(
            solver_results
        )

    def _setup_flowsheet_obj_results(self, model):
        """
        Set up flowsheet proxy objective results.
        This includes:

        - total cost
        - total CAPEX
        - total OPEX
        - individual CAPEX terms
        - individual OPEX terms
        """
        self.flowsheet_obj = Bunch()
        col_attr_expr_map = dict(
            obj=model.fs.obj,
            capex_cost=model.fs.capex_cost,
            **dict(model.fs.capex_cost_terms.items()),
            opex_cost=model.fs.opex_cost,
            **dict(model.fs.opex_cost_terms.items()),
        )
        self.flowsheet_obj.update({
            name: pyo.value(expr)
            for name, expr in col_attr_expr_map.items()
        })

    def _setup_solvent_loading_results(self, model):
        """
        Set up solvent CO2 loading results.

        This sets up a Bunch attribute `solvent_loading`
        with entries:

        - rich_loading : float
            Rich solvent loading, in mol CO2/mol MEA.
        - lean_loading : float
            Lean solvent loading, in mol CO2/mol MEA.
        """
        self.solvent_loading = Bunch(
            rich_loading=pyo.value(model.fs.rich_loading[0]),
            lean_loading=pyo.value(model.fs.lean_loading[0]),
        )

    def _setup_solvent_temperature_results(self, model):
        """
        Setup solvent temperature results.
        """
        self.solvent_temperature = Bunch(
            rich_temperature=pyo.value(
                model.fs.stripper_section.reflux_mixer.rich_solvent
                .temperature[0]
            ),
            lean_temperature=pyo.value(
                model.fs.absorber_section.absorber.liquid_inlet.temperature[0]
            ),
        )

    def _prepare_column_blk_results(self, column_blk):
        """
        Set up column block results.
        """
        section_blk = column_blk.parent_block()

        port_names = [
            "liquid_inlet", "liquid_outlet", "vapor_inlet", "vapor_outlet"
        ]

        col_attr_expr_map = dict(
            length_column=column_blk.length_column,
            diameter_column=column_blk.diameter_column,
            ld_ratio=section_blk.HDratio,
            lg_ratio=(
                # absorber ratio is not indexed, but stripper ratio is
                section_blk.LGratio[section_blk.LGratio.index_set().first()]
            ),
            volume_column_withheads=section_blk.volume_column_withheads / 1e3,
            volume_column=section_blk.volume_column / 1e3,
            top_flood_fraction=column_blk.flood_fraction[
                0,
                column_blk.vapor_phase.length_domain.last(),
            ],
            bottom_flood_fraction=column_blk.flood_fraction[
                0,
                column_blk.vapor_phase.length_domain.next(
                    column_blk.vapor_phase.length_domain.first()
                ),
            ],
        )
        for port_name in port_names:
            port_attr = getattr(column_blk, port_name)
            col_attr_expr_map.update({
                f"{port_name}_flow_rate": port_attr.flow_mol[0] / 1e3,
                f"{port_name}_temperature": port_attr.temperature[0],
                f"{port_name}_pressure": port_attr.pressure[0],
            })
            col_attr_expr_map.update({
                f"{port_name}_{comp.lower()}_mole_frac": mole_frac
                for (_, comp), mole_frac in port_attr.mole_frac_comp.items()
            })

        return col_attr_expr_map

    def _setup_absorber_column_results(self, model):
        """
        Set up absorber column results.

        This sets up a Bunch attribute `absorber_column`
        with entries:

        - co2_capture_target : float
            CO2 capture target (%).
        - co2_capture_rate : float
            CO2 capture rate (%).
        - length_column : float
            Column length, m.
        - diameter_column : float
            Column diameter, m.
        - ld_ratio : float
            Length/diameter ratio.
        - lg_ratio : float
            Ratio of liquid inlet to gas inlet molar flows.
        - volume_column_withheads : float
            Total volume of the column (cylinder + hemispherical heads),
            1000 m^3.
        - volume_column_withheads : float
            Total volume of the column (just the cylinder),
            1000 m^3.
        - liquid_inlet_flow_rate : float
            Liquid inlet molar flow, kmol/s.
        - liquid_outlet_flow_rate : float
            Liquid outlet molar flow, kmol/s.
        - vapor_inlet_flow_rate : float
            Vapor inlet molar flow, kmol/s.
        - vapor_outlet_flow_rate : float
            Vapor outlet molar flow, kmol/s.
        - vapor_outlet_h2o_mole_frac : float
            Vapor outlet H2O mole fraction.
        - vapor_outlet_co2_mole_frac : float
            Vapor outlet co2 mole fraction.
        """
        column_blk = model.fs.absorber_section.absorber
        col_attr_expr_map = dict(
            co2_capture_target=model.fs.co2_capture_target,
            co2_capture_rate=column_blk.co2_capture[0],
            **self._prepare_column_blk_results(column_blk),
        )
        self.absorber_column = Bunch(**{
            name: pyo.value(expr)
            for name, expr in col_attr_expr_map.items()
        })

    def _setup_stripper_column_results(self, model):
        """
        Set up stripper column results.

        This sets up a Bunch attribute `stripper_column`
        with entries:

        - co2_capture_target : float
            CO2 capture target (%).
        - co2_capture_rate : float
            CO2 capture rate (%).
        - length_column : float
            Column length, m.
        - diameter_column : float
            Column diameter, m.
        - ld_ratio : float
            Length/diameter ratio.
        - lg_ratio : float
            Ratio of liquid inlet to gas inlet molar flows.
        - volume_column_withheads : float
            Total volume of the column (cylinder + hemispherical heads),
            1000 m^3.
        - volume_column_withheads : float
            Total volume of the column (just the cylinder),
            1000 m^3.
        - liquid_inlet_flow_rate : float
            Liquid inlet molar flow, kmol/s.
        - liquid_outlet_flow_rate : float
            Liquid outlet molar flow, kmol/s.
        - vapor_inlet_flow_rate : float
            Vapor inlet molar flow, kmol/s.
        - vapor_outlet_flow_rate : float
            Vapor outlet molar flow, kmol/s.
        - vapor_outlet_h2o_mole_frac : float
            Vapor outlet H2O mole fraction.
        - vapor_outlet_co2_mole_frac : float
            Vapor outlet co2 mole fraction.
        """
        column_blk = model.fs.stripper_section.stripper
        col_attr_expr_map = self._prepare_column_blk_results(column_blk)
        self.stripper_column = Bunch(**{
            name: pyo.value(expr)
            for name, expr in col_attr_expr_map.items()
        })

    def _setup_recirc_stream_results(self, model):
        """
        Set up results for some important streams

        This sets up a bunch attribute with entries:

        - mea_recirculation_rate : float
            MEA component flow rate in lean solvent stream,
            kmol/s.
        - h2o_makeup : float
            H2O makeup flow rate, kmol/s.
        """
        self.other_streams = Bunch(
            mea_recirculation_rate=pyo.value(
                model.fs.mea_recirculation_rate[0] / 1e3
            ),
            h2o_makeup=pyo.value(model.fs.makeup.flow_mol[0] / 1e3),
        )

    def _prepare_solvent_pump_results(self, pump_blk):
        """
        Prepare solvent pump results.
        """
        return dict(
            flow_mol=pump_blk.control_volume.properties_in[0].flow_mol,
            inlet_flow_vol=pump_blk.control_volume.properties_in[0].flow_vol,
            inlet_temp=pump_blk.control_volume.properties_in[0].temperature,
            inlet_pressure=pump_blk.control_volume.properties_in[0].pressure,
            outlet_flow_vol=pump_blk.control_volume.properties_out[0].flow_vol,
            outlet_temp=pump_blk.control_volume.properties_out[0].temperature,
            outlet_pressure=pump_blk.control_volume.properties_out[0].pressure,
            deltaP=pump_blk.deltaP[0],
            power=pump_blk.control_volume.work[0],
            max_power=pump_blk.max_work,
        )

    def _setup_rich_solvent_pump_results(self, model):
        """
        Set up rich solvent pump results.

        This sets up a Bunch attribute with entries:

        - deltaP : float
            Pressure differential, Pa.
        - power : float
            Power requirement, W.
        - max_power : float
            Max power output specified by the design, W.
        """
        pump_blk = model.fs.absorber_section.rich_solvent_pump
        pump_attr_expr_map = self._prepare_solvent_pump_results(pump_blk)
        self.rich_solvent_pump = Bunch(**{
            name: pyo.value(expr)
            for name, expr in pump_attr_expr_map.items()
        })

    def _setup_lean_solvent_pump_results(self, model):
        """
        Set up lean solvent pump results.

        This sets up a Bunch attribute with entries:

        - deltaP : float
            Pressure differential, Pa.
        - power : float
            Power requirement, W.
        - max_power : float
            Max power output specified by the design, W.
        """
        pump_blk = model.fs.absorber_section.lean_solvent_pump
        pump_attr_expr_map = self._prepare_solvent_pump_results(pump_blk)
        self.lean_solvent_pump = Bunch(**{
            name: pyo.value(expr)
            for name, expr in pump_attr_expr_map.items()
        })

    def _prepare_hex_process_stream_data(self, stream_dict):
        """
        Prepare heat exchanger process stream data.
        """
        stream_data = dict()
        for stream_name, stream_blk in stream_dict.items():
            stream_data.update({
                f"{stream_name}_flow": (
                    stream_blk.flow_mol / 1e3
                ),
                f"{stream_name}_temp": (
                    stream_blk.temperature
                ),
                f"{stream_name}_pressure": (
                    stream_blk.pressure
                ),
            })
            stream_data.update({
                f"{stream_name}_{comp.lower()}_mole_frac": (
                    comp_mole_frac
                )
                for comp, comp_mole_frac
                in stream_blk.mole_frac_comp.items()
            })

        return stream_data

    def _setup_lean_rich_hex_results(self, model):
        """
        Set up results for the lean/rich HEX.

        This sets up a Bunch attribute with entries:

        - heat_duty : float
            Heat duty, MW.
        - area : float
            Heat exchanger area, m^2.
        - cold_side_inlet_temp : float
            Cold side stream inlet temperature, K.
        - cold_side_outlet_temp : float
            Cold side stream outlet temperature, K.
        - hot_side_inlet_temp : float
            Hot side stream inlet temperature, K.
        - hot_side_outlet_temp : float
            Hot side stream outlet temperature, K.
        - cold_side_flow_rate : float
            Cold side flow rate, kmol/s.
        - hot_side_flow_rate : float
            Hot side flow rate, kmol/s.
        """
        hex_block = model.fs.absorber_section.lean_rich_heat_exchanger
        process_streams = {
            "cold_side_inlet": hex_block.cold_side.properties_in[0],
            "cold_side_outlet": hex_block.cold_side.properties_out[0],
            "hot_side_inlet": hex_block.hot_side.properties_in[0],
            "hot_side_outlet": hex_block.hot_side.properties_out[0],
        }
        attr_expr_map = dict(
            heat_duty=hex_block.heat_duty[0] / 1e6,
            area=hex_block.area,
            **self._prepare_hex_process_stream_data(process_streams),
        )
        self.lean_rich_heat_exchanger = Bunch(**{
            name: pyo.value(expr)
            for name, expr in attr_expr_map.items()
        })

    def _setup_stripper_reboiler_results(self, model):
        """
        Set up results for the stripper reboiler.

        This sets up a Bunch attribute with entries:

        - heat_duty : float
            Heat duty, MW.
        - area : float
            Heat exchanger area, m^2.
        - steam_flow_mol : float
            Steam (utility) flow rate, kmol/s.
        - utility_inlet_temp : float
            Utility (steam) stream inlet temperature, K.
        - utility_outlet_temp : float
            Utility (steam) stream outlet temperature, K.
        - process_side_inlet_temp : float
            Process side inlet temperature, K.
        - process_side_outlet_temp : float
            Process side vapor outlet temperature, K.
        - process_side_liquid_outlet_temp : float
            Process side liquid outlet temperature, K.
        - process_side_liquid_inlet_flow : float
            Process side inlet flow, kmol/s.
            Process stream is 100% liquid, so this is the
            total process side inlet flow.
        - process_side_liquid_outlet_flow : float
            Process side liquid outlet flow, kmol/s.
        - process_side_vapor_outlet_flow : float
            Process side vapor outlet flow, kmol/s.
        """
        hex_block = model.fs.stripper_section.reboiler
        process_streams = {
            "process_side_liquid_inlet": (
                hex_block.liquid_phase.properties_in[0]
            ),
            "process_side_liquid_outlet": (
                hex_block.liquid_phase.properties_out[0]
            ),
            "process_side_vapor_outlet": hex_block.vapor_phase[0],
        }
        attr_expr_map = dict(
            heat_duty=hex_block.heat_duty[0] / 1e6,
            area=hex_block.area,
            steam_flow_mol=hex_block.steam_flow_mol[0] / 1e3,
            utility_inlet_temp=hex_block.steam_temperature_inlet,
            utility_outlet_temp=hex_block.condensate_temperature_outlet[0],
            **self._prepare_hex_process_stream_data(process_streams)
        )
        self.stripper_reboiler = Bunch(**{
            name: pyo.value(expr)
            for name, expr in attr_expr_map.items()
        })

    def _setup_stripper_condenser_results(self, model):
        """
        Set up results for the stripper reboiler.

        This sets up a Bunch attribute with entries:

        - heat_duty : float
            Heat duty, MW.
        - area : float
            Heat exchanger area, m^2.
        - cw_flow_mol : float
            Cooling water (utility) flow rate, kmol/s.
        - utility_inlet_temp : float
            Utility (CW) stream inlet temperature, K.
        - utility_outlet_temp : float
            Utility (CW) stream outlet temperature, K.
        - process_side_inlet_temp : float
            Process side inlet temperature, K.
        - process_side_outlet_temp : float
            Process side vapor outlet temperature, K.
        - process_side_liquid_outlet_temp : float
            Process side liquid outlet temperature, K.
        - process_side_vapor_inlet_flow : float
            Process side inlet flow, kmol/s.
            Process stream is 100% vapor, so this is the
            total process side inlet flow.
        - process_side_vapor_outlet_flow : float
            Process side vapor outlet flow, kmol/s.
        - process_side_liquid_outlet_flow : float
            Process side liquid outlet flow, kmol/s.
        - process_side_vapor_outlet_h2o_mole_frac : kmol/s
            Process side vapor outlet H2O mole fraction.
        """
        hex_block = model.fs.stripper_section.condenser
        process_streams = {
            "process_side_vapor_inlet": hex_block.vapor_phase.properties_in[0],
            "process_side_vapor_outlet": (
                hex_block.vapor_phase.properties_out[0]
            ),
            "process_side_liquid_outlet": hex_block.liquid_phase[0],
        }
        attr_expr_map = dict(
            heat_duty=hex_block.heat_duty[0] / 1e6,
            area=hex_block.area,
            cw_flow_mol=hex_block.cooling_flow_mol[0] / 1e3,
            utility_inlet_temp=hex_block.cooling_water_temperature_inlet,
            utility_outlet_temp=hex_block.cooling_water_temperature_outlet[0],
            **self._prepare_hex_process_stream_data(process_streams),
        )
        self.stripper_condenser = Bunch(**{
            name: pyo.value(expr)
            for name, expr in attr_expr_map.items()
        })

    def _setup_detailed_stream_results(self, model):
        """
        Setup detailed results for all process streams by extracting
        data from active Ports.

        The condenser cooling water and reboiler steam streams
        are not included.
        """
        from pyomo.network import Port
        self.detailed_stream_results_attr_names = []

        all_stream_ports = [
            port for port in model.component_data_objects(Port, active=True)
        ]
        for port in all_stream_ports:
            port_results = Bunch()

            for var in port.vars.values():
                vardata_objs = var.values() if var.is_indexed else [var[None]]
                for vardata in vardata_objs:
                    port_results[vardata.name] = pyo.value(vardata)

            attr_name = f"detailed_stream_{port.name}"
            setattr(self, attr_name, port_results)
            self.detailed_stream_results_attr_names.append(attr_name)

    def to_dict(self, include_detailed_stream_results=False):
        """
        Compile all the results objects stored in `self` to a dict.

        Parameters
        ----------
        include_detailed_stream_results : bool, optional
            True to include detailed stream results in the dict,
            False otherwise.

        Returns
        -------
        dict
            Compiled results. Each entry maps the name of
            one of the Bunch attributes of `self` to the
            value of the corresponding Bunch attribute.
        """
        res_dict = dict()

        if include_detailed_stream_results:
            detailed_stream_results_dict = {
                attr_name: getattr(self, attr_name)
                for attr_name in self.detailed_stream_results_attr_names
            }
        else:
            detailed_stream_results_dict = dict()

        attr_name_bunch_map = dict(
            absorber_column=self.absorber_column,
            stripper_column=self.stripper_column,
            solvent_loading=self.solvent_loading,
            solvent_temperature=self.solvent_temperature,
            stripper_condenser=self.stripper_condenser,
            stripper_reboiler=self.stripper_reboiler,
            lean_rich_heat_exchanger=self.lean_rich_heat_exchanger,
            rich_solvent_pump=self.rich_solvent_pump,
            lean_solvent_pump=self.lean_solvent_pump,
            other_streams=self.other_streams,
            flowsheet_obj=self.flowsheet_obj,
            solver_termination=self.solver_termination,
            presolver_termination=self.presolver_termination,
            pyros_solver_termination=self.pyros_solver_termination,
            **detailed_stream_results_dict,
        )
        for name, bunch in attr_name_bunch_map.items():
            res_dict[name] = dict(bunch)

        return res_dict

    def to_series(self, include_detailed_stream_results=False):
        """
        Compile all the results stored in `self` into a pandas
        Series object.

        Parameters
        ----------
        include_detailed_stream_results : bool, optional
            True to include detailed stream results in the series,
            False otherwise.

        Returns
        -------
        pandas.Series
            Compiled results. The index is a pandas MultiIndex
            object with two levels. Each entry of the index
            is a 2-tuple of which the first entry is the
            name of a Bunch attribute and the second entry
            is the name of an entry (key) of the Bunch attribute.
            This 2-tuple is mapped to the corresponding value
            stored in the entry of the Bunch attribute.
        """
        self_as_dict = self.to_dict(
            include_detailed_stream_results=include_detailed_stream_results,
        )
        series = pd.Series(
            index=pd.MultiIndex.from_product([[], []]),
            dtype=float,
        )
        for attr_name, sub_dict in self_as_dict.items():
            for key, val in sub_dict.items():
                series[(attr_name, key)] = val

        return series

    @staticmethod
    def get_dof_var_column_tuples():
        """
        Return list mapping entries of `self.to_series.index`
        to tuples for labeling first-stage variable and second-stage
        variable values.
        """
        first_stage_var_tuples = [
            (
                ("absorber_column", "length_column"),
                ("first_stage_vars", "absorber_length"),
            ),
            (
                ("absorber_column", "diameter_column"),
                ("first_stage_vars", "absorber_diameter"),
            ),
            (
                ("stripper_column", "length_column"),
                ("first_stage_vars", "stripper_length"),
            ),
            (
                ("stripper_column", "diameter_column"),
                ("first_stage_vars", "stripper_diameter"),
            ),
            (
                ("lean_rich_heat_exchanger", "area"),
                ("first_stage_vars", "lean_rich_hex_area"),
            ),
            (
                ("stripper_condenser", "area"),
                ("first_stage_vars", "stripper_condenser_area"),
            ),
            (
                ("stripper_reboiler", "area"),
                ("first_stage_vars", "stripper_reboiler_area"),
            ),
            (
                ("rich_solvent_pump", "max_power"),
                ("first_stage_vars", "rich_solvent_pump_max_power"),
            ),
        ]

        second_stage_var_tuples = [
            (
                ("other_streams", "mea_recirculation_rate"),
                ("second_stage_vars", "mea_recirculation_rate"),
            ),
            (
                ("stripper_reboiler", "steam_flow_mol"),
                ("second_stage_vars", "stripper_reboiler_steam_flow_mol"),
            ),
            (
                ("stripper_condenser", "cw_flow_mol"),
                ("second_stage_vars", "stripper_condenser_cw_flow_mol"),
            ),
        ]

        return first_stage_var_tuples + second_stage_var_tuples

    @staticmethod
    def produce_active_component_dataframes(target_to_fs_res_dict):
        """
        Create dataframe summarizing active constraints.
        """
        # set up dataframes
        active_cons_df = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                iterables=[
                    [target for target in target_to_fs_res_dict],
                    ("lb", "ub"),
                ],
            ),
        )
        active_vars_df = pd.DataFrame(
            columns=active_cons_df.columns.copy()
        )
        for target, fs_res in target_to_fs_res_dict.items():
            for btype in ["lb", "ub"]:
                con_names = getattr(fs_res, f"active_{btype}_con_names")
                for con_name in con_names:
                    active_cons_df.loc[con_name, (target, btype)] = True

                var_names = getattr(fs_res, f"active_{btype}_var_names")
                for var_name in var_names:
                    active_vars_df.loc[var_name, (target, btype)] = True

        return active_cons_df, active_vars_df

    @staticmethod
    def create_fs_results_dataframe(
            target_to_fs_res_dict,
            include_detailed_stream_results=False,
            ):
        """
        Create dataframe of results.
        """
        # cast all results to Series.
        # makes this method flexible: value entries need not be
        # results type; can be dict or even Series
        target_to_series_dict = _generate_target_to_series_map(
            target_to_fs_res_dict,
            include_detailed_stream_results=include_detailed_stream_results,
        )

        # now produce the dataframe
        first_series = next(iter(target_to_series_dict.values()))
        fs_res_df = pd.DataFrame(
            index=target_to_fs_res_dict.keys(),
            columns=first_series.index,
        )
        for target, series in target_to_series_dict.items():
            fs_res_df.loc[target] = series

        return fs_res_df

    @staticmethod
    def plot_fs_results(fs_results_dict, outdir):
        """
        Plot capture target response of flowsheet results.
        """
        fs_res_df = MEAFlowsheetResults.create_fs_results_dataframe(
            fs_results_dict
        )
        plot_proxy_costs(fs_res_df, outdir)
        plot_column_dimensions(fs_res_df, outdir)
        plot_hex_areas_and_duties(fs_res_df, outdir)
        plot_solvent_loading(fs_res_df, outdir)
        plot_solvent_temperatures(fs_res_df, outdir)
        plot_column_lg_ratios(fs_res_df, outdir)
        plot_mea_recirculation_rate(fs_res_df, outdir)
        plot_lean_solvent_flow_rate(fs_res_df, outdir)
        plot_column_port_temperatures(fs_res_df, outdir)
        plot_flood_fraction_profiles(
            fs_results_dict,
            outfile=os.path.join(outdir, "flood_fraction_profiles.png"),
            check_solver_termination=True,
            transpose=False,
        )
        plot_temperature_profiles(
            fs_results_dict,
            outfile=os.path.join(outdir, "temperature_profiles.png"),
            check_solver_termination=True,
        )


def get_active_flooding_constraints(column_blk, active_only=False, tol=1e-4):
    """
    Check column block for flooding constraints satisfied
    tightly.
    """
    active_lb_cons = [
        con for con in column_blk.LB_flood_ratio.values()
        if con.lslack() < tol
    ]
    active_ub_cons = [
        con for con in column_blk.UB_flood_ratio.values()
        if con.uslack() < tol
    ]

    return active_lb_cons, active_ub_cons


def build_and_solve_flowsheet_model(
        co2_capture_targets,
        solver=None,
        solver_options=None,
        progress_logger=None,
        keep_model_in_results=True,
        include_presolve=False,
        pre_solver=None,
        pre_solver_options=None,
        tee=False,
        flowsheet_model_data_type=None,
        flowsheet_model_data_kwargs=None,
        **deterministic_build_kwargs,
        ):
    """
    Build and solve flowsheet model for specified capture
    target, or for each target in a sequence of such.
    """
    capture_targets_iterable = (
        isinstance(co2_capture_targets, Iterable)
        and not isinstance(co2_capture_targets, str)
    )
    if capture_targets_iterable:
        co2_capture_targets = list(co2_capture_targets)
    else:
        co2_capture_targets = [co2_capture_targets]

    assert co2_capture_targets
    assert len(co2_capture_targets) == len(set(co2_capture_targets))

    if flowsheet_model_data_type is None:
        flowsheet_model_data_type = FlowsheetModelData
    if flowsheet_model_data_kwargs is None:
        flowsheet_model_data_kwargs = dict()

    if progress_logger is None:
        progress_logger = default_logger

    # we want to keep track of the simulation capture target as needed
    build_target = (
        "simulation"
        if "simulation" in co2_capture_targets else co2_capture_targets[0]
    )

    # state of model here is independent of CO2 capture
    # (apart from capture constraints), so we
    # need only build model once to speed things up,
    # then clone later as necessary
    model = build_deterministic_flowsheet_model(
        co2_capture_target=build_target,
        progress_logger=progress_logger,
        flowsheet_model_data_type=flowsheet_model_data_type,
        flowsheet_model_data_kwargs=flowsheet_model_data_kwargs,
        **deterministic_build_kwargs,
    )

    # replace 'simulation' with the actual initial square model
    # solution target
    if build_target == "simulation":
        co2_capture_targets = [
            target if target != "simulation"
            else pyo.value(model.fs.co2_capture_target)
            for target in co2_capture_targets
        ]

    # set up solver. default is customized IPOPT
    if solver is None:
        solver = default_solver
    if solver_options is None:
        solver_options = default_solver_options

    opt = get_solver(solver=solver, solver_options=solver_options)
    workflow_opt = SolverWithBackup(opt, logger=progress_logger)

    # now solve model for each of specified targets
    all_fs_results = dict()
    for target in co2_capture_targets:
        progress_logger.info(
            f"Solving deterministic optimization model for capture target "
            f"{target}..."
        )

        mdl = model.clone()
        adjust_co2_capture_target(mdl, target)
        model_data = flowsheet_model_data_type(
            original_model=mdl,
            scaled_model=None,
            **flowsheet_model_data_kwargs,
        )

        all_fs_results[target] = fs_res = model_data.solve_deterministic(
            solver=workflow_opt,
            tee=tee,
            symbolic_solver_labels=True,
            logger=progress_logger,
        )
        if pyo.check_optimal_termination(fs_res.solver_results):
            model_data.log_objective_breakdown(
                progress_logger,
                level=logging.DEBUG,
            )
            for btype in ["lb", "ub"]:
                con_names = getattr(fs_res, f"active_{btype}_con_names")
                progress_logger.debug(f"Active {btype.upper()} Constraints:")
                progress_logger.debug(
                    " " + "\n ".join(f"{con_name!r}" for con_name in con_names)
                )
            for btype in ["lb", "ub"]:
                var_names = getattr(fs_res, f"active_{btype}_var_names")
                progress_logger.debug(f"Vars Near {btype.upper()}:")
                progress_logger.debug(
                    " " + "\n ".join(f"{var_name!r}" for var_name in var_names)
                )

        progress_logger.debug(
            fs_res
            .to_series(include_detailed_stream_results=False)
            .to_string()
        )

    progress_logger.info("All done solving.")

    if capture_targets_iterable:
        return all_fs_results
    else:
        return fs_res


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_temperature_profiles(
        results_dict,
        outfile=None,
        logger=default_logger,
        check_solver_termination=True,
        ):
    """
    Plot temperature profiles.
    """
    assert results_dict

    fig, (absorber_axs, stripper_axs) = plt.subplots(
        figsize=(12, 6),
        nrows=2,
        ncols=2,
    )

    for idx, (target, res) in enumerate(results_dict.items()):
        # validate model
        model = res.model_data.original_model
        if model is None:
            raise ValueError(
                f"{MEAFlowsheetResults.__name__} object for target "
                f"{target} not present; required for flood fraction "
                "profile plots."
            )

        is_termation_acceptable = (
            not check_solver_termination
            or pyo.check_optimal_termination(res.solver_results)
        )

        col_zip = zip(
            (absorber_axs, stripper_axs),
            ("absorber", "stripper"),
            (
                model.fs.absorber_section.absorber,
                model.fs.stripper_section.stripper,
            ),
        )
        for (liq_ax, vap_ax), colname, colblk in col_zip:
            for ax, phase in zip([liq_ax, vap_ax], ["liquid", "vapor"]):
                phase_blk = getattr(colblk, f"{phase}_phase")
                height_to_temp_var_dict = {
                    height: prop_blk.temperature
                    for (_, height), prop_blk
                    in phase_blk.properties.items()
                }
                heights = list(height_to_temp_var_dict.keys())

                if is_termation_acceptable:
                    temps = [
                        var.value - 273.15
                        for var in height_to_temp_var_dict.values()
                    ]
                    ax.plot(
                        heights,
                        temps,
                        label=f"{target:.2f}",
                        marker="none",
                    )

                if idx == len(results_dict) - 1:
                    heightlabel = "Normalized Height"
                    templabel = wrap_quantity_str(
                        namestr=(
                            f"{colname.capitalize()} {phase.capitalize()} "
                            "Phase Temperature"
                        ),
                        unitstr="°C",
                    )
                    ax.set_xlabel(heightlabel)
                    ax.set_ylabel(templabel)
                    ax.legend(ncols=2)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)
        logger.info(
            f"Successfully serialized temperature "
            f"profiles to path {outfile!r}."
        )
    plt.close()


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_flood_fraction_profiles(
        results_dict,
        flood_fraction_lb=0.5,
        flood_fraction_ub=0.8,
        outfile=None,
        logger=default_logger,
        check_solver_termination=True,
        transpose=False,
        ):
    """
    Plot absorber and stripper column flood fraction profiles.

    Parameters
    ----------
    results_dict : dict
        Mapping from minimum capture rates to corresponding
        MEAColumnSolveResults.
    flood_fraction_lb : float
        Flood fraction lower bound.
        Plotted as a dshed line.
    flood_fraction_ub : float
        Flood fraction upper bound.
        Plotted as a dashed line.
    outfile : None or path-like, optional
        File to which to export plot. If `None` is provided,
        then plot is not exported.
    logger : logging.Logger, optional
        Progress logger.
    transpose : bool, optional
        True to make the flood fraction, rather than the
        normalized height, the x-coordinate, False otherwise.
    """
    assert results_dict

    fig, (absorber_ax, stripper_ax) = plt.subplots(
        figsize=(6, 7.2) if transpose else (6, 5),
        nrows=1 if transpose else 2,
        ncols=2 if transpose else 1,
    )

    for idx, (target, res) in enumerate(results_dict.items()):
        # validate model
        model = res.model_data.original_model
        if model is None:
            raise ValueError(
                f"{MEAFlowsheetResults.__name__} object for target "
                f"{target} not present; required for flood fraction "
                "profile plots."
            )

        is_termation_acceptable = (
            not check_solver_termination
            or pyo.check_optimal_termination(res.solver_results)
        )

        col_zip = zip(
            (absorber_ax, stripper_ax),
            ("absorber", "stripper"),
            (
                model.fs.absorber_section.absorber,
                model.fs.stripper_section.stripper,
            ),
        )
        for ax, colname, colblk in col_zip:
            if is_termation_acceptable:
                bed_heights, flood_fracs = np.array([
                    [height, pyo.value(frac_var)]
                    for (_, height), frac_var
                    in colblk.flood_fraction.items()
                    # exclude bed height 0: out of scope of the model
                    if height > 0
                ]).T
                if transpose:
                    ax.plot(
                        flood_fracs,
                        bed_heights,
                        label=f"{target:.2f}",
                        marker="none",
                    )
                else:
                    ax.plot(
                        bed_heights,
                        flood_fracs,
                        label=f"{target:.2f}",
                        marker="none",
                    )

            if idx == len(results_dict) - 1:
                frac_line_func = ax.axvline if transpose else ax.axhline
                frac_line_func(
                    flood_fraction_lb,
                    color="k",
                    linestyle="--",
                    label="bounds",
                )
                frac_line_func(
                    flood_fraction_ub,
                    color="k",
                    linestyle="--",
                )

                set_bed_ax_lim = ax.set_ylim if transpose else ax.set_xlim
                set_bed_ax_lim(0, 1)

                heightlabel = "Normalized Height"
                fraclabel = f"{colname.capitalize()} Flood Fraction"
                ax.set_xlabel(fraclabel if transpose else heightlabel)
                ax.set_ylabel(heightlabel if transpose else fraclabel)
                ax.legend(ncols=1 if transpose else 2)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)
        logger.info(
            f"Successfully serialized flood fraction "
            f"profiles to path {outfile!r}."
        )
    plt.close()


def _generate_target_to_series_map(
        target_to_results_dict,
        include_detailed_stream_results=False,
        ):
    """
    Standardize capture target to results dict
    so that each entry maps target to a dict.
    """
    # set up dataframe
    target_to_results_dict = target_to_results_dict.copy()
    for target, res in target_to_results_dict.items():
        if isinstance(res, MEAFlowsheetResults):
            target_to_results_dict[target] = res.to_series(
                include_detailed_stream_results=(
                    include_detailed_stream_results
                ),
            )
        elif isinstance(res, dict):
            series = pd.Series(
                index=pd.MultiIndex.from_product([[], []]),
                dtype=float,
            )
            for attr_name, sub_dict in res.items():
                for key, val in sub_dict.items():
                    series[(attr_name, key)] = val
            target_to_results_dict[target] = series

    return target_to_results_dict


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_lean_solvent_flow_rate(deterministic_results_df, outdir):
    """
    Plot MEA solvent flow rate.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    ax.set_ylabel(wrap_quantity_str("Lean Solvent Flow Rate", "kmol/s"))
    ax.plot(
        acceptable_results_df.index,
        acceptable_results_df[("absorber_column", "liquid_inlet_flow_rate")],
    )

    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )

    # export to outfile
    fig.savefig(
        os.path.join(outdir, "lean_solvent_flow_rate.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_mea_recirculation_rate(deterministic_results_df, outdir):
    """
    Plot deterministic model MEA recirculation rate results.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    ax.set_ylabel("MEA Recirculation Rate (kmol/s)")
    ax.plot(
        acceptable_results_df.index,
        acceptable_results_df[("other_streams", "mea_recirculation_rate")],
    )

    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )

    # export to outfile
    fig.savefig(
        os.path.join(outdir, "mea_recirculation_rate.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_solvent_temperatures(deterministic_results_df, outdir):
    """
    Plot deterministic model solvent temperature results.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\text{CO}_2$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr="Solvent Temperature",
        unitstr="°C",
    ))
    for solvent_type in ["rich", "lean"]:
        ax.plot(
            acceptable_results_df.index,
            acceptable_results_df[
                ("solvent_temperature", f"{solvent_type}_temperature")
            ] - 273.15,
            label=solvent_type,
        )

    ax.legend()
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )

    # export to outfile
    fig.savefig(
        os.path.join(outdir, "solvent_temperatures.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_solvent_loading(deterministic_results_df, outdir):
    """
    Plot deterministic model solvent loading results.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    ax.set_ylabel("Solvent Loading (mol $\\text{CO}_2$/mol MEA)")

    for loading_type in ["rich", "lean"]:
        ax.plot(
            acceptable_results_df.index,
            acceptable_results_df[
                ("solvent_loading", f"{loading_type}_loading")
            ],
            label=loading_type,
        )

    ax.legend()
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )

    # export to outfile
    fig.savefig(
        os.path.join(outdir, "solvent_loading.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_column_lg_ratios(deterministic_results_df, outdir):
    """
    Plot absorber and stripper column L/G
    (liquid inlet/vapor inlet molar flow) ratios.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    ax.set_ylabel("L/G ratio (mol liquid/mol vapor)")

    for column_type in ["absorber", "stripper"]:
        ax.plot(
            acceptable_results_df.index,
            acceptable_results_df[(f"{column_type}_column", "lg_ratio")],
            label=column_type,
        )

    ax.legend()
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )

    # export to outfile
    fig.savefig(
        os.path.join(outdir, "lg_ratios.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_column_port_temperatures(deterministic_results_df, outdir):
    """
    Plot absorber and stripper column port temperatures.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, [absorber_ax, stripper_ax] = plt.subplots(ncols=2, figsize=(6, 3.5))

    column_type_ax_dict = {"absorber": absorber_ax, "stripper": stripper_ax}
    port_names = [
        "liquid_inlet",
        "liquid_outlet",
        "vapor_inlet",
        "vapor_outlet",
    ]

    for column_type, ax in column_type_ax_dict.items():
        ax.set_xlabel(wrap_quantity_str(
            namestr="$\\text{CO}_2$ Capture Target",
            unitstr="\\%",
        ))
        ax.set_ylabel(wrap_quantity_str(
            namestr=f"{column_type.capitalize()} Port Temperature",
            unitstr="°C",
        ))
        for port_name in port_names:
            ax.plot(
                acceptable_results_df.index,
                acceptable_results_df[
                    (f"{column_type}_column", f"{port_name}_temperature")
                ] - 273.15,
                label=port_name.replace("_", " "),
            )

        ax.legend(
            bbox_to_anchor=(0, -0.2),
            loc="upper left",
            ncol=1,
        )
        set_nonoverlapping_fixed_xticks(
            fig=fig,
            ax=ax,
            locs=acceptable_results_df.index,
        )

    # export to outfile
    fig.savefig(
        os.path.join(outdir, "column_port_temperatures.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_proxy_cost_summary(results_df, fig, ax):
    """
    Plot summary of proxy costs on given Axes.
    """
    ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    ax.set_ylabel("Proxy Cost")

    ax.plot(
        results_df.index,
        results_df[("flowsheet_obj", "obj")],
        label="total",
    )
    ax.plot(
        results_df.index,
        results_df[("flowsheet_obj", "capex_cost")],
        label="CAPEX",
    )
    ax.plot(
        results_df.index,
        results_df[("flowsheet_obj", "opex_cost")],
        label="OPEX",
    )
    ax.legend(
        bbox_to_anchor=(0, -0.2),
        loc="upper left",
        ncol=1,
    )
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=results_df.index,
    )


def _plot_proxy_cost_terms(results_df, fig, capex_ax, opex_ax):
    """
    Plot deterministic model CAPEX cost terms.
    """
    for ax, cost_type in zip([capex_ax, opex_ax], ["capex", "opex"]):
        cost_cols = [
            col for col in results_df.columns
            if col[0] == "flowsheet_obj"
            and f"{cost_type}_cost" in col[1]
        ]

        plot_labels = []
        for _, term_name in cost_cols:
            if term_name == f"{cost_type}_cost":
                plot_labels.append("total")
            else:
                plot_labels.append(
                    term_name
                    .replace(f"{cost_type}_cost", "")
                    .replace("_", " ")
                    .replace("mea", "MEA")
                    .replace("h2o", "$\\text{H_2O}$")
                    .replace("hex", "HEX")
                    .replace("cw", "CW")
                )

        ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
        ax.set_ylabel(f"Proxy {cost_type.upper()} Cost")
        ax.plot(
            results_df.index,
            results_df[cost_cols].to_numpy(),
            label=plot_labels,
        )
        ax.legend(
            bbox_to_anchor=(0, -0.2),
            loc="upper left",
            ncol=1,
        )
        set_nonoverlapping_fixed_xticks(
            fig=fig,
            ax=ax,
            locs=results_df.index,
        )


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_proxy_costs(deterministic_results_df, outdir):
    """
    Plot response of deterministically optimized proxy cost
    to the CO2 capture target.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, (summary_ax, capex_ax, opex_ax) = plt.subplots(
        figsize=(12, 5),
        ncols=3,
    )

    _plot_proxy_cost_summary(acceptable_results_df, fig, summary_ax)
    _plot_proxy_cost_terms(acceptable_results_df, fig, capex_ax, opex_ax)

    # export to outfile
    plt.savefig(
        os.path.join(outdir, "proxy_costs.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_deterministic_results_spreadsheet(
        co2_capture_targets,
        progress_logger=None,
        include_plots=False,
        outdir=None,
        include_detailed_stream_results=False,
        flowsheet_model_data_type=None,
        flowsheet_model_data_kwargs=None,
        **deterministic_build_and_solve_kwargs,
        ):
    """
    Solve the nominal deterministic flowsheet optimization model
    for a given sequence of capture targets.
    """
    # cast sequence of targets to list; must be nonempty
    co2_capture_targets = list(co2_capture_targets)
    assert co2_capture_targets

    if progress_logger is None:
        progress_logger = default_logger
    if flowsheet_model_data_type is None:
        flowsheet_model_data_type = FlowsheetModelData
    if flowsheet_model_data_kwargs is None:
        flowsheet_model_data_kwargs = dict()

    progress_logger.info(
        f"Starting {generate_deterministic_results_spreadsheet.__name__!r}..."
    )
    progress_logger.info(
        f"Invoked at UTC: {datetime.datetime.utcnow().isoformat()}"
    )
    log_main_dependency_module_info(logger=progress_logger, level=logging.INFO)

    # solve model
    target_to_fs_res_dict = build_and_solve_flowsheet_model(
        co2_capture_targets=co2_capture_targets,
        progress_logger=progress_logger,
        flowsheet_model_data_type=flowsheet_model_data_type,
        flowsheet_model_data_kwargs=flowsheet_model_data_kwargs,
        **deterministic_build_and_solve_kwargs,
    )

    # serialize, if desired
    fs_res_type = flowsheet_model_data_type.get_flowsheet_results_type()
    if outdir is not None:
        fs_res_df = (
            flowsheet_model_data_type
            .get_flowsheet_results_type()
            .create_fs_results_dataframe(
                target_to_fs_res_dict,
                include_detailed_stream_results=(
                    include_detailed_stream_results
                ),
            )
        )
        fs_res_df.to_csv(os.path.join(outdir, "deterministic_results.csv"))

        active_cons_df, active_vars_df = (
            fs_res_type.produce_active_component_dataframes(
                target_to_fs_res_dict,
            )
        )
        active_cons_df[active_cons_df.any(axis=1)].to_csv(
            os.path.join(outdir, "active_cons.csv")
        )
        active_vars_df[active_vars_df.any(axis=1)].to_csv(
            os.path.join(outdir, "vars_near_bounds.csv")
        )

        if include_plots:
            # generate plots. see `plot_results` of original script
            # for plots to generate model objects needed to
            # successfully generate flood fraction plots
            fs_restype = flowsheet_model_data_type.get_flowsheet_results_type()
            fs_restype.plot_fs_results(
                fs_results_dict=target_to_fs_res_dict,
                outdir=outdir,
            )

        progress_logger.info(
            f"All results and plots written to directory {outdir!r}."
        )

    return target_to_fs_res_dict


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_column_dimensions(deterministic_results_df, outdir):
    """
    Plot response of optimized column dimensions to CO2 capture target.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_terminations = {
        "optimal",
        "globallyOptimal",
        "locallyOptimal",
    }
    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin(acceptable_terminations)
    ]

    # set up plots
    fig, [axdims, axvols] = plt.subplots(nrows=2, ncols=2, figsize=(6, 5))
    plotzip = zip(
        [axdims, axvols],
        [["length_column", "diameter_column"], ["volume_column_withheads"]],
    )
    for axs, dim_names in plotzip:
        for ax, col_name in zip(axs, ["absorber", "stripper"]):
            col_name_capitalized = col_name.capitalize()
            ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")

            vol_str = wrap_quantity_str(
                namestr=f"{col_name_capitalized} Volume (with heads)",
                unitstr="$10^3\\,\\text{m}^3$",
            )
            dim_str = wrap_quantity_str("Column Dimension", "m")
            ax.set_ylabel(dim_str if axs is axdims else vol_str)
            for dim_name in dim_names:
                col_dim_values = acceptable_results_df[
                    f"{col_name}_column", dim_name,
                ]
                dim_label = (
                    dim_name
                    .replace("_column", "")
                    .replace("_withheads", " (with heads)")
                )
                ax.plot(
                    acceptable_results_df.index,
                    col_dim_values,
                    label=dim_label,
                )
            if axs is axdims:
                ax.legend()
            set_nonoverlapping_fixed_xticks(
                fig=fig,
                ax=ax,
                locs=acceptable_results_df.index,
            )

    # export to outfile
    plt.savefig(
        os.path.join(outdir, "column_dimensions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_hex_areas_and_duties(deterministic_results_df, outdir):
    """
    Generate and serialize two plots:

    - column dimensions
    - HEX areas and duties

    Parameters
    ----------
    target_to_results_dict : dict
        Maps each design capture target to a corresponding
        results object of type MEAFlowsheetResults, dict,
        or pandas.Series.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_terminations = {
        "optimal",
        "globallyOptimal",
        "locallyOptimal",
    }
    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin(acceptable_terminations)
    ]

    # now we plot CAPEX and OPEX
    hex_fig, (area_ax, duty_ax) = plt.subplots(figsize=(6, 3.2), ncols=2)

    area_ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    area_ax.set_ylabel(r"Heat exchanger area ($\text{m}^2$)")
    duty_ax.set_xlabel("$\\text{CO}_2$ Capture Target (\\%)")
    duty_ax.set_ylabel(r"Heat exchanger duty ($\text{MW}$)")

    hex_area_plot_list = (
        ("lean_rich_heat_exchanger", "lean-rich heat exchanger"),
        ("stripper_reboiler", "stripper reboiler"),
        ("stripper_condenser", "stripper condenser"),
    )
    for hex_name, hex_label in hex_area_plot_list:
        area_ax.plot(
            acceptable_results_df.index,
            acceptable_results_df[(hex_name, "area")],
            label=hex_label,
        )
        duty_ax.plot(
            acceptable_results_df.index,
            (
                acceptable_results_df[(hex_name, "heat_duty")]
                * (-1 if hex_name == "stripper_condenser" else 1)
            ),
            label=hex_label,
        )
    for hex_ax in [area_ax, duty_ax]:
        hex_ax.legend(
            bbox_to_anchor=(0, -0.2),
            loc="upper left",
            ncol=1,
        )
        set_nonoverlapping_fixed_xticks(
            fig=hex_fig,
            ax=hex_ax,
            locs=acceptable_results_df.index,
        )

    hex_fig.savefig(
        os.path.join(outdir, "hex_areas_and_duties.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(hex_fig)


def declare_and_substitute_uncertain_params(model, logger=default_logger):
    """
    Declare new mutable Param components for the uncertain
    quantities, and substitute these params for the
    fixed Vars originally representing those quantities
    into the model objective/constraints/expressions.

    This method contains (potentially very expensive) checks
    to ensure the substitutions were carried out as expected.
    """
    model.fs.k_eq_b_bicarbonate = pyo.Param(mutable=True)
    model.fs.k_eq_b_carbamate = pyo.Param(mutable=True)
    model.fs.lwm_coeff_1 = pyo.Param(mutable=True)
    model.fs.lwm_coeff_2 = pyo.Param(mutable=True)
    model.fs.lwm_coeff_3 = pyo.Param(mutable=True)
    model.fs.lwm_coeff_4 = pyo.Param(mutable=True)

    absorber_blk = model.fs.absorber_section
    stripper_blk = model.fs.stripper_section

    param_to_fixed_var_list_map = ComponentMap()
    param_to_fixed_var_list_map[model.fs.k_eq_b_bicarbonate] = [
        absorber_blk.liquid_properties.reaction_bicarbonate.k_eq_coeff_2,
        stripper_blk.liquid_properties.reaction_bicarbonate.k_eq_coeff_2,
    ]
    param_to_fixed_var_list_map[model.fs.k_eq_b_carbamate] = [
        absorber_blk.liquid_properties.reaction_carbamate.k_eq_coeff_2,
        stripper_blk.liquid_properties.reaction_carbamate.k_eq_coeff_2,
    ]
    param_to_fixed_var_list_map[model.fs.lwm_coeff_1] = [
        absorber_blk.liquid_properties.CO2.lwm_coeff_1,
        stripper_blk.liquid_properties.CO2.lwm_coeff_1,
        absorber_blk.liquid_properties_no_ions.CO2.lwm_coeff_1,
        stripper_blk.liquid_properties_no_ions.CO2.lwm_coeff_1,
    ]
    param_to_fixed_var_list_map[model.fs.lwm_coeff_2] = [
        absorber_blk.liquid_properties.CO2.lwm_coeff_2,
        stripper_blk.liquid_properties.CO2.lwm_coeff_2,
        absorber_blk.liquid_properties_no_ions.CO2.lwm_coeff_2,
        stripper_blk.liquid_properties_no_ions.CO2.lwm_coeff_2,
    ]
    param_to_fixed_var_list_map[model.fs.lwm_coeff_3] = [
        absorber_blk.liquid_properties.CO2.lwm_coeff_3,
        stripper_blk.liquid_properties.CO2.lwm_coeff_3,
        absorber_blk.liquid_properties_no_ions.CO2.lwm_coeff_3,
        stripper_blk.liquid_properties_no_ions.CO2.lwm_coeff_3,
    ]
    param_to_fixed_var_list_map[model.fs.lwm_coeff_4] = [
        absorber_blk.liquid_properties.CO2.lwm_coeff_4,
        stripper_blk.liquid_properties.CO2.lwm_coeff_4,
        absorber_blk.liquid_properties_no_ions.CO2.lwm_coeff_4,
        stripper_blk.liquid_properties_no_ions.CO2.lwm_coeff_4,
    ]

    assert len(param_to_fixed_var_list_map) == 6

    fixed_var_to_con_map = ComponentMap()
    fixed_var_to_expr_map = ComponentMap()

    for param, varlist in param_to_fixed_var_list_map.items():
        assert all(varlist[0].value == fvar.value for fvar in varlist[1:])
        assert all(fvar.fixed for fvar in varlist)

        param.set_value(varlist[0].value)
        for fixed_var in varlist:
            fixed_var_to_con_map[fixed_var] = ComponentSet(
                get_cons_with_component(fixed_var, active=None)
            )
            fixed_var_to_expr_map[fixed_var] = ComponentSet(
                get_exprs_with_component(fixed_var)
            )

    substitute_uncertain_params(
        m=model,
        substitution_map=param_to_fixed_var_list_map,
        include_var_like_exprs=True,
        logger=logger,
    )

    # confirm the substitutions worked as expected
    for param, varlist in param_to_fixed_var_list_map.items():
        expected_cons_with_param = ComponentSet(itertools.chain(
            *tuple(fixed_var_to_con_map[fvar] for fvar in varlist)
        ))
        expected_exprs_with_param = ComponentSet(itertools.chain(
            *tuple(fixed_var_to_expr_map[fvar] for fvar in varlist)
        ))

        actual_cons_with_param = ComponentSet(get_cons_with_component(param))
        actual_exprs_with_param = ComponentSet(get_exprs_with_component(param))

        assert expected_cons_with_param == actual_cons_with_param
        assert expected_exprs_with_param == actual_exprs_with_param

        for fvar in varlist:
            assert not ComponentSet(get_cons_with_component(fvar))
            assert not ComponentSet(get_exprs_with_component(fvar))


def incorporate_model_uncertainty(
        model,
        cov_mat_infile,
        full_uncertainty=True,
        ):
    """
    Declare uncertain model parameters (mutable `Param` objects),
    and substitute for the corresponding fixed `Var` objects.

    Parameters
    ----------
    m : ConcreteModel
        Column model.
    input_file : path-like
        File from which to read covariance matrix.
    full_uncertainty : bool, optional
        True to use 6-dimensional uncertainty set
        (2 reaction equilibrium coefficients + 4 VLE parameters),
        False to use 2-dimensional uncertainty set
        (2 reaction equilibrium coefficients).
        In the event the 2D set is used, a hard-coded covariance
        matrix is returned in lieu of the matrix read from
        `input_file`.

    Returns
    -------
    uncertain_params : list of ParamData
        Uncertain model parameters.
    cov_mat : (N, N) numpy.ndarray
        Covariance matrix. If ``full_uncertainty=True``, then `N=6`,
        otherwise `N=2`.
    """
    if full_uncertainty:
        cov_mat_df = pd.read_csv(cov_mat_infile, index_col=0)
        assert cov_mat_df.index.tolist() == cov_mat_df.columns.tolist()

        # ensure order of uncertain params and cov matrix is consistent
        df_idx_to_param_map = {
            "bic_k_eq_coeff_2": model.fs.k_eq_b_bicarbonate,
            "car_k_eq_coeff_2": model.fs.k_eq_b_carbamate,
            "lwm_coeff_1": model.fs.lwm_coeff_1,
            "lwm_coeff_2": model.fs.lwm_coeff_2,
            "lwm_coeff_3": model.fs.lwm_coeff_3,
            "lwm_coeff_4": model.fs.lwm_coeff_4,
        }
        uncertain_params = [
            df_idx_to_param_map[idx]
            for idx in cov_mat_df.index
        ]
        cov_mat = cov_mat_df.to_numpy()
    else:
        uncertain_params = [
            model.fs.k_eq_b_bicarbonate,
            model.fs.k_eq_b_carbamate,
        ]
        cov_mat = np.array([
            [30.23431579, 27.26328881],
            [27.26328881, 435.8242653],
        ])

    # ensure params are all unique
    assert len(uncertain_params) == len(ComponentSet(uncertain_params))

    return uncertain_params, cov_mat


def pyros_workflow(
        co2_capture_target,
        confidence_level,
        cov_mat_infile,
        include_plots=False,
        progress_logger=None,
        pyros_options=None,
        outdir=None,
        prioritize_absorber_lb_flood_idxs=None,
        prioritize_absorber_ub_flood_idxs=None,
        prioritize_stripper_lb_flood_idxs=None,
        prioritize_stripper_ub_flood_idxs=None,
        capture_targets_for_heatmaps=None,
        heatmap_samples_infile=None,
        heatmap_confidence_level=99.0,
        heatmap_output_dir=None,
        heatmap_solvers=None,
        include_deterministic_solution_heatmap=False,
        heatmap_tee=False,
        export_nonoptimal_heatmap_models=True,
        acceptable_det_terminations=None,
        acceptable_pyros_terminations_for_heatmap=None,
        deterministic_build_and_solve_kwargs=None,
        flowsheet_model_data_type=None,
        flowsheet_model_data_kwargs=None,
        include_detailed_stream_results=False,
        ):
    """
    Solve flowsheet RO model with PyROS.
    """
    if progress_logger is None:
        progress_logger = default_logger
    if deterministic_build_and_solve_kwargs is None:
        deterministic_build_and_solve_kwargs = dict()
    if flowsheet_model_data_type is None:
        flowsheet_model_data_type = FlowsheetModelData
    if flowsheet_model_data_kwargs is None:
        flowsheet_model_data_kwargs = dict()

    progress_logger.info(
        f"Starting {pyros_workflow.__name__!r}..."
    )
    progress_logger.info(
        f"Workflow for {co2_capture_target=!r}, {confidence_level=!r}"
    )
    progress_logger.info(
        f"Invoked at UTC: {datetime.datetime.utcnow().isoformat()}"
    )
    log_main_dependency_module_info(logger=progress_logger, level=logging.INFO)

    deterministic_fs_res = build_and_solve_flowsheet_model(
        co2_capture_targets=co2_capture_target,
        progress_logger=progress_logger,
        keep_model_in_results=True,
        flowsheet_model_data_type=flowsheet_model_data_type,
        flowsheet_model_data_kwargs=flowsheet_model_data_kwargs,
        **deterministic_build_and_solve_kwargs,
    )
    model = deterministic_fs_res.model_data.original_model

    # resolve co2 capture target
    co2_capture_target = pyo.value(model.fs.co2_capture_target)

    if acceptable_det_terminations is None:
        acceptable_det_terminations = {
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.globallyOptimal,
        }
    else:
        acceptable_det_terminations = set(acceptable_det_terminations)

    termination_acceptable = (
        deterministic_fs_res.solver_results.solver.termination_condition
        in acceptable_det_terminations
    )
    if not termination_acceptable:
        tcond = (
            deterministic_fs_res.solver_results.solver.termination_condition
        )
        raise ValueError(
            "Could not solve deterministic model for capture target "
            f"{co2_capture_target}% in advance of PyROS workflow "
            f"with confidence level {confidence_level}. "
            f"Termination condition {tcond}"
            f"not among acceptable values {acceptable_det_terminations}"
        )

    # get uncertainty quantification
    uncertain_params, cov_mat = incorporate_model_uncertainty(
        model=model,
        cov_mat_infile=cov_mat_infile,
        full_uncertainty=True,
    )
    uncertainty_set = get_pyros_ellipsoidal_set(
        mean=np.array([pyo.value(param) for param in uncertain_params]),
        cov_mat=cov_mat,
        level=confidence_level / 100,
    )

    deterministic_fs_res.model_data.add_pyros_separation_priorities(
        model=model,
        prioritize_absorber_lb_flood_idxs=prioritize_absorber_lb_flood_idxs,
        prioritize_absorber_ub_flood_idxs=prioritize_absorber_ub_flood_idxs,
        prioritize_stripper_lb_flood_idxs=prioritize_stripper_lb_flood_idxs,
        prioritize_stripper_ub_flood_idxs=prioritize_stripper_ub_flood_idxs,
    )

    # do not deprioritize bounds on some temperatures
    progress_logger.info(
        f"Solving RO problem for capture rate {co2_capture_target}% "
        f"subject to confidence level {confidence_level}% . . ."
    )
    # instantiate PyROS; solve RO model
    pyros_model_data = flowsheet_model_data_type(
        original_model=model,
        scaled_model=None,
        **flowsheet_model_data_kwargs,
    )
    pyros_fs_res = pyros_model_data.solve_pyros(
        uncertainty_set=uncertainty_set,
        progress_logger=progress_logger,
        keep_model_in_results=True,
        **pyros_options,
    )

    progress_logger.info("Done. Final PyROS stats:")
    progress_logger.info(f" For confidence level {confidence_level}%")
    progress_logger.info(
        f" Solve time: {pyros_fs_res.pyros_solver_results.time}"
    )
    progress_logger.info(
        f" Iterations: {pyros_fs_res.pyros_solver_results.iterations}"
    )
    progress_logger.info(
        " Termination condition: "
        f"{pyros_fs_res.pyros_solver_results.pyros_termination_condition}"
    )

    progress_logger.info("FLOWSHEET MODEL RESULTS:")
    progress_logger.info(
        pyros_fs_res
        .to_series(include_detailed_stream_results=False)
        .to_string()
    )
    robust_termination = (
        pyros_fs_res.pyros_solver_results.pyros_termination_condition.name
        in {"robust_feasible", "robust_optimal"}
    )
    if robust_termination:
        progress_logger.debug("Robust solution was found successfully.")
        pyros_model_data.log_objective_breakdown(
            logger=progress_logger,
            level=logging.INFO,
        )
        for btype in ["lb", "ub"]:
            con_names = getattr(pyros_fs_res, f"active_{btype}_con_names")
            progress_logger.debug(f"Active {btype.upper()} Constraints:")
            progress_logger.debug(
                " " + "\n ".join(f"{con_name!r}" for con_name in con_names)
            )
        for btype in ["lb", "ub"]:
            var_names = getattr(pyros_fs_res, f"active_{btype}_var_names")
            progress_logger.debug(f"Vars Near {btype.upper()}:")
            progress_logger.debug(
                " " + "\n ".join(f"{var_name!r}" for var_name in var_names)
            )

        fs_res_type = flowsheet_model_data_type.get_flowsheet_results_type()
        if outdir is not None:
            active_cons_df, active_vars_df = (
                fs_res_type.produce_active_component_dataframes(
                    {co2_capture_target: pyros_fs_res},
                )
            )
            active_cons_df[active_cons_df.any(axis=1)].to_csv(
                os.path.join(outdir, "active_cons.csv")
            )
            active_vars_df[active_vars_df.any(axis=1)].to_csv(
                os.path.join(outdir, "vars_near_bounds.csv")
            )

    if outdir is not None:
        outfile = os.path.join(outdir, "solve_results.csv")
        pyros_fs_res.to_series(
            include_detailed_stream_results=include_detailed_stream_results,
        ).to_csv(outfile)
        progress_logger.info(
            f"Successfully serialized final PyROS solution to path {outfile!r}"
        )
        if include_plots and robust_termination:
            plot_flood_fraction_profiles(
                {co2_capture_target: pyros_fs_res},
                outfile=os.path.join(
                    outdir, "flood_fraction_profiles.png"
                ),
                logger=progress_logger,
                check_solver_termination=False,
                transpose=False,
            )
            plot_temperature_profiles(
                {co2_capture_target: pyros_fs_res},
                outfile=os.path.join(
                    outdir, "temperature_profiles.png"
                ),
                logger=progress_logger,
                check_solver_termination=False,
            )

    if acceptable_pyros_terminations_for_heatmap is None:
        acceptable_pyros_terminations = {
            pyros.pyrosTerminationCondition.robust_optimal,
            pyros.pyrosTerminationCondition.robust_feasible,
        }
    else:
        acceptable_pyros_terminations = set(
            acceptable_pyros_terminations_for_heatmap
        )

    # proceed with heatmap workflow
    all_hm_results = dict()
    evaluate_heatmap = (
        capture_targets_for_heatmaps is not None
        and pyros_fs_res.pyros_solver_results.pyros_termination_condition
        in acceptable_pyros_terminations
    )
    if evaluate_heatmap:
        solutions_dict = dict()
        solutions_dict["pyros"] = ComponentMap(
            (var, var.value)
            for var
            in pyros_model_data.original_model.component_data_objects(pyo.Var)
        )

        param_samples = get_uncertain_parameter_samples(
            param_samples_infile=heatmap_samples_infile,
            cov_mat_infile=cov_mat_infile,
        )

        if heatmap_solvers is None:
            heatmap_solvers = (
                pyros_options["local_solver"]
                + list(pyros_options.get("backup_local_solvers", []))
            )

        std_heatmap_solvers = []
        for opt in heatmap_solvers:
            if isinstance(opt, SolverWithBackup):
                std_heatmap_solvers.extend(opt.solvers)
            else:
                std_heatmap_solvers.append(opt)

        for target in capture_targets_for_heatmaps:
            progress_logger.info(
                f"Evaluating heatmap for capture target {target} "
                f"subject to {heatmap_confidence_level} confidence..."
            )
            adjust_co2_capture_target(model, target)

            all_hm_results[target], _ = evaluate_design_heatmap(
                model=pyros_model_data.original_model,
                first_stage_variables=(
                    pyros_model_data.get_first_stage_variables()
                ),
                uncertain_params=pyros_model_data.get_uncertain_params(),
                exprs_to_evaluate=(
                    pyros_model_data.get_heatmap_expressions_to_evaluate()
                ),
                solver=pyros_model_data.create_heatmap_solver_wrapper(
                    SolverWithBackup(
                        *std_heatmap_solvers,
                        logger=progress_logger,
                    ),
                    logger=progress_logger,
                ),
                cov_mat=cov_mat,
                uncertain_parameter_samples=param_samples,
                initialization_func=(
                    pyros_model_data.heatmap_initialization_func
                ),
                model_solutions=solutions_dict,
                tee=heatmap_tee,
                output_dir=os.path.join(
                    heatmap_output_dir,
                    f"{get_co2_capture_str(model, co2_capture_target)}_"
                    f"capture_{float_to_str(confidence_level)}_conf_"
                    f"dr{pyros_options.pop('decision_rule_order', 0)}"
                    f"_solution/"
                    f"{get_co2_capture_str(model, target)}_capture_"
                    f"{float_to_str(heatmap_confidence_level)}_conf/"
                ),
                nominal_parameter_values=None,
                progress_logger=progress_logger,
                export_nonoptimal_models=export_nonoptimal_heatmap_models,
            )

    return pyros_fs_res, all_hm_results


def get_uncertain_parameter_samples(
        param_samples_infile=None,
        cov_mat_infile=None,
        ):
    """
    Get uncertain parameter samples.

    If `param_samples_infile` is not provided, then 400 points
    points are pseudo-randomly drawn from the 99% confidence
    region with covariance matrix read from `cov_mat_infile`.
    The pseudo-random number generator is seeded in advance,
    so the samples should be deterministic (up to machine precision).
    The center of the confidence region is preprended to the 400
    samples.
    """
    if param_samples_infile is not None:
        # get samples for heatmap evaluation
        return pd.read_csv(param_samples_infile, index_col=0).to_numpy()
    else:
        from confidence_ellipsoid import sample_ellipsoid
        from flowsheets.mea_properties import parmest_parameters

        # get nominal point (center of the ellipsoid)
        param_names = [
            "bic_k_eq_coeff_2",
            "car_k_eq_coeff_2",
            "lwm_coeff_1",
            "lwm_coeff_2",
            "lwm_coeff_3",
            "lwm_coeff_4",
        ]
        center = np.array(
            [parmest_parameters["VLE"][name] for name in param_names]
        )

        # get covariance matrix
        cov_mat = pd.read_csv(cov_mat_infile, index_col=0).to_numpy()

        # generate samples: 400 from the 99% confidence ellipsoid
        conflvl = 0.99
        random_samples = sample_ellipsoid(
            mean=center,
            cov_mat=cov_mat,
            level=conflvl,
            rng=np.random.default_rng(123456),
            samples=400,
        )
        all_samples = np.vstack(([center], random_samples))

        return all_samples


def get_heatmap_expressions_to_evaluate(model_data):
    """
    Get model expressions to evaluate in Monte Carlo
    ("heatmap") runs.
    """
    def add_hex_exprs(hex_blk, hex_streams, other_hex_exprs):
        """
        Get heat exchanger expressions for heatmap evaluation.
        """
        hex_exprs = (
            [hex_blk.area, hex_blk.heat_duty[0]]
            + list(other_hex_exprs)
        )
        for stream in hex_streams:
            comps_to_evaluate.extend([
                stream.flow_mol,
                stream.temperature,
                stream.pressure,
                *stream.mole_frac_comp.values(),
            ])
        return hex_exprs

    original_model = model_data.original_model
    fs_blk = original_model.fs
    column_blks = [
        fs_blk.absorber_section.absorber,
        fs_blk.stripper_section.stripper,
    ]

    comps_to_evaluate = [
        fs_blk.co2_capture_target,
        fs_blk.absorber_section.absorber.co2_capture[0],
    ]

    col_port_names = [
        "liquid_inlet",
        "vapor_inlet",
        "liquid_outlet",
        "vapor_outlet",
    ]
    for col_blk in column_blks:
        comps_to_evaluate.extend([
            col_blk.length_column,
            col_blk.diameter_column,
            col_blk.parent_block().volume_column,
            col_blk.parent_block().volume_column_withheads,
            col_blk.parent_block().HDratio,
            *col_blk.parent_block().LGratio.values(),
        ])
        for port_name in col_port_names:
            port = getattr(col_blk, port_name)
            comps_to_evaluate.extend([
                port.flow_mol[0],
                port.temperature[0],
                port.pressure[0],
                *port.mole_frac_comp.values(),
            ])
        comps_to_evaluate.extend(col_blk.flood_fraction.values())
        comps_to_evaluate.extend(
            liq_prop_blk.temperature
            for liq_prop_blk in col_blk.liquid_phase.properties.values()
        )
        comps_to_evaluate.extend(
            vap_prop_blk.temperature
            for vap_prop_blk in col_blk.vapor_phase.properties.values()
        )

    lrhx = fs_blk.absorber_section.lean_rich_heat_exchanger
    reb = fs_blk.stripper_section.reboiler
    cond = fs_blk.stripper_section.condenser
    comps_to_evaluate.extend(add_hex_exprs(
        hex_blk=lrhx,
        hex_streams=[
            lrhx.cold_side.properties_in[0],
            lrhx.cold_side.properties_out[0],
            lrhx.hot_side.properties_in[0],
            lrhx.hot_side.properties_out[0],
        ],
        other_hex_exprs=[],
    ))
    comps_to_evaluate.extend(add_hex_exprs(
        hex_blk=reb,
        hex_streams=[
            reb.liquid_phase.properties_in[0],
            reb.liquid_phase.properties_out[0],
            reb.vapor_phase[0],
        ],
        other_hex_exprs=[
            reb.steam_temperature_inlet,
            reb.steam_flow_mol[0],
            reb.condensate_temperature_outlet[0],
        ],
    ))
    comps_to_evaluate.extend(add_hex_exprs(
        hex_blk=cond,
        hex_streams=[
            cond.vapor_phase.properties_in[0],
            cond.vapor_phase.properties_out[0],
            cond.liquid_phase[0],
        ],
        other_hex_exprs=[
            cond.cooling_flow_mol[0],
            cond.cooling_water_temperature_inlet,
            cond.cooling_water_temperature_outlet[0],
            fs_blk.condenser_cw_temperature_outlet_excess_temp,
        ],
    ))

    pump_blks = [
        fs_blk.absorber_section.rich_solvent_pump,
        fs_blk.absorber_section.lean_solvent_pump,
    ]
    for pump_blk in pump_blks:
        comps_to_evaluate.extend([
            pump_blk.control_volume.work[0],
            pump_blk.control_volume.properties_in[0].flow_mol,
            pump_blk.control_volume.properties_in[0].flow_vol,
            pump_blk.control_volume.properties_in[0].temperature,
            pump_blk.control_volume.properties_in[0].pressure,
            pump_blk.control_volume.properties_out[0].flow_vol,
            pump_blk.control_volume.properties_out[0].temperature,
            pump_blk.control_volume.properties_out[0].pressure,
            pump_blk.deltaP[0],
            pump_blk.control_volume.work[0],
            pump_blk.max_work,
        ])

    comps_to_evaluate.extend([
        fs_blk.mea_recirculation_rate[0],
        fs_blk.makeup.flow_mol[0],
        fs_blk.makeup.temperature[0],
        fs_blk.makeup.pressure[0],
        *fs_blk.makeup.mole_frac_comp.values(),
        fs_blk.rich_loading[0],
        fs_blk.lean_loading[0],
        fs_blk.obj,
        fs_blk.capex_cost,
        fs_blk.opex_cost,
        *fs_blk.capex_cost_terms.values(),
        *fs_blk.opex_cost_terms.values(),
    ])

    acceptable_comp_types = (
        ParamData,
        VarData,
        ExpressionData,
        ObjectiveData,
        VarLikeExpressionData,
    )
    assert all(
        isinstance(comp, acceptable_comp_types) for comp in comps_to_evaluate
    )

    return {
        comp.name: comp
        for comp in ComponentSet(comps_to_evaluate)
    }


def heatmap_workflow(
        outdir,
        samples_infile,
        cov_mat_infile,
        capture_targets=None,
        solvers_and_options=None,
        logger=default_logger,
        tee=False,
        export_nonoptimal_models=False,
        **build_and_solve_kwargs,
        ):
    """
    Deterministic heatmap workflow.

    Parameters
    ----------
    outdir : path-like
        Directory to which to output all heatmap results.
    capture_rates : dict, optional
        Mapping each capture rate for which to obtain determinstic
        solution to a list of capture rates under which to
        assess robustness of the solution.
        Capture rates are in % (0-100).
        Default is
        ``{85.0: [85.0,...,97.0], ..., 97.0: [85.0,...,97.0]}``.
    solvers_and_options : list of tuples, optional
        List of 2-tuples, each of which has a
        Pyomo `SolverFactory` keys as the first entry and
        a dict of solver options as the second entry.
        If None is passed, then we use ``[(None, None)]``.
    samples_infile : path-like, optional
        Path to CSV of ucnertain parameter samples for heatmap
        evaluation.
    logger : logging.Logger, optional
        Progress logger.

    Returns
    -------
    all_hm_results : dict
        Nested dict. Each capture rate `cap_rate` in
        ``capture_rates.keys()`` is mapped to a dict which itself
        maps each capture rate in ``capture_rates[cap_rate]``
        to a heatmap results object.
    """
    logger.info(f"Starting {heatmap_workflow.__name__!r}...")
    logger.info(f"Invoked at UTC: {datetime.datetime.utcnow().isoformat()}")
    log_main_dependency_module_info(logger=logger, level=logging.INFO)

    # set up default capture rates
    if capture_targets is None:
        targets = [85.0, 87.5, 90.0, 92.5, 95.0, 97.0]
        capture_targets = {target: targets.copy() for target in targets}

    # set up the solvers
    solvers = []
    if solvers_and_options is None:
        solvers_and_options = [(None, None)]
    for idx, (solver, options) in enumerate(solvers_and_options):
        if idx == 0:
            main_solver = solver
            main_solver_options = options
        solvers.append(get_solver(solver, options))

    all_hm_results = dict()

    # parameter samples for heatmap
    param_samples = get_uncertain_parameter_samples(
        param_samples_infile=samples_infile,
        cov_mat_infile=cov_mat_infile,
    )

    for des_target, eval_targets in capture_targets.items():
        logger.info(f"Getting solution for capture target {des_target}%")
        fs_res = build_and_solve_flowsheet_model(
            solver=main_solver,
            solver_options=main_solver_options,
            co2_capture_targets=des_target,
            keep_model_in_results=True,
            tee=True,
            progress_logger=logger,
            **build_and_solve_kwargs,
        )
        model_data = fs_res.model_data
        logger.info("Done.")

        # DOF info, uncertainty
        _, cov_mat = incorporate_model_uncertainty(
            model=model_data.original_model,
            cov_mat_infile=cov_mat_infile,
            full_uncertainty=True,
        )

        capture_str = get_co2_capture_str(
            model_data.original_model,
            model_data.original_model.fs.co2_capture_target,
        )

        if not pyo.check_optimal_termination(fs_res.solver_results):
            logger.info(
                "Deterministic model not solved successfully for "
                f"design target {des_target}%."
            )
            continue

        cap_hm_res_dict = dict()
        for eval_target in eval_targets:
            logger.info(
                "Evaluating heatmap of solution for design target "
                f"{des_target} subject to new capture target {eval_target}"
            )
            problem_str = f"{float_to_str(eval_target)}_percent_problem"
            model_data.adjust_co2_capture_target(eval_target)

            # get samples for heatmap evaluation
            hm_res_list, _ = evaluate_design_heatmap(
                model=model_data.original_model,
                first_stage_variables=model_data.get_first_stage_variables(),
                uncertain_params=model_data.get_uncertain_params(),
                solver=model_data.create_heatmap_solver_wrapper(
                    solver=SolverWithBackup(*solvers, logger=logger),
                    logger=logger,
                ),
                backup_solvers=None,
                cov_mat=cov_mat,
                uncertain_parameter_samples=param_samples,
                initialization_func=model_data.heatmap_initialization_func,
                model_solutions={
                    "deterministic": ComponentMap(
                        (var, var.value)
                        for var in
                        model_data.original_model.component_data_objects(
                            pyo.Var
                        )
                    )
                },
                output_dir=os.path.join(
                    outdir,
                    f"{capture_str}_solution",
                    f"{problem_str}",
                ),
                nominal_parameter_values=None,
                progress_logger=logger,
                exprs_to_evaluate=(
                    model_data.get_heatmap_expressions_to_evaluate()
                ),
                tee=tee,
                export_nonoptimal_models=export_nonoptimal_models,
            )
            cap_hm_res_dict[eval_target] = hm_res_list[0]

        all_hm_results[des_target] = cap_hm_res_dict

    logger.info(f"Done {heatmap_workflow.__name__!r}.")

    return all_hm_results


def plot_solve_results_conf_lvl_sens(
        results_df,
        outdir,
        quantity_plot_data_list,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Plot solve results.
    """
    acceptable_terminations = {
        "optimal",
        "globallyOptimal",
        "locallyOptimal",
    }
    acceptable_pyros_terminations = {
        "pyrosTerminationCondition.robust_feasible",
        "pyrosTerminationCondition.robust_optimal",
    }
    mpl_rc_params = DEFAULT_MPL_RC_PARAMS.copy()
    mpl_rc_params.update({"font.size": 12})
    are_all_conf_lvls_int = all(
        clvl == int(clvl) for clvl in results_df.index.get_level_values(1)
    )
    with plt.rc_context(mpl_rc_params):
        # filter by acceptability of solver termination
        is_deterministic_and_acceptable = (
            (results_df.index.get_level_values(1) == 0)
            & results_df.solver_termination.termination_condition.isin(
                acceptable_terminations
            )
        )
        is_pyros_and_acceptable = (
            (results_df.index.get_level_values(1) > 0)
            & (
                results_df.pyros_solver_termination
                .pyros_termination_condition.isin(
                    acceptable_pyros_terminations
                )
            )
        )
        acceptable_results_df = results_df[
            is_deterministic_and_acceptable
            | is_pyros_and_acceptable
        ]
        for qty_plot_data in quantity_plot_data_list:
            default_logger.info(
                "Plotting confidence level sensitivity results "
                f"for {qty_plot_data.qty_name!r}..."
            )
            if qty_plot_data.col_to_plot not in acceptable_results_df.columns:
                default_logger.warning(
                    f"Could not plot conf lvl sensitivity results "
                    f"for quantity {qty_plot_data.qty_unit!r}, "
                    f"as the column {qty_plot_data.col_to_plot!r} "
                    "is not present in the results "
                    "spreadsheet/DataFrame. Skipping..."
                )
                continue
            # we don't use index.levels[0] since that has been
            # frozen, so filtering out the acceptable rows
            # doesn't also filter out the levels
            unique_targets = np.unique(
                acceptable_results_df.index.get_level_values(0)
            )
            for cap_targ in unique_targets:
                fig, ax = plt.subplots(figsize=(3.1, 2.8))
                ax.grid(axis="x")
                ax.set_xlabel(wrap_quantity_str(
                    namestr=(
                        "Magnification Factor" if label_conf_lvls_as_stdevs
                        else "Confidence Level"
                    ),
                    unitstr=None if label_conf_lvls_as_stdevs else "\\%",
                ))
                ax.set_ylabel(wrap_quantity_str(
                    namestr=qty_plot_data.qty_name,
                    unitstr=qty_plot_data.qty_unit,
                ))

                yvals = (
                    qty_plot_data.post_mult_offset
                    + qty_plot_data.multiplier
                    * (
                        acceptable_results_df.loc[cap_targ][
                            qty_plot_data.col_to_plot
                        ].astype(float)
                        + qty_plot_data.pre_mult_offset
                    )
                )
                ax.bar(
                    range(yvals.index.size),
                    yvals,
                    color="gray",
                    width=0.5,
                )
                tick_labels = []
                for clvl in yvals.index:
                    if clvl == 0:
                        label = "0 (det.)"
                    else:
                        label = get_conf_lvl_plot_label(
                            conf_lvl=clvl,
                            n=6,
                            decimals=int(not are_all_conf_lvls_int),
                            conf_lvl_suffix="",
                            as_mag_factor=label_conf_lvls_as_stdevs,
                        )
                    tick_labels.append(label)

                ax.set_xticks(range(yvals.index.size), tick_labels)

                co2_capture_str = get_co2_capture_str(None, cap_targ)
                fig.savefig(
                    os.path.join(
                        outdir,
                        (
                            f"plot_conf_lvl_sens_"
                            f"{qty_plot_data.fname}_{co2_capture_str}"
                            f".{output_plot_fmt}"
                        )
                    ),
                    bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                )
                plt.close(fig)


def plot_solve_results_capture_target_sens(
        results_df,
        outdir,
        quantity_plot_data_list,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Plot solve results.
    """
    acceptable_terminations = {
        "optimal",
        "globallyOptimal",
        "locallyOptimal",
    }
    acceptable_pyros_terminations = {
        "pyrosTerminationCondition.robust_feasible",
        "pyrosTerminationCondition.robust_optimal",
    }
    mpl_rc_params = DEFAULT_MPL_RC_PARAMS.copy()
    mpl_rc_params.update({"font.size": 12, "legend.fontsize": 11})
    are_all_conf_lvls_int = all(
        clvl == int(clvl) for clvl in results_df.index.get_level_values(1)
    )
    with plt.rc_context(mpl_rc_params):
        # filter by acceptability of solver termination
        is_deterministic_and_acceptable = (
            (results_df.index.get_level_values(1) == 0)
            & results_df.solver_termination.termination_condition.isin(
                acceptable_terminations
            )
        )
        is_pyros_and_acceptable = (
            (results_df.index.get_level_values(1) > 0)
            & (
                results_df.pyros_solver_termination
                .pyros_termination_condition.isin(
                    acceptable_pyros_terminations
                )
            )
        )
        acceptable_results_df = results_df[
            is_deterministic_and_acceptable
            | is_pyros_and_acceptable
        ]
        reordered_df = acceptable_results_df.reorder_levels([1, 0])
        for qty_plot_data in quantity_plot_data_list:
            default_logger.info(
                "Plotting capture target sensitivity results "
                f"for quantity name {qty_plot_data.qty_name!r}..."
            )
            if qty_plot_data.col_to_plot not in reordered_df.columns:
                default_logger.warning(
                    f"Could not plot solve results for quantity "
                    f"{qty_plot_data.qty_name!r}, "
                    f"as the column {qty_plot_data.col_to_plot!r} "
                    "is not present in the results "
                    "spreadsheet/DataFrame. Skipping..."
                )
                continue

            fig, ax = plt.subplots(figsize=(3.1, 3.5))
            all_yvals = (
                qty_plot_data.multiplier
                * (
                    reordered_df[qty_plot_data.col_to_plot].astype(float)
                    + qty_plot_data.pre_mult_offset
                )
                + qty_plot_data.post_mult_offset
            )

            ax.grid(False)  # no grid in main plot
            if qty_plot_data.include_inset:
                # get yaxis bound for inset
                inset_targ_lb, inset_targ_ub = 89.5, 98.5
                if qty_plot_data.inset_targ_lims is not None:
                    inset_targ_lb, inset_targ_ub = (
                        qty_plot_data.inset_targ_lims
                    )
                inset_yvals = all_yvals[
                    (all_yvals.index.get_level_values(1) <= inset_targ_ub)
                    & (all_yvals.index.get_level_values(1) >= inset_targ_lb)
                ]
                ylb = inset_yvals.min()
                yub = inset_yvals.max()

                # set up the inset axis
                axin = ax.inset_axes(
                    [0.26, 0.5, 0.4, 0.4],
                    xlim=(inset_targ_lb, inset_targ_ub),
                    ylim=(ylb - (yub - ylb) * 0.1, yub + (yub - ylb) * 0.1),
                    xticks=[
                        inset_yvals.index.get_level_values(1).min(),
                        inset_yvals.index.get_level_values(1).max(),
                    ],
                )
                axin.set_title("inset")

            all_xtick_labels = set()
            # we don't use index.levels[0] since that has been
            # frozen, so filtering out the acceptable rows
            # doesn't also filter out the levels
            unique_conf_lvls = reordered_df.index.get_level_values(0).unique()
            for conf_lvl in unique_conf_lvls:
                if conf_lvl == 0:
                    label = "det."
                else:
                    label = get_conf_lvl_plot_label(
                        conf_lvl=conf_lvl,
                        n=6,
                        decimals=int(not are_all_conf_lvls_int),
                        as_mag_factor=label_conf_lvls_as_stdevs,
                    )

                ax.set_xlabel(wrap_quantity_str(
                    namestr="$\\text{CO}_2$ Capture Target",
                    unitstr="\\%",
                ))
                ax.set_ylabel(wrap_quantity_str(
                    namestr=qty_plot_data.qty_name,
                    unitstr=qty_plot_data.qty_unit,
                ))

                all_xtick_labels.update(all_yvals.loc[conf_lvl].index)
                if qty_plot_data.include_inset:
                    axin.plot(
                        all_yvals.loc[conf_lvl].index,
                        all_yvals.loc[conf_lvl],
                        markersize=(
                            DEFAULT_MPL_RC_PARAMS["lines.markersize"] * 2/3
                        ),
                        linewidth=(
                            DEFAULT_MPL_RC_PARAMS["lines.linewidth"] * 2/3
                        ),
                    )
                ax.plot(
                    reordered_df.loc[conf_lvl].index,
                    all_yvals.loc[conf_lvl],
                    label=label,
                    clip_on=False,
                )
            ax.legend(
                bbox_to_anchor=(0.5, 1.05),
                loc="lower center",
                ncol=2,
                columnspacing=1,
            )
            set_nonoverlapping_fixed_xticks(
                fig, ax, sorted(list(all_xtick_labels)),
            )
            if qty_plot_data.include_inset:
                inset_xlb, inset_xub = axin.get_xlim()
                inset_xtick_labels = sorted([
                    label for label in all_xtick_labels
                    if inset_xlb <= label <= inset_xub
                ])
                set_nonoverlapping_fixed_xticks(fig, axin, inset_xtick_labels)
            fig.savefig(
                os.path.join(
                    outdir,
                    f"plot_results_{qty_plot_data.fname}.{output_plot_fmt}"
                ),
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
            plt.close(fig)


class QuantityPlotData:
    def __init__(
            self,
            qty_name,
            qty_unit,
            fname,
            multiplier,
            col_to_plot,
            pre_mult_offset=0,
            post_mult_offset=0,
            include_inset=False,
            inset_targ_lims=None,
            ):
        self.qty_name = qty_name
        self.qty_unit = qty_unit
        self.fname = fname
        self.multiplier = multiplier
        self.col_to_plot = col_to_plot
        self.pre_mult_offset = pre_mult_offset
        self.post_mult_offset = post_mult_offset
        self.include_inset = include_inset
        self.inset_targ_lims = inset_targ_lims


class GroupedPlotData:
    def __init__(
            self,
            group_name,
            multiplier,
            col_to_group,
            pre_mult_offset=0,
            post_mult_offset=0,
            ):
        self.group_name = group_name
        self.multiplier = multiplier
        self.col_to_group = col_to_group
        self.pre_mult_offset = pre_mult_offset
        self.post_mult_offset = post_mult_offset


class GroupedPlotDataContainer:
    def __init__(
            self,
            fname=None,
            qty_name=None,
            qty_unit=None,
            group_data_list=None,
            legend_kwargs=None,
            ):
        """
        Initialize self (see class docstring).
        """
        self.fname = fname
        self.qty_name = qty_name
        self.qty_unit = qty_unit
        self.group_data_list = group_data_list
        self.legend_kwargs = legend_kwargs


class StackedPlotData:
    def __init__(
            self,
            fname,
            cols_to_stack,
            other_cols_to_stack,
            qty_name,
            qty_units,
            label_names,
            col_multipliers,
            pre_mult_offsets=0,
            post_mult_offsets=0,
            ):
        """
        Initialize self (see class docstring).
        """
        self.fname = fname
        self.cols_to_stack = cols_to_stack
        self.other_cols_to_stack = other_cols_to_stack
        self.qty_name = qty_name
        self.qty_units = qty_units
        self.label_names = label_names
        self.col_multipliers = col_multipliers
        self.pre_mult_offsets = pre_mult_offsets
        self.post_mult_offsets = post_mult_offsets


def plot_solve_results_stacked_conf_lvl_sens(
        results_df,
        outdir,
        stacked_plot_data_list,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    For each capture target, plot response of solution to the
    ellipsoidal confidence level.
    """
    acceptable_terminations = {
        "optimal",
        "globallyOptimal",
        "locallyOptimal",
    }
    acceptable_pyros_terminations = {
        "pyrosTerminationCondition.robust_feasible",
        "pyrosTerminationCondition.robust_optimal",
    }
    are_all_conf_lvls_int = all(
        clvl == int(clvl) for clvl in results_df.index.get_level_values(1)
    )

    custom_rc_params = DEFAULT_MPL_RC_PARAMS.copy()
    custom_rc_params["font.size"] = 12
    custom_rc_params["legend.fontsize"] = 10
    with plt.rc_context(custom_rc_params):
        # filter by acceptability of solver termination
        is_deterministic_and_acceptable = (
            (results_df.index.get_level_values(1) == 0)
            & results_df.solver_termination.termination_condition.isin(
                acceptable_terminations
            )
        )
        is_pyros_and_acceptable = (
            (results_df.index.get_level_values(1) > 0)
            & (
                results_df.pyros_solver_termination
                .pyros_termination_condition.isin(
                    acceptable_pyros_terminations
                )
            )
        )
        acceptable_results_df = results_df[
            is_deterministic_and_acceptable
            | is_pyros_and_acceptable
        ]

        for stacked_plot_data in stacked_plot_data_list:
            default_logger.info(
                "Plotting stacked confidence level sensitivity results "
                f"for {stacked_plot_data.qty_name!r}..."
            )

            stack_cardinality = (
                len(stacked_plot_data.cols_to_stack)
                + bool(stacked_plot_data.other_cols_to_stack)
            )

            # we don't use index.levels[0] since that has been
            # frozen, so filtering out the acceptable rows
            # doesn't also filter out the levels
            unique_targets = np.unique(
                acceptable_results_df.index.get_level_values(0)
            )
            for targ in unique_targets:
                targ_subdf = acceptable_results_df.loc[
                    targ,
                    (
                        stacked_plot_data.cols_to_stack
                        + stacked_plot_data.other_cols_to_stack
                    )
                ].astype(float)
                yvals = (
                    stacked_plot_data.col_multipliers
                    * (targ_subdf + stacked_plot_data.pre_mult_offsets)
                    + stacked_plot_data.post_mult_offsets
                )
                fig, ax = plt.subplots(
                    figsize=(
                        3.4,
                        3.1 + 0.1 * np.ceil(stack_cardinality - 2),
                    ),
                )
                ax.set_ylabel(wrap_quantity_str(
                    namestr=stacked_plot_data.qty_name,
                    unitstr=stacked_plot_data.qty_units,
                    width=30,
                ))
                ax.grid(axis="x")

                cmap_obj = plt.get_cmap("cubehelix")
                bar_colors = cmap_obj(np.linspace(0.1, 0.8, stack_cardinality))

                bar_bases = np.zeros(targ_subdf.index.size)
                plot_zip = zip(
                    stacked_plot_data.cols_to_stack,
                    stacked_plot_data.label_names,
                    (
                        bar_colors[:-1]
                        if stacked_plot_data.other_cols_to_stack
                        else bar_colors
                    ),
                )
                for col_name, label_name, color in plot_zip:
                    bar_heights = yvals[col_name].values
                    ax.bar(
                        range(yvals.index.size),
                        bar_heights,
                        width=0.5,
                        bottom=bar_bases,
                        label=label_name,
                        color=color,
                    )
                    bar_bases += bar_heights

                if stacked_plot_data.other_cols_to_stack:
                    other_heights = yvals[
                        stacked_plot_data.other_cols_to_stack
                    ].sum(axis=1).values
                    ax.bar(
                        range(yvals.index.size),
                        other_heights,
                        width=0.5,
                        bottom=bar_bases,
                        label="other",
                        color=bar_colors[-1],
                    )

                ax.legend(
                    bbox_to_anchor=(0.5, 1.05),
                    loc="lower center",
                    ncol=2,
                )

                tick_labels = []
                for clvl in yvals.index:
                    if clvl == 0:
                        label = "0 (det.)"
                    else:
                        label = get_conf_lvl_plot_label(
                            conf_lvl=clvl,
                            n=6,
                            decimals=int(not are_all_conf_lvls_int),
                            as_mag_factor=label_conf_lvls_as_stdevs,
                            conf_lvl_suffix="",
                        )
                    tick_labels.append(label)

                ax.set_xticks(range(yvals.index.size), tick_labels)
                ax.set_xlabel(wrap_quantity_str(
                    namestr=(
                        "Magnification Factor"
                        if label_conf_lvls_as_stdevs else "Confidence Level"
                    ),
                    unitstr=None if label_conf_lvls_as_stdevs else "\\%",
                ))

                targ_str = get_co2_capture_str(None, targ)
                fig.savefig(
                    os.path.join(
                        outdir,
                        (
                            f"plot_results_{stacked_plot_data.fname}_"
                            f"{targ_str}.{output_plot_fmt}"
                        ),
                    ),
                    dpi=300,
                    transparent=True,
                    bbox_inches="tight",
                )
                plt.close(fig)


def plot_solve_results_grouped_conf_lvl_sens(
        results_df,
        outdir,
        grouped_plot_data_list,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    For each capture target, plot response of solution to the
    ellipsoidal confidence level.
    """
    acceptable_terminations = {
        "optimal",
        "globallyOptimal",
        "locallyOptimal",
    }
    acceptable_pyros_terminations = {
        "pyrosTerminationCondition.robust_feasible",
        "pyrosTerminationCondition.robust_optimal",
    }
    are_all_conf_lvls_int = all(
        clvl == int(clvl) for clvl in results_df.index.get_level_values(1)
    )

    custom_rc_params = DEFAULT_MPL_RC_PARAMS.copy()
    custom_rc_params["font.size"] = 11
    with plt.rc_context(custom_rc_params):
        # filter by acceptability of solver termination
        is_deterministic_and_acceptable = (
            (results_df.index.get_level_values(1) == 0)
            & results_df.solver_termination.termination_condition.isin(
                acceptable_terminations
            )
        )
        is_pyros_and_acceptable = (
            (results_df.index.get_level_values(1) > 0)
            & (
                results_df.pyros_solver_termination
                .pyros_termination_condition.isin(
                    acceptable_pyros_terminations
                )
            )
        )
        acceptable_results_df = results_df[
            is_deterministic_and_acceptable
            | is_pyros_and_acceptable
        ]
        for grouped_plot_data in grouped_plot_data_list:
            fname = grouped_plot_data.fname
            qty_name = grouped_plot_data.qty_name
            qty_unit = grouped_plot_data.qty_unit
            legend_kwargs = grouped_plot_data.legend_kwargs
            group_data_list = grouped_plot_data.group_data_list
            if legend_kwargs is None:
                legend_kwargs = dict(
                    bbox_to_anchor=(0, -0.25),
                    loc="upper left",
                    ncol=1,
                )
            default_logger.info(
                "Plotting grouped confidence level sensitivity results "
                f"for {qty_name!r}..."
            )
            labels = [gdata.group_name for gdata in group_data_list]
            mults = [gdata.multiplier for gdata in group_data_list]
            pre_offsets = [gdata.pre_mult_offset for gdata in group_data_list]
            post_offsets = [
                gdata.post_mult_offset for gdata in group_data_list
            ]
            cols = [gdata.col_to_group for gdata in group_data_list]

            # we don't use index.levels[0] since that has been
            # frozen, so filtering out the acceptable rows
            # doesn't also filter out the levels
            unique_targets = np.unique(
                acceptable_results_df.index.get_level_values(0)
            )
            for targ in unique_targets:
                targ_subdf = acceptable_results_df.loc[targ, cols]
                yvals = (
                    mults * (targ_subdf.astype(float) + pre_offsets)
                    + post_offsets
                ).T

                fig, ax = plt.subplots(figsize=(3.4, 3.2))
                ax.set_ylabel(wrap_quantity_str(qty_name, qty_unit, 30))
                ax.grid(axis="x")

                cmap_obj = plt.get_cmap("PuBu_r")
                bar_colors = cmap_obj(
                    np.linspace(0.1, 0.8, yvals.columns.size)
                )

                bar_width = 1 / (2 * yvals.columns.size)
                for clvl_idx, conf_lvl in enumerate(yvals.columns):
                    bar_x_positions = [
                        qty_idx
                        + bar_width * (clvl_idx + 0.5 - yvals.columns.size / 2)
                        for qty_idx, _ in enumerate(yvals.index)
                    ]
                    bar_heights = yvals[conf_lvl].to_numpy()

                    if conf_lvl == 0:
                        label = "det."
                    else:
                        label = get_conf_lvl_plot_label(
                            conf_lvl=conf_lvl,
                            decimals=int(not are_all_conf_lvls_int),
                            n=6,
                            as_mag_factor=label_conf_lvls_as_stdevs,
                        )

                    ax.bar(
                        bar_x_positions,
                        bar_heights,
                        width=0.9 * bar_width,
                        label=label,
                        color=bar_colors[clvl_idx],
                    )

                ax.legend(
                    bbox_to_anchor=(0.5, 1.05),
                    loc="lower center",
                    ncol=2,
                )
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels([
                    wrap_quantity_str(label, None, 11)
                    for label in labels
                ])

                targ_str = get_co2_capture_str(None, targ)
                fig.savefig(
                    os.path.join(
                        outdir,
                        f"plot_results_{fname}_{targ_str}.{output_plot_fmt}"
                    ),
                    dpi=300,
                    transparent=True,
                    bbox_inches="tight",
                )
                plt.close(fig)


def plot_solver_performance_heatmaps(
        results_df,
        outdir,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Plot PyROS solver performance heatmaps.
    """
    pyros_term_status_order = [
        pyros.pyrosTerminationCondition.robust_optimal,
        pyros.pyrosTerminationCondition.robust_feasible,
        pyros.pyrosTerminationCondition.robust_infeasible,
        pyros.pyrosTerminationCondition.subsolver_error,
        pyros.pyrosTerminationCondition.max_iter,
        pyros.pyrosTerminationCondition.time_out,
    ]

    solve_times_df = pd.DataFrame(
        index=results_df.index.levels[0],
        columns=results_df.index.levels[1],
    )
    iterations_df = pd.DataFrame(
        index=results_df.index.levels[0],
        columns=[
            conf_lvl
            for conf_lvl in results_df.index.levels[1]
            if conf_lvl != 0
        ],
    )
    term_status_df = pd.DataFrame(
        index=results_df.index.levels[0],
        columns=[
            conf_lvl
            for conf_lvl in results_df.index.levels[1]
            if conf_lvl != 0
        ],
    )

    term_statuses_in_df = sorted(
        [
            pyros.pyrosTerminationCondition[term_status_str.split(".")[-1]]
            for term_status_str in (
                results_df[results_df.index.get_level_values(1) > 0]
                .pyros_solver_termination
                .pyros_termination_condition
                .unique()
            )
        ],
        key=lambda status: pyros_term_status_order.index(status)
    )
    are_all_conf_lvls_int = all(
        clvl == int(clvl)
        for clvl in results_df.index.get_level_values(1).unique()
    )

    for capture_target, conf_lvl in results_df.index:
        if conf_lvl == 0:
            solve_times_df.loc[capture_target, conf_lvl] = float(
                results_df.loc[capture_target, conf_lvl]
                .solver_termination
                .total_solve_time
            ) / 60
        else:
            solve_times_df.loc[capture_target, conf_lvl] = float(
                results_df.loc[capture_target, conf_lvl]
                .pyros_solver_termination
                .total_solve_time
            ) / 60
            iterations_df.loc[capture_target, conf_lvl] = int(
                results_df.loc[capture_target, conf_lvl]
                .pyros_solver_termination
                .iterations
            )
            term_status_df.loc[capture_target, conf_lvl] = (
                term_statuses_in_df.index(
                    pyros.pyrosTerminationCondition[
                        results_df.loc[capture_target, conf_lvl]
                        .pyros_solver_termination
                        .pyros_termination_condition
                        .split(".")[-1]
                    ]
                )
            )

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # plot solve times first
    rc_params = DEFAULT_MPL_RC_PARAMS.copy()
    rc_params.update({"font.size": 9})
    with mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(
            3.1, 0.3 * solve_times_df.index.size + 0.2
        ))
        ax.grid(False)
        im, _ = heatmap(
            data=solve_times_df.to_numpy().astype(float),
            row_labels=solve_times_df.index,
            col_labels=[
                get_conf_lvl_plot_label(
                    conf_lvl=clvl,
                    n=6,
                    decimals=int(not are_all_conf_lvls_int),
                    as_mag_factor=label_conf_lvls_as_stdevs,
                    conf_lvl_suffix="",
                    mag_factor_suffix=""
                )
                for clvl in solve_times_df.columns
            ],
            ax=ax,
            xlabel=(
                "Magnification Factor ($\\boldsymbol\\sigma$)"
                if label_conf_lvls_as_stdevs
                else "Confidence Level ($\\%$)"
            ),
            ylabel="$\\mathrm{CO}_2$ Capture Target ($\\%$)",
            cmap="plasma_r",
            cbarlabel="Solver Wall-Clock Time (min)",
        )
        annotate_heatmap(im=im, valfmt="{x:.1f}", fontsize=8)
        fig.savefig(
            os.path.join(outdir, f"solve_times.{output_plot_fmt}"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

        # now plot iterations
        fig, ax = plt.subplots(figsize=(
            1.1 + 0.5 * iterations_df.shape[1],
            0.3 * iterations_df.index.size + 0.2
        ))
        ax.grid(False)
        min_iters = int(np.nanmin(iterations_df.values))
        max_iters = int(np.nanmax(iterations_df.values))
        iter_range = max_iters - min_iters
        im, _ = heatmap(
            iterations_df.to_numpy().astype(float),
            row_labels=iterations_df.index,
            col_labels=[
                get_conf_lvl_plot_label(
                    conf_lvl=clvl,
                    n=6,
                    decimals=int(not are_all_conf_lvls_int),
                    as_mag_factor=label_conf_lvls_as_stdevs,
                    conf_lvl_suffix="",
                    mag_factor_suffix="",
                )
                for clvl in iterations_df.columns
            ],
            ax=ax,
            xlabel=(
                "Magnification Factor ($\\boldsymbol\\sigma$)"
                if label_conf_lvls_as_stdevs
                else "Confidence Level ($\\%$)"
            ),
            ylabel="$\\mathrm{CO}_2$ Capture Target ($\\%$)",
            cmap=mpl.colormaps["plasma_r"].resampled(
                iter_range + 1,
            ),
            cbarlabel="Number of PyROS Iterations",
            norm=mpl.colors.BoundaryNorm(
                np.linspace(min_iters - 0.5, max_iters + 0.5, iter_range + 2),
                ncolors=iter_range + 1,
            ),
            cbar_kw=dict(ticks=np.arange(min_iters, max_iters + 1)),
        )
        annotate_heatmap(im=im, valfmt="{x:.0f}", fontsize=8)
        fig.savefig(
            os.path.join(outdir, f"pyros_iterations.{output_plot_fmt}"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

        # now plot termination status
        fig, ax = plt.subplots(figsize=(
            1.1 + 0.5 * term_status_df.shape[1],
            0.3 * term_status_df.index.size + 0.2
        ))
        ax.grid(False)
        min_status_int = int(np.nanmin(term_status_df.values))
        max_status_int = int(np.nanmax(term_status_df.values))
        status_int_range = max_status_int - min_status_int
        im, cbar = heatmap(
            term_status_df.to_numpy().astype(float),
            row_labels=term_status_df.index,
            col_labels=[
                get_conf_lvl_plot_label(
                    conf_lvl=clvl,
                    n=6,
                    decimals=int(not are_all_conf_lvls_int),
                    as_mag_factor=label_conf_lvls_as_stdevs,
                    conf_lvl_suffix="",
                    mag_factor_suffix="",
                )
                for clvl in term_status_df.columns
            ],
            ax=ax,
            xlabel=(
                "Magnification Factor ($\\boldsymbol\\sigma$)"
                if label_conf_lvls_as_stdevs
                else "Confidence Level ($\\%$)"
            ),
            ylabel="$\\mathrm{CO}_2$ Capture Target ($\\%$)",
            cmap=mpl.colormaps["plasma_r"].resampled(
                status_int_range + 1,
            ),
            cbarlabel="PyROS Termination Condition",
            norm=mpl.colors.BoundaryNorm(
                np.linspace(
                    min_status_int - 0.5,
                    max_status_int + 0.5,
                    status_int_range + 2,
                ),
                ncolors=status_int_range + 1,
            ),
            cbar_kw=dict(ticks=np.arange(min_status_int, max_status_int + 1)),
        )
        # note: we don't annotate this plot
        cbar.ax.set_yticklabels(
            [st.name.replace("_", " ") for st in term_statuses_in_df]
        )
        fig.savefig(
            os.path.join(outdir, f"pyros_termination_cond.{output_plot_fmt}"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def get_pyros_timing_breakdown(logfile):
    timing_breakdown_start = "Timing breakdown"
    data_ids = [
        f"pyros_{stat}" for stat in
        ("ncalls", "cumtime", "percall", "percent")
    ]
    timing_ids = [
        "main",
        "preprocessing",
        "master_feasibility",
        "dr_polishing",
        "master",
        "other",
        "local_separation",
        "global_separation",
    ]
    with open(logfile) as f:
        solver_log_str = f.read()

    timing_ser = pd.Series(
        index=pd.MultiIndex.from_product([data_ids, timing_ids]),
        dtype="float",
    )

    solver_log_lines = solver_log_str.split("\n")
    try:
        timing_breakdown_start_idx = next(
            idx for (idx, line) in enumerate(solver_log_lines)
            if timing_breakdown_start in line
        )
    except StopIteration:
        print(f"No timing breakdown found in logfile {logfile} ")
        raise
    timing_breakdown_lines = solver_log_lines[timing_breakdown_start_idx:]
    main_line = timing_breakdown_lines[4]
    other_line_idx = next(
        idx for (idx, line) in enumerate(timing_breakdown_lines)
        if "other" in line and "n/a" in line
    )
    timing_breakdown_data_lines = (
        [main_line] + timing_breakdown_lines[6:other_line_idx + 1]
    )
    timing_breakdown_data_dict = {
        br_line.split()[0]: br_line.split()
        for br_line in timing_breakdown_data_lines
    }
    for timing_id in timing_ids:
        if timing_id not in timing_breakdown_data_dict:
            assert timing_id not in ["main", "other"]
            timing_ser[("pyros_ncalls", timing_id)] = 0
            timing_ser[("pyros_cumtime", timing_id)] = 0
            timing_ser[("pyros_percent", timing_id)] = 0
            continue
        br_line_data = timing_breakdown_data_dict.get(timing_id, None)
        time_sub_id = br_line_data[0]
        if time_sub_id != "other":
            timing_ser[("pyros_ncalls", time_sub_id)] = float(br_line_data[1])
        timing_ser[("pyros_cumtime", time_sub_id)] = float(br_line_data[2])
        if time_sub_id != "other":
            timing_ser[("pyros_percall", time_sub_id)] = float(br_line_data[3])
        timing_ser[("pyros_percent", time_sub_id)] = float(br_line_data[4])

    return timing_ser


def get_all_solve_results(indir, pyros_dr_order):
    """
    Get all deterministic and PyROS solve results on file.
    """
    deterministic_results_infile = os.path.join(
        indir,
        "deterministic",
        "deterministic_results.csv",
    )
    pyros_results_indir = os.path.join(indir, "pyros")

    acceptable_pyros_terminations = {
        pyros.pyrosTerminationCondition.robust_feasible,
        pyros.pyrosTerminationCondition.robust_optimal,
    }
    acceptable_det_terminations = {
        pyo.TerminationCondition.globallyOptimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.optimal,
    }

    resdict = dict()
    for name in os.listdir(pyros_results_indir):
        if not os.path.isdir(os.path.join(pyros_results_indir, name)):
            continue
        destarget_str, conf_lvl_str, _ = name.split("_")
        design_target = float(
            destarget_str.split("capture")[0].replace("pt", ".")
        )
        conf_lvl = float(conf_lvl_str.replace("pt", "."))
        series_infile = os.path.join(
            pyros_results_indir, name, "solve_results.csv"
        )
        log_fname = (
            f"pyros_{get_co2_capture_str(None, design_target)}_"
            f"{conf_lvl_str}-conf_dr{pyros_dr_order}_log.log"
        )
        if os.path.exists(series_infile):
            series = pd.read_csv(series_infile, index_col=[0, 1]).iloc[:, 0]
            pyros_term_stats = series["pyros_solver_termination"].copy()

            pyros_term_cond = pyros.pyrosTerminationCondition[
                pyros_term_stats.pyros_termination_condition.split(".")[-1]
            ]
            if pyros_term_cond not in acceptable_pyros_terminations:
                series[:] = None
                series["pyros_solver_termination"] = pyros_term_stats
            logging_infile = os.path.join(indir, "logs", log_fname)
            pyros_timing_stats = get_pyros_timing_breakdown(logging_infile)
            series = pd.concat([series, pyros_timing_stats])
            resdict[(design_target, conf_lvl)] = series.to_dict()

    det_res_df = pd.read_csv(
        deterministic_results_infile,
        header=[0, 1],
        index_col=0,
    )
    for target, row in det_res_df.iterrows():
        det_solver_term_stats = row["solver_termination"].copy()
        det_term_cond = pyo.TerminationCondition(
            det_solver_term_stats.termination_condition
        )
        if det_term_cond not in acceptable_det_terminations:
            row[:] = None
            row["solver_termination"] = det_solver_term_stats
        resdict[(target, 0)] = row

    res_df = pd.DataFrame(resdict).transpose()
    res_df.index.names = ["design_target", "confidence_level"]
    res_df.sort_values(
        ["design_target", "confidence_level"],
        axis=0,
        inplace=True,
    )

    # ensure columns have not been resorted
    col_name_to_idx_map = {
        col: idx for idx, col in enumerate(det_res_df.columns)
    }
    res_df.sort_values(
        [],
        axis=1,
        inplace=True,
        key=lambda col: col_name_to_idx_map[col],
    )

    return res_df


default_capture_target_sens_plot_data_list = [
    QuantityPlotData(
        qty_name="Proxy Cost",
        qty_unit=None,
        multiplier=1,
        col_to_plot=("flowsheet_obj", "obj"),
        fname="proxy_obj",
    ),
    QuantityPlotData(
        qty_name="Proxy CAPEX Cost",
        qty_unit=None,
        multiplier=1,
        col_to_plot=("flowsheet_obj", "capex_cost"),
        fname="proxy_capex",
    ),
    QuantityPlotData(
        qty_name="Absorber Packed Height",
        qty_unit="m",
        multiplier=1,
        col_to_plot=("absorber_column", "length_column"),
        fname="absorber_length",
    ),
    QuantityPlotData(
        qty_name="Absorber Diameter",
        qty_unit="m",
        multiplier=1,
        col_to_plot=("absorber_column", "diameter_column"),
        fname="absorber_diameter",
        inset_targ_lims=(89.5, 96),
    ),
    QuantityPlotData(
        qty_name="Absorber Volume (with heads)",
        qty_unit="1000 $\\text{m}^3$",
        multiplier=1,
        col_to_plot=("absorber_column", "volume_column_withheads"),
        fname="absorber_volume_withheads",
        inset_targ_lims=(89.5, 96),
    ),
    QuantityPlotData(
        qty_name="Absorber Volume (without heads)",
        qty_unit="1000 $\\text{m}^3$",
        multiplier=1,
        col_to_plot=("absorber_column", "volume_column"),
        fname="absorber_volume",
        inset_targ_lims=(89.5, 96),
    ),
    QuantityPlotData(
        qty_name="Stripper Packed Height",
        qty_unit="m",
        multiplier=1,
        col_to_plot=("stripper_column", "length_column"),
        fname="stripper_length",
        inset_targ_lims=(89.5, 96),
    ),
    QuantityPlotData(
        qty_name="Stripper Diameter",
        qty_unit="m",
        multiplier=1,
        col_to_plot=("stripper_column", "diameter_column"),
        fname="stripper_diameter",
    ),
    QuantityPlotData(
        qty_name="Stripper Volume (with heads)",
        qty_unit="1000 $\\text{m}^3$",
        multiplier=1,
        col_to_plot=("stripper_column", "volume_column_withheads"),
        fname="stripper_volume_withheads",
        inset_targ_lims=(89.5, 96),
    ),
    QuantityPlotData(
        qty_name="Stripper Volume (without heads)",
        qty_unit="1000 $\\text{m}^3$",
        multiplier=1,
        col_to_plot=("stripper_column", "volume_column"),
        fname="stripper_volume",
        inset_targ_lims=(89.5, 96),
    ),
    QuantityPlotData(
        qty_name="Cross Heat Exchanger Duty",
        qty_unit="MW",
        multiplier=1,
        col_to_plot=("lean_rich_heat_exchanger", "heat_duty"),
        fname="lrhx_duty",
    ),
    QuantityPlotData(
        qty_name="Reboiler Heat Duty",
        qty_unit="MW",
        multiplier=1,
        col_to_plot=("stripper_reboiler", "heat_duty"),
        fname="reboiler_duty",
    ),
    QuantityPlotData(
        qty_name="Condenser Cooling Duty",
        qty_unit="MW",
        multiplier=-1,
        col_to_plot=("stripper_condenser", "heat_duty"),
        fname="condenser_duty",
    ),
    QuantityPlotData(
        qty_name="$\\text{CO}_2$ Capture Rate",
        qty_unit="\\%",
        multiplier=1,
        col_to_plot=("absorber_column", "co2_capture_rate"),
        fname="co2_capture_rate",
    ),
    QuantityPlotData(
        qty_name="Proxy OPEX Cost",
        qty_unit=None,
        multiplier=1,
        col_to_plot=("flowsheet_obj", "opex_cost"),
        fname="proxy_opex",
    ),
    QuantityPlotData(
        qty_name="Condenser Area",
        qty_unit="1000 $\\text{m}^2$",
        multiplier=1e-3,
        col_to_plot=("stripper_condenser", "area"),
        fname="condenser_area",
    ),
    QuantityPlotData(
        qty_name="Reboiler Area",
        qty_unit="1000 $\\text{m}^2$",
        multiplier=1e-3,
        col_to_plot=("stripper_reboiler", "area"),
        fname="reboiler_area",
    ),
    QuantityPlotData(
        qty_name="Cross Heat Exchanger Area",
        qty_unit="1000 $\\text{m}^2$",
        multiplier=1e-3,
        col_to_plot=("lean_rich_heat_exchanger", "area"),
        fname="lrhx_area",
    ),
    QuantityPlotData(
        qty_name="Condenser Utility Outlet Temperature",
        qty_unit="°C",
        multiplier=1,
        col_to_plot=("stripper_condenser", "utility_outlet_temp"),
        fname="condenser_utility_outlet_temp",
        pre_mult_offset=-273.15,
    ),
    QuantityPlotData(
        qty_name="Condenser Process Outlet Temperature",
        qty_unit="°C",
        multiplier=1,
        col_to_plot=("stripper_condenser", "process_side_vapor_outlet_temp"),
        fname="condenser_process_outlet_temp",
        pre_mult_offset=-273.15,
    ),
    QuantityPlotData(
        qty_name=(
            "Stripper Vapor Distillate $\\text{H}_2\\text{O}$ "
            "Mole Fraction"
        ),
        qty_unit="\\%",
        multiplier=1e2,
        col_to_plot=(
            "stripper_condenser",
            "process_side_vapor_outlet_h2o_mole_frac",
        ),
        fname="condenser_vapor_outlet_h2o_mole_frac",
    ),
    QuantityPlotData(
        qty_name="Condenser Cooling Water Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("stripper_condenser", "cw_flow_mol"),
        fname="condenser_cw_flow_mol",
    ),
    QuantityPlotData(
        qty_name="Reboiler Steam Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("stripper_reboiler", "steam_flow_mol"),
        fname="reboiler_steam_flow_mol",
    ),
    QuantityPlotData(
        qty_name="$\\text{H_2O}$ Makeup Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("other_streams", "h2o_makeup"),
        fname="h2o_makeup",
    ),
    QuantityPlotData(
        qty_name="MEA Recirculation Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("other_streams", "mea_recirculation_rate"),
        fname="mea_recirculation_rate",
    ),
    QuantityPlotData(
        qty_name="Lean Solvent Loading",
        qty_unit="$\\text{mol}\\,\\text{CO}_2/\\text{mol}\\,\\text{MEA}$",
        multiplier=1,
        col_to_plot=("solvent_loading", "lean_loading"),
        fname="solvent_lean_loading",
    ),
    QuantityPlotData(
        qty_name="Rich Solvent Loading",
        qty_unit="$\\text{mol}\\,\\text{CO}_2/\\text{mol}\\,\\text{MEA}$",
        multiplier=1,
        col_to_plot=("solvent_loading", "rich_loading"),
        fname="solvent_rich_loading",
    ),
    QuantityPlotData(
        qty_name="Stripper Top Flood Fraction",
        qty_unit=None,
        multiplier=1,
        col_to_plot=("stripper_column", "top_flood_fraction"),
        fname="stripper_top_flood_fraction",
    ),
    QuantityPlotData(
        qty_name="Absorber L/G Ratio",
        qty_unit=None,
        multiplier=1,
        col_to_plot=("absorber_column", "lg_ratio"),
        fname="absorber_lg_ratio",
    ),
    QuantityPlotData(
        qty_name="Stripper L/G Ratio",
        qty_unit=None,
        multiplier=1,
        col_to_plot=("stripper_column", "lg_ratio"),
        fname="stripper_lg_ratio",
    ),
    QuantityPlotData(
        qty_name="Lean Solvent Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("absorber_column", "liquid_inlet_flow_rate"),
        fname="absorber_liquid_inlet_flow_mol",
    ),
    QuantityPlotData(
        qty_name="Rich Solvent Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("stripper_column", "liquid_inlet_flow_rate"),
        fname="stripper_liquid_inlet_flow_mol",
    ),
    QuantityPlotData(
        qty_name="Lean Solvent Temperature",
        qty_unit="°C",
        multiplier=1,
        col_to_plot=("solvent_temperature", "lean_temperature"),
        fname="lean_solvent_temperature",
        pre_mult_offset=-273.15,
    ),
    QuantityPlotData(
        qty_name="Rich Solvent Temperature",
        qty_unit="°C",
        multiplier=1,
        col_to_plot=("solvent_temperature", "rich_temperature"),
        fname="rich_solvent_temperature",
        pre_mult_offset=-273.15,
    ),
]


default_conf_lvl_sens_plot_data_list = [
    QuantityPlotData(
        qty_name="MEA Component Recirculation Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("other_streams", "mea_recirculation_rate"),
        fname="mea_recirculation_rate",
    ),
    QuantityPlotData(
        qty_name="Reboiler Steam Utility Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("stripper_reboiler", "steam_flow_mol"),
        fname="reboiler_steam_flow_mol",
    ),
    QuantityPlotData(
        qty_name="Condenser Cooling Water Utility Flow Rate",
        qty_unit="kmol/s",
        multiplier=1,
        col_to_plot=("stripper_condenser", "cw_flow_mol"),
        fname="condenser_cw_flow_mol",
    ),
]


default_grouped_conf_lvl_sens_plot_data_list = [
    GroupedPlotDataContainer(
        fname="column_volumes",
        qty_name="Column Volume (with heads)",
        qty_unit="1000 $\\text{m}^3$",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=f"{col_name} column",
                multiplier=1,
                col_to_group=(
                    f"{col_name}_column", "volume_column_withheads"
                ),
            )
            for col_name in ["absorber", "stripper"]
        ],
    ),
    GroupedPlotDataContainer(
        fname="column_dimensions",
        qty_name="Column Dimension",
        qty_unit="m",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=f"{col_name} {dim_desc}",
                multiplier=1,
                col_to_group=(f"{col_name}_column", f"{dim_name}_column"),
            )
            for col_name in ["absorber", "stripper"]
            for dim_name, dim_desc in (
                ["length", "height"],
                ["diameter", "diameter"],
            )
        ],
    ),
    GroupedPlotDataContainer(
        fname="hex_areas",
        qty_name="Heat Exchanger Area",
        qty_unit="$\\text{m}^2$",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=hex_name,
                multiplier=1,
                col_to_group=(hex_col_name, "area"),
            )
            for hex_name, hex_col_name in {
                "reboiler": "stripper_reboiler",
                "condenser": "stripper_condenser",
                "cross heat exchanger": "lean_rich_heat_exchanger",
            }.items()
        ],
    ),
    GroupedPlotDataContainer(
        fname="hex_duties",
        qty_name="Heat Exchanger Duty",
        qty_unit="MW",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=hex_name,
                multiplier=-1 if hex_name == "condenser" else 1,
                col_to_group=(hex_col_name, "heat_duty"),
            )
            for hex_name, hex_col_name in {
                "reboiler": "stripper_reboiler",
                "condenser": "stripper_condenser",
                "cross heat exchanger": "lean_rich_heat_exchanger",
            }.items()
        ],
    ),
    GroupedPlotDataContainer(
        fname="solvent_loading",
        qty_name="Solvent $\\text{CO}_2$ Loading",
        qty_unit="$\\text{mol\\,CO}_2/\\text{mol\\,MEA}$",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=f"{solv} solvent",
                multiplier=1,
                col_to_group=("solvent_loading", f"{solv}_loading"),
            )
            for solv in ["rich", "lean"]
        ],
    ),
    GroupedPlotDataContainer(
        fname="solvent_temperature",
        qty_name="Solvent Temperature",
        qty_unit="°C",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=f"{solv} solvent",
                multiplier=1,
                col_to_group=("solvent_temperature", f"{solv}_temperature"),
                pre_mult_offset=-273.15,
                post_mult_offset=0,
            )
            for solv in ["rich", "lean"]
        ],
    ),
    GroupedPlotDataContainer(
        fname="column_liquid_inlet_flows",
        qty_name="Column Liquid Inlet Flow Rate",
        qty_unit="kmol/s",
        legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        group_data_list=[
            GroupedPlotData(
                group_name=f"{col_name} column",
                multiplier=1,
                col_to_group=(f"{col_name}_column", "liquid_inlet_flow_rate"),
            )
            for col_name in ["absorber", "stripper"]
        ],
    ),
]


default_stacked_conf_lvl_sens_plot_data_list = [
    StackedPlotData(
        fname="hex_areas",
        cols_to_stack=[
            ("stripper_reboiler", "area"),
            ("stripper_condenser", "area"),
            ("lean_rich_heat_exchanger", "area"),
        ],
        other_cols_to_stack=[],
        qty_name="Heat Transfer Area",
        qty_units="$\\text{m}^2$",
        label_names=["reboiler", "condenser", "cross HEX"],
        col_multipliers=[1, 1, 1],
    ),
]


def process_solve_results(indir, pyros_dr_order, **kwargs):
    """
    Process all PyROS solve results, as well as results for
    corresponding deterministic counterparts.
    """
    return process_solve_results_df(
        res_df=get_all_solve_results(indir, pyros_dr_order),
        outdir=os.path.join(indir, "solve_results_analysis"),
        **kwargs,
    )


def process_solve_results_df(
        res_df,
        outdir,
        capture_target_sens_plot_data_list=None,
        conf_lvl_sens_plot_data_list=None,
        grouped_conf_lvl_sens_plot_data_list=None,
        stacked_conf_lvl_sens_plot_data_list=None,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Process all PyROS solve results, as well as results for
    corresponding deterministic counterparts.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    res_df.to_csv(os.path.join(outdir, "pyros_results_full.csv"))

    plot_solver_performance_heatmaps(
        res_df,
        os.path.join(outdir, "performance_stats"),
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )

    if capture_target_sens_plot_data_list is None:
        capture_target_sens_plot_data_list = (
            default_capture_target_sens_plot_data_list
        )
    if conf_lvl_sens_plot_data_list is None:
        conf_lvl_sens_plot_data_list = default_conf_lvl_sens_plot_data_list
    if grouped_conf_lvl_sens_plot_data_list is None:
        grouped_conf_lvl_sens_plot_data_list = (
            default_grouped_conf_lvl_sens_plot_data_list
        )
    if stacked_conf_lvl_sens_plot_data_list is None:
        stacked_conf_lvl_sens_plot_data_list = (
            default_stacked_conf_lvl_sens_plot_data_list
        )

    capture_target_sens_plot_outdir = os.path.join(
        outdir,
        "capture_target_sensitivity",
    )
    conf_lvl_sens_plot_outdir = os.path.join(
        outdir,
        "confidence_level_sensitivity",
    )
    grouped_conf_lvl_sens_plot_outdir = os.path.join(
        outdir,
        "grouped_confidence_level_sensitivity",
    )
    stacked_conf_lvl_sens_plot_outdir = os.path.join(
        outdir,
        "stacked_confidence_level_sensitivity",
    )
    subdirs = [
        capture_target_sens_plot_outdir,
        conf_lvl_sens_plot_outdir,
        grouped_conf_lvl_sens_plot_outdir,
        stacked_conf_lvl_sens_plot_outdir,
    ]
    for sdir in subdirs:
        if not os.path.exists(sdir):
            os.mkdir(sdir)

    plot_solve_results_capture_target_sens(
        results_df=res_df,
        outdir=capture_target_sens_plot_outdir,
        quantity_plot_data_list=capture_target_sens_plot_data_list,
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )
    plot_solve_results_conf_lvl_sens(
        results_df=res_df,
        outdir=conf_lvl_sens_plot_outdir,
        quantity_plot_data_list=conf_lvl_sens_plot_data_list,
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )
    plot_solve_results_grouped_conf_lvl_sens(
        results_df=res_df,
        outdir=grouped_conf_lvl_sens_plot_outdir,
        grouped_plot_data_list=grouped_conf_lvl_sens_plot_data_list,
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )
    plot_solve_results_stacked_conf_lvl_sens(
        results_df=res_df,
        outdir=stacked_conf_lvl_sens_plot_outdir,
        stacked_plot_data_list=stacked_conf_lvl_sens_plot_data_list,
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )

    # summarize DOF variable values
    res_dof_col_name_pairs = MEAFlowsheetResults.get_dof_var_column_tuples()
    dof_vals_df = pd.DataFrame(
        index=res_df.index,
        columns=pd.MultiIndex.from_tuples(
            [dof_col for _, dof_col in res_dof_col_name_pairs]
        ),
    )
    for res_col, dof_col in res_dof_col_name_pairs:
        dof_vals_df[dof_col] = res_df[res_col]
    dof_vals_df.to_csv(os.path.join(outdir, "pyros_results_dof_vals.csv"))

    default_logger.info("All done processing solve results.")


HeatmapProcessData = namedtuple(
    typename="HeatmapProcessData",
    field_names=[
        "design_target",
        "design_conf_lvl",
        "eval_target",
        "eval_conf_lvl",
        "heatmap_df",
        "heatmap_summary_series",
    ],
)


def get_heatmap_proc_data_sort_key(proc_data):
    return (
        proc_data.design_target,
        proc_data.design_conf_lvl,
        proc_data.eval_target,
        proc_data.eval_conf_lvl,
    )


def get_deterministic_heatmap_results(
        base_heatmap_dir,
        cov_mat_infile,
        ):
    """
    Get deterministic heatmap results.
    """
    from confidence_ellipsoid import calc_conf_lvl

    heatmap_proc_data_list = []
    for design_subdir_name in os.listdir(base_heatmap_dir):
        isdir = os.path.isdir(
            os.path.join(base_heatmap_dir, design_subdir_name)
        )
        if not isdir:
            continue
        design_subdir_path = os.path.join(
            base_heatmap_dir,
            design_subdir_name,
        )
        design_target = float(
            design_subdir_name.split("_")[0].replace("pt", ".")
        )
        design_conf_lvl = 0

        for eval_subdir_name in os.listdir(design_subdir_path):
            eval_target = float(
                eval_subdir_name.split("_")[0].replace("pt", ".")
            )
            heatmap_path = os.path.join(
                design_subdir_path,
                eval_subdir_name,
                "dataframes",
                "results_deterministic.csv",
            )
            summary_path = os.path.join(
                design_subdir_path,
                eval_subdir_name,
                "results",
                "results_deterministic.csv",
            )
            heatmap_summary_series = pd.read_csv(
                summary_path, index_col=0
            )["0"]
            heatmap_df = pd.read_csv(heatmap_path, index_col=0)
            heatmap_df["Parameter Values"] = [
                np.array([
                    float(val.strip(" \n"))
                    for val in param_val_str.strip("[]").split(" ") if val
                ])
                for param_val_str in heatmap_df["Parameter Values"]
            ]

            param_values_arr = np.stack(
                heatmap_df["Parameter Values"].to_numpy()
            )
            mean = param_values_arr[0]
            param_cov_mat = pd.read_csv(cov_mat_infile, index_col=0)
            mahalanobis_distances = np.sqrt(
                (
                    (param_values_arr - mean)
                    @ np.linalg.inv(param_cov_mat)
                    * (param_values_arr - mean)
                ).sum(axis=-1)
            )
            conf_lvls = np.array([
                100 * calc_conf_lvl(dist, row.size)
                for dist, row in zip(mahalanobis_distances, param_values_arr)
            ])
            eval_conf_lvl = round(conf_lvls.max(), 2)

            heatmap_proc_data_list.append(HeatmapProcessData(
                design_target=design_target,
                design_conf_lvl=design_conf_lvl,
                eval_target=eval_target,
                eval_conf_lvl=eval_conf_lvl,
                heatmap_df=heatmap_df,
                heatmap_summary_series=heatmap_summary_series,
            ))

    heatmap_proc_data_list.sort(key=get_heatmap_proc_data_sort_key)

    return heatmap_proc_data_list


default_heatmap_plot_qty_info_list = [
    HeatmapQuantityInfo(
        qty_expr_col="fs.obj",
        qty_mult=1,
        qty_yax_str="Proxy Cost",
        qty_yax_unit=None,
        fname="proxy_obj",
        solve_results_col=("flowsheet_obj", "obj"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.capex_cost",
        qty_mult=1,
        qty_yax_str="Proxy CAPEX Cost",
        qty_yax_unit=None,
        fname="proxy_capex",
        solve_results_col=("flowsheet_obj", "capex_cost"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.opex_cost",
        qty_mult=1,
        qty_yax_str="Proxy OPEX Cost",
        qty_yax_unit=None,
        fname="proxy_opex",
        solve_results_col=("flowsheet_obj", "opex_cost"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.absorber_section.absorber.co2_capture[0.0]",
        qty_mult=1,
        qty_yax_str="$\\text{CO}_2$ Capture Rate",
        qty_yax_unit="\\%",
        fname="co2_capture_rate",
        solve_results_col=("absorber_column", "co2_capture_rate"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.stripper_section.reboiler.liquid_phase.heat[0.0]"
        ),
        qty_mult=1e-6,
        qty_yax_str="Reboiler Heating Duty",
        qty_yax_unit="MW",
        fname="reboiler_duty",
        solve_results_col=("stripper_reboiler", "heat_duty"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.stripper_section.condenser.vapor_phase.heat[0.0]"
        ),
        qty_mult=-1e-6,
        qty_yax_str="Condenser Cooling Duty",
        qty_yax_unit="MW",
        fname="condenser_duty",
        solve_results_col=("stripper_condenser", "heat_duty"),
        solve_results_mult=-1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.absorber_section.lean_rich_heat_exchanger"
            ".cold_side.heat[0.0]"
        ),
        qty_mult=1e-6,
        qty_yax_str="Cross Heat Exchanger Duty",
        qty_yax_unit="MW",
        fname="lean_rich_hex_duty",
        solve_results_col=("lean_rich_heat_exchanger", "heat_duty"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.stripper_section.condenser.vapor_phase"
            ".properties_out[0.0].mole_frac_comp[H2O]"
        ),
        qty_mult=1e2,
        qty_yax_str=(
            "Stripper Distillate $\\text{H}_2\\text{O}$ Mole Fraction"
        ),
        qty_yax_unit="\\%",
        fname="distillate_h2o_mole_frac",
        solve_results_col=(
            "stripper_condenser",
            "process_side_vapor_outlet_h2o_mole_frac"
        ),
        solve_results_mult=1e2,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.stripper_section.stripper.flood_fraction[0.0,1.0]"
        ),
        qty_mult=1,
        qty_yax_str="Stripper Top Flood Fraction",
        qty_yax_unit=None,
        fname="stripper_top_flood_fraction",
        solve_results_col=("stripper_column", "top_flood_fraction"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.mea_recirculation_rate[0.0]",
        qty_mult=1e-3,
        qty_yax_str="MEA Recirculation Rate",
        qty_yax_unit="kmol/s",
        fname="mea_recirculation_rate",
        solve_results_col=(
            "other_streams", "mea_recirculation_rate"
        ),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.stripper_section.reboiler.steam_flow_mol[0.0]"
        ),
        qty_mult=1e-3,
        qty_yax_str="Reboiler Steam Flow",
        qty_yax_unit="kmol/s",
        fname="reboiler_steam_flow_mol",
        solve_results_col=("stripper_reboiler", "steam_flow_mol"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col=(
            "fs.stripper_section.condenser.cooling_flow_mol[0.0]"
        ),
        qty_mult=1e-3,
        qty_yax_str="Condenser Cooling Water Flow",
        qty_yax_unit="kmol/s",
        fname="condenser_cw_flow_mol",
        solve_results_col=("stripper_condenser", "cw_flow_mol"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.rich_loading[0.0]",
        qty_mult=1,
        qty_yax_str="Rich Solvent Loading",
        qty_yax_unit="$\\text{mol\\,CO}_2/\\text{mol\\,MEA}$",
        fname="rich_loading",
        solve_results_col=("solvent_loading", "rich_loading"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.lean_loading[0.0]",
        qty_mult=1,
        qty_yax_str="Lean Solvent Loading",
        qty_yax_unit="$\\text{mol\\,CO}_2/\\text{mol\\,MEA}$",
        fname="lean_loading",
        solve_results_col=("solvent_loading", "lean_loading"),
        solve_results_mult=1,
    ),
    HeatmapQuantityInfo(
        qty_expr_col="fs.makeup_scaler.properties[0.0].flow_mol",
        qty_mult=1e-3,
        qty_yax_str="$\\text{H}_2\\text{O}$ Makeup",
        qty_yax_unit="kmol/s",
        fname="h2o_makeup",
        solve_results_col=("other_streams", "h2o_makeup"),
        solve_results_mult=1,
    ),
]


def process_all_heatmap_results(
        results_dir,
        outdir,
        cov_mat_infile,
        pyros_dr_order=0,
        include_deterministic_results=True,
        include_pyros_results=True,
        include_nominal_point_in_analysis=False,
        export_summary=True,
        heatmap_plot_qty_info_list=None,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Process heatmap results for nominally optimal (deterministic)
    and heuristically robust feasible (PyROS) solutions.
    """
    cov_mat = pd.read_csv(cov_mat_infile, index_col=0).to_numpy()

    heatmap_proc_data_list = []
    if include_deterministic_results:
        heatmap_proc_data_list.extend(get_deterministic_heatmap_results(
            os.path.join(results_dir, "deterministic_heatmaps"),
            cov_mat_infile=cov_mat_infile,
        ))
    if include_pyros_results:
        heatmap_proc_data_list.extend(get_pyros_heatmap_results(
            os.path.join(results_dir, "pyros_heatmaps"),
            decision_rule_order=pyros_dr_order,
            cov_mat_infile=cov_mat_infile,
        ))
    assert heatmap_proc_data_list, "No heatmap results found."

    heatmap_proc_data_list.sort(key=get_heatmap_proc_data_sort_key)

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=False)

    subdir_names = [
        "feasibility_analysis_plots",
        "feasibility_target_sensitivity",
        "capture_target_sensitivity",
        "confidence_level_sensitivity",
        "scatter_plots",
        "quantity_distributions",
    ]
    subdir_dict = {
        name: os.path.join(outdir, name)
        for name in subdir_names
    }
    for subdir in subdir_dict.values():
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    finaldf = process_heatmap_results(
        heatmap_proc_data_list=heatmap_proc_data_list,
        param_cov_mat=cov_mat,
        feasible_termination_conditions=None,
        include_nominal_point_in_analysis=include_nominal_point_in_analysis,
    )
    if export_summary:
        finaldf.to_csv(os.path.join(outdir, "heatmap_summary.csv"))

    plot_heatmap_feasibility_target_response(
        heatmap_proc_data_list=heatmap_proc_data_list,
        heatmap_summary_df=finaldf,
        outdir=subdir_dict["feasibility_target_sensitivity"],
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )
    plot_heatmap_feasibility_results(
        heatmap_proc_data_list=heatmap_proc_data_list,
        param_cov_mat=cov_mat,
        outdir=subdir_dict["feasibility_analysis_plots"],
        feasible_termination_conditions=None,
        ignore_termination_conditions=None,
        include_nominal_point_in_analysis=False,
        output_plot_fmt=output_plot_fmt,
    )

    # quantities to plot
    if heatmap_plot_qty_info_list is None:
        heatmap_plot_qty_info_list = default_heatmap_plot_qty_info_list

    solver_results_df = get_all_solve_results(
        results_dir,
        pyros_dr_order=pyros_dr_order,
    )
    plot_quantity_distributions(
        heatmap_proc_data_list=heatmap_proc_data_list,
        heatmap_df_col_info_list=heatmap_plot_qty_info_list,
        outdir=subdir_dict["quantity_distributions"],
        heatmap_summary_df=finaldf,
        solver_results_df=solver_results_df,
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )
    plot_capture_target_sensitivity(
        heatmap_proc_data_list=heatmap_proc_data_list,
        outdir=subdir_dict["capture_target_sensitivity"],
        param_cov_mat=cov_mat,
        heatmap_summary_df=finaldf,
        heatmap_df_col_info_list=heatmap_plot_qty_info_list,
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )
    plot_confidence_level_sensitivity(
        heatmap_proc_data_list=heatmap_proc_data_list,
        outdir=subdir_dict["confidence_level_sensitivity"],
        param_cov_mat=cov_mat,
        heatmap_summary_df=finaldf,
        heatmap_df_col_info_list=heatmap_plot_qty_info_list,
    )
    plot_heatmap_scatter_results(
        heatmap_proc_data_list=heatmap_proc_data_list,
        outdir=subdir_dict["scatter_plots"],
        param_cov_mat=cov_mat,
        savefig_kwds=dict(dpi=300),
        projection_idx_pairs=[(0, 1), (2, 3), (4, 5)],
        heatmap_df_col_info_list=heatmap_plot_qty_info_list,
        output_plot_fmt=output_plot_fmt,
        include_only_designs={98: [0, 90, 95, 99]},
    )

    default_logger.info("All done processing heatmap results.")


def get_pyros_heatmap_results(
        base_heatmap_dir,
        decision_rule_order,
        cov_mat_infile,
        in_sample_realizations_only=False,
        ):
    """
    Get PyROS heatmap results for a given decision rule order.
    """
    from confidence_ellipsoid import calc_conf_lvl

    heatmap_proc_data_list = []
    for design_subdir_name in os.listdir(base_heatmap_dir):
        isdir = os.path.isdir(
            os.path.join(base_heatmap_dir, design_subdir_name)
        )
        if not isdir:
            continue
        design_subdir_path = os.path.join(
            base_heatmap_dir,
            design_subdir_name,
        )
        design_target = float(
            design_subdir_name.split("_")[0].replace("pt", ".")
        )
        design_conf_lvl = float(
            design_subdir_name.split("_")[4].replace("pt", ".")
        )
        if f"dr{decision_rule_order}" not in design_subdir_name:
            continue

        for eval_subdir_name in os.listdir(design_subdir_path):
            eval_target = float(
                eval_subdir_name.split("_")[0].replace("pt", ".")
            )
            heatmap_path = os.path.join(
                design_subdir_path,
                eval_subdir_name,
                "dataframes",
                "results_pyros.csv",
            )
            summary_path = os.path.join(
                design_subdir_path,
                eval_subdir_name,
                "results",
                "results_pyros.csv",
            )
            heatmap_summary_series = pd.read_csv(
                summary_path, index_col=0
            )["0"]
            heatmap_df = pd.read_csv(heatmap_path, index_col=0)
            heatmap_df["Parameter Values"] = [
                np.array([
                    float(val.strip(" \n"))
                    for val in param_val_str.strip("[]").split(" ") if val
                ])
                for param_val_str in heatmap_df["Parameter Values"]
            ]

            if in_sample_realizations_only:
                eval_conf_lvl = design_conf_lvl
            else:
                param_values_arr = np.stack(
                    heatmap_df["Parameter Values"].to_numpy()
                )
                mean = param_values_arr[0]
                param_cov_mat = pd.read_csv(cov_mat_infile, index_col=0)
                mahalanobis_distances = np.sqrt(
                    (
                        (param_values_arr - mean)
                        @ np.linalg.inv(param_cov_mat)
                        * (param_values_arr - mean)
                    ).sum(axis=-1)
                )
                conf_lvls = np.array([
                    100 * calc_conf_lvl(dist, row.size)
                    for dist, row in zip(
                        mahalanobis_distances, param_values_arr
                    )
                ])
                eval_conf_lvl = round(conf_lvls.max(), 2)

            heatmap_proc_data_list.append(HeatmapProcessData(
                design_target=design_target,
                design_conf_lvl=design_conf_lvl,
                eval_target=eval_target,
                eval_conf_lvl=eval_conf_lvl,
                heatmap_df=heatmap_df,
                heatmap_summary_series=heatmap_summary_series,
            ))

    heatmap_proc_data_list.sort(key=get_heatmap_proc_data_sort_key)

    return heatmap_proc_data_list
