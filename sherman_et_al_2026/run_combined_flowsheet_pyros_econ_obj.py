#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script: Workflows for economic optimization of the MEA scrubbing
process model (flowsheet model).
"""


import argparse
from contextlib import nullcontext
import logging
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pyomo.common.collections import Bunch, ComponentMap, ComponentSet
from pyomo.common.errors import ApplicationError
from pyomo.common.log import LogStream
from pyomo.common.modeling import unique_component_name
from pyomo.common.tee import capture_output
import pyomo.contrib.pyros as pyros
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.expr.visitor import replace_expressions, identify_variables
import pyomo.environ as pyo
from pyomo.opt import SolverResults
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.check_units import (
    assert_units_equivalent,
    check_units_equivalent,
)

from flowsheets.combined_flowsheet_withcosting_CCS import (
    MEACombinedFlowsheet,
)
from flowsheets.combined_flowsheet_withcosting_NGCC_CCS import (
    MEACombinedFlowsheetData as MEACombinedFlowsheetDataNGCCPlusCCS
)
from model_utils import (
    ColoredFormatter,
    float_to_str,
    get_co2_capture_str,
    get_solver,
    log_script_invocation,
    SolverWithBackup,
    SolverWithScaling,
)
from plotting_utils import (
    DEFAULT_MPL_RC_PARAMS,
    set_nonoverlapping_fixed_xticks,
    wrap_quantity_str,
)
from combined_flowsheet_pyros_base import (
    FlowsheetModelData,
    generate_deterministic_results_spreadsheet,
    heatmap_workflow,
    pyros_workflow,
    MEAFlowsheetResults,
)

from monkey_patch_costing import monkey_patch_costing

default_logger = logging.getLogger(__name__)
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
monkey_patch_costing()


def add_ccs_capex_cost_epigraph_vars(model):
    """
    Add 'epigraph'-like variables that serve as design proxies for
    the CCS CAPEX model.
    """
    fs = model.fs

    # these variables are to be considered first-stage
    # degrees of freedom
    fs.design_epigraph_vars = pyo.Var(
        [
            "rich_solvent_flow_vol",
            "lean_solvent_flow_vol",
            "reboiler_heat_duty",
            "CO2_capture_flow_rate",
            "solvent_fill_init",
        ]
    )
    fs.design_epigraph_cons = pyo.Constraint(
        fs.design_epigraph_vars.index_set()
    )

    # each of the design variables will be substitued
    # for the expression it is meant to be the design
    # proxy for in the CAPEX bare erected cost
    # constraint expressions
    costing_blk_replacement_map = [
        [
            "rich_solvent_flow_vol",
            fs.absorber_section.rich_solvent_flow_vol[0],
            [fs.b30],
        ],
        [
            "lean_solvent_flow_vol",
            fs.stripper_section.lean_solvent_flow_vol[0],
            [fs.b37, fs.b40, fs.b41],
        ],
        [
            "reboiler_heat_duty",
            fs.stripper_section.reboiler.liquid_phase.heat[0],
            [fs.b49],
        ],
        [
            "CO2_capture_flow_rate",
            fs.costing_setup.CO2_capture_rate[0],
            [fs.b36, fs.b42],
        ],
        [
            "solvent_fill_init",
            fs.costing_setup.solvent_fill_init[0],
            [],
        ],
    ]

    scaling_suffix_finder = SuffixFinder("scaling_factor")
    for epi_index, var_expr, costing_blk_list in costing_blk_replacement_map:
        epi_var = fs.design_epigraph_vars[epi_index]
        epi_var.set_value(pyo.value(var_expr))

        # "epigraph" variable itself has no units,
        # but we ensure that the units of the constraint
        # expressions are consistent
        fs.design_epigraph_cons[epi_index] = (
            var_expr - epi_var * pyo.units.get_units(var_expr) <= 0
        )

        var_expr_scaling_factor = scaling_suffix_finder.find(var_expr)
        if var_expr_scaling_factor is not None:
            fs.scaling_factor[epi_var] = var_expr_scaling_factor
            fs.scaling_factor[fs.design_epigraph_cons[epi_index]] = (
                var_expr_scaling_factor
            )

        # subsitute new design variables for operating/state
        # variables/expressions in the costing constraints
        for costing_blk in costing_blk_list:
            for cost_eqn in costing_blk.bare_erected_cost_eq.values():
                cost_eqn.set_value(
                    replace_expressions(
                        expr=cost_eqn.expr,
                        substitution_map={
                            id(var_expr): (
                                epi_var * pyo.units.get_units(var_expr)
                            ),
                        },
                        descend_into_named_expressions=False,
                    ),
                )

        # cover the one CAPEX term not associated with a
        # bare erected cost constraint
        if epi_index == "solvent_fill_init":
            removal_cost_expr = model.fs.costing.RemovalSystem_Equip_Adjust
            removal_cost_expr.expr = replace_expressions(
                expr=removal_cost_expr.expr,
                substitution_map={
                    id(var_expr): epi_var * pyo.units.get_units(var_expr)
                },
                descend_into_named_expressions=False,
            )


def add_ccs_capex_cost_term_expressions(model):
    """
    Add expressions for tracking the *annualized*
    contributions of equipment costs to the CCS to the AIC.
    """
    fs = model.fs

    # these are total plant costs, in MM $ (2018)
    ccs_tpc_name_to_expr_map = {
        (
            "total_solvent_purchased"
            if cexpr is fs.costing.RemovalSystem_Equip_Adjust
            else cexpr.local_name.replace("_cost", "")
        ): cexpr
        for cexpr in [
            fs.costing.absorber_column_cost,
            fs.costing.absorber_packing_cost,
            fs.costing.lean_rich_hex_cost,
            fs.costing.rich_solvent_pump_cost,
            fs.costing.lean_solvent_pump_cost,
            fs.costing.stripper_column_cost,
            fs.costing.stripper_packing_cost,
            fs.costing.stripper_condenser_cost,
            fs.costing.stripper_reboiler_cost,
            fs.costing.reboiler_condensate_pot_cost,
            fs.costing.stripper_reflux_drum_cost,
            fs.costing.stripper_reflux_pump_cost,
            fs.costing.solvent_filtration_cost,
            fs.costing.solvent_storage_tank_cost,
            fs.costing.RemovalSystem_Equip_Adjust,
        ]
    }

    cost_term_to_var_map = ComponentMap()
    for cost_term in ccs_tpc_name_to_expr_map.values():
        assert check_units_equivalent(
            pyo.units.get_units(cost_term),
            pyo.units.MUSD_2018,
        )

        # check assumptions. most importantly,
        # assume each CAPEX cost term has only one unique variable
        vars_in_cost_term = list(identify_variables(cost_term))
        assert len(vars_in_cost_term) == 1
        costvar = vars_in_cost_term[0]
        assert costvar not in ComponentSet(cost_term_to_var_map.values())

        cost_term_to_var_map[cost_term] = costvar

    vars_in_ccs_aic = ComponentSet(identify_variables(fs.costing.CCS_AIC))
    vars_in_costing_map = ComponentSet(cost_term_to_var_map.values())
    assert vars_in_ccs_aic == vars_in_costing_map

    visitor = LinearRepnVisitor(
        subexpression_cache={},
        var_map={},
        var_order={},
        sorter=None,
    )
    repn = visitor.walk_expression(fs.costing.CCS_AIC)
    assert repn.nonlinear is None

    fs.ccs_aic_terms = pyo.Expression(list(ccs_tpc_name_to_expr_map.keys()))
    for key in fs.ccs_aic_terms:
        var_in_term = cost_term_to_var_map[ccs_tpc_name_to_expr_map[key]]
        fs.ccs_aic_terms[key].set_value(
            repn.linear[id(var_in_term)]
            * var_in_term
            / pyo.units.get_units(var_in_term)
            * pyo.units.get_units(fs.costing.CCS_AIC)
        )
        assert_units_equivalent(
            fs.ccs_aic_terms[key],
            pyo.units.USD_2018 / pyo.units.a,
        )


def add_ccs_opex_cost_term_expressions(model):
    """
    Add expressions for tracking the annualized
    contributions to the CCS AOC.
    """
    fs = model.fs
    utility_costs = [
        # in USD/h
        fs.costing.UC_rich_solvent_pump,
        fs.costing.UC_lean_solvent_pump,
        # Other utility costs, in USD/s
        fs.costing.UC_reboiler,
        fs.costing.UC_condenser,
        fs.costing.UC_H2O_makeup,
    ]

    fs.ccs_aoc_terms = pyo.Expression(
        [
            util_cost.local_name.replace("UC_", "")
            for util_cost in utility_costs
        ],
    )
    for key, utilcost in zip(fs.ccs_aoc_terms, utility_costs):
        utilcost_units = pyo.units.get_units(utilcost)
        musd_units = pyo.units.USD_2018
        # instead of using pyomo unit converter,
        # we manually convert seconds/hours to years,
        # since we take 1 year = 8760 h and the converter takes
        # 1 yr = 8766 h
        if check_units_equivalent(utilcost_units, musd_units / pyo.units.a):
            fs.ccs_aoc_terms[key].set_value(utilcost)
        elif check_units_equivalent(utilcost_units, musd_units / pyo.units.h):
            fs.ccs_aoc_terms[key].set_value(
                utilcost * 8760 * pyo.units.h / pyo.units.a
            )
        elif check_units_equivalent(utilcost_units, musd_units / pyo.units.s):
            fs.ccs_aoc_terms[key].set_value(
                utilcost * 8760 * 3600 * pyo.units.s / pyo.units.a
            )
        else:
            raise ValueError(
                f"Utility cost units {utilcost_units} "
                f"for utility cost {utilcost} not supported."
            )
        assert_units_equivalent(
            fs.ccs_aoc_terms[key],
            musd_units / pyo.units.a,
        )


class FlowsheetEconomicOptModelData(FlowsheetModelData):
    """
    Interface to flowsheet model with economic objective.
    """
    def __init__(self, original_model, scaled_model=None, with_costing=True):
        super().__init__(original_model, scaled_model=scaled_model)
        self.with_costing = with_costing

    def clone(self):
        return self.__init__(
            original_model=self.original_model.clone(),
            scaled_model=None,
            with_costing=self.with_costing,
        )

    def create_heatmap_solver_wrapper(self, solver, logger):
        """
        Create wrapper around a solver object for deterministic
        and 'heatmap' workflows.

        This wrapper is not meant to be used directly as a PyROS
        subsolver in PyROS workflows.

        Parameters
        ----------
        solver : Pyomo solver object
            Solver with which to run the heatmap workflows.
            This is usually an instance of `SolverWithBackup`.
        logger : logging.Logger
            Logger with which to equip the wrapper.

        Returns
        -------
        FlowsheetWithCostingSolver
            Solver wrapper.
        """
        return self.get_heatmap_solver_wrapper_type()(
            solver=solver,
            logger=logger,
            flowsheet_model_data_type=type(self),
            with_costing=self.with_costing,
        )

    @staticmethod
    def get_heatmap_solver_wrapper_type():
        return FlowsheetNGCCPlusCCSCostingSolver

    def log_objective_breakdown(self, logger, level=logging.INFO):
        """
        Log objective breakdown.
        """
        def log_msg(msg):
            return logger.log(level=level, msg=msg)

        self.log_proxy_objective_breakdown(logger, level=level)
        log_msg("-" * 80)
        log_msg("CCS TAC Objective Breakdown")

        model = self.original_model

        solvent_fill_cost = pyo.value(
            model.fs.costing.RemovalSystem_Equip_Adjust
        )
        log_msg(
            " Solvent cost (MM $ (2018)) : "
            f"{solvent_fill_cost:.2f}"
        )
        tec_name = "TEC (MM $ (2018))"
        log_msg(
            f" {tec_name:43s} : "
            f"{pyo.value(model.fs.costing.CCS_TEC) / 1e6:>15.2f}"
        )
        tpc_name = "TPC (MM $ (2018))"
        log_msg(
            f" {tpc_name:43s} : "
            f"{pyo.value(model.fs.costing.CCS_TPC) / 1e6:>15.2f}"
        )

        log_msg(" AIC breakdown (MM $ (2018)/yr)")
        for costname, costexpr in model.fs.ccs_aic_terms.items():
            equip_cost_name = costname.replace("_", " ")
            costval = pyo.value(
                costexpr
                * pyo.units.convert(pyo.units.USD_2018, pyo.units.MUSD_2018)
            )
            log_msg(f"    {equip_cost_name:40s} : {costval:>15.2f}")

        log_msg(" AOC breakdown (MM $ (2018) / yr)")
        for costname, costexpr in model.fs.ccs_aoc_terms.items():
            util_cost_name = costname.replace("_", " ")
            costval = pyo.value(
                costexpr
                * pyo.units.convert(pyo.units.USD_2018, pyo.units.MUSD_2018)
            )
            log_msg(f"    {util_cost_name:40s} : {costval:>15.2f}")

        annualized_costs = [
            # annualized economic metrics, in USD/yr
            model.fs.costing.CCS_AIC,
            model.fs.costing.CCS_AOC,
            model.fs.costing.CCS_TAC,
            model.fs.costing.CCS_TAC_perCO2captured,
        ]
        log_msg(" Overall annualized cost figures")
        for anncost in annualized_costs:
            anncost_name = (
                anncost.local_name.split("CCS_")[-1].replace("_", " ")
            )
            if "perCO2captured" in anncost_name:
                anncost_unit = " ($ (2018) / tonne CO2)"
                anncost_val = pyo.value(anncost)
            else:
                anncost_unit = " (MM $ (2018) / yr)"
                anncost_val = pyo.value(
                    anncost * pyo.units.convert(
                        pyo.units.USD_2018,
                        pyo.units.MUSD_2018,
                    )
                )
            log_msg(
                f"    {anncost_name + anncost_unit:40s} : {anncost_val:>15.2f}"
            )

        log_msg(" TAC Objective terms")
        for name, expr in model.fs.tac_obj_terms.items():
            log_msg(f"    {name:40s} : {pyo.value(expr):>15.2f}")

    @staticmethod
    def _template_heatmap_initialization_func(
            model,
            first_stage_variables,
            uncertain_params,
            solver,
            ):
        """
        Custom function used to initialize a clone of
        the control problem model in advance of solving.

        Parameters
        ----------
        model : ConcreteModel
            (Copy of the) flowsheet model passed to the heatmap
            evalaution method.
        first_stage_variables : list of VarData
            First-stage variables. Note: these have been fixed.
        uncertain_params : list of ParamData
            Uncertain parameters. These have been updated.
        solver : SolverWithBackup or ExtendedFlowsheetWithCostingSolver
            Solver passed to the heatmap function.
            Should have a logger-type attribute with name
            `logger`.

        Returns
        -------
        ComponentMap or None
            Mapping from variables that are not first-stage
            (or fixed)
            to their desired values.

            If None is returned, then the model is initalized to
            to the original solution passed to the heatmap
            evaluation function.
        """
        from pyomo.core import TransformationFactory
        from pyomo.common.collections import ComponentSet
        tol = 1e-4

        slack_model = model.clone()
        violated_cons = []
        active_slack_cons_iter = slack_model.component_data_objects(
            pyo.Constraint, active=True
        )
        for con in active_slack_cons_iter:
            if con.lslack() < -tol or con.uslack() < - tol:
                violated_cons.append(con)

        TransformationFactory("core.add_slack_variables").apply_to(
            model=slack_model,
            targets=violated_cons,
        )
        slack_var_block = slack_model._core_add_slack_variables
        slack_vars = ComponentSet(
            slack_var_block.component_data_objects(pyo.Var)
        )
        for con in violated_cons:
            slack_vars_in_con = (
                ComponentSet(identify_variables(con.expr))
                & slack_vars
            )
            for slack_var in slack_vars_in_con:
                slack_var_relname = slack_var.getname(
                    relative_to=slack_var_block, fully_qualified=True
                )
                if slack_var_relname.startswith("'_slack_minus"):
                    slack_var.set_value(
                        max(0, -pyo.value(model.find_component(con).uslack()))
                    )
                elif slack_var_relname.startswith("'_slack_plus"):
                    slack_var.set_value(
                        max(0, -pyo.value(model.find_component(con).lslack()))
                    )
                else:
                    raise ValueError(f"Unsupported slack var {slack_var.name}")

        slack_obj = next(
            slack_model.component_data_objects(pyo.Objective, active=True)
        )
        slack_obj.deactivate()
        co2_capture_var = (
            slack_model.fs.absorber_section.absorber.co2_capture[0]
        )
        reboiler_duty_var = (
            slack_model.fs.stripper_section.reboiler.heat_duty[0]
        )
        distillate_h2o_comp_var = (
            slack_model.fs.stripper_section.condenser
            .vapor_phase.properties_out[0].mole_frac_comp["H2O"]
        )
        stripper_top_flood_frac_var = (
            slack_model.fs.stripper_section.stripper.flood_fraction[0, 1]
        )
        custom_obj = pyo.Objective(
            expr=(
                slack_obj.expr
                + 1e-3 * co2_capture_var
                + 1e-8 * reboiler_duty_var
                + distillate_h2o_comp_var
                - stripper_top_flood_frac_var
            )
        )

        slack_model.add_component(
            unique_component_name(slack_model, "custom_slack_obj"), custom_obj
        )

        solver.logger.info("Values before:")
        solver.logger.info(f" Slack obj: {pyo.value(slack_obj.expr)}")
        solver.logger.info(f" CO2 capture: {pyo.value(co2_capture_var)}")
        solver.logger.info(f" Reboiler duty: {pyo.value(reboiler_duty_var)}")
        solver.logger.info(
            f" Distillate h2o mole frac: {pyo.value(distillate_h2o_comp_var)}"
        )
        solver.logger.info(
            " Stripper top flood frac: "
            f"{pyo.value(stripper_top_flood_frac_var)}"
        )
        solver.logger.info(f" TAC obj: {pyo.value(slack_model.fs.tac_obj)}")

        res = solver.solve(
            slack_model,
            load_solutions=False,
            tee=False,
        )
        if pyo.check_optimal_termination(res):
            slack_model.solutions.load_from(res)
            init_var_val_map = ComponentMap()
            for var in model.component_data_objects(pyo.Var):
                if not var.fixed:
                    init_var_val_map[var] = (
                        slack_model.find_component(var).value
                    )

            solver.logger.info("Values after:")
            solver.logger.info(f" slack obj: {pyo.value(slack_obj.expr)}")
            solver.logger.info(f" CO2 capture: {pyo.value(co2_capture_var)}")
            solver.logger.info(
                f" Reboiler duty: {pyo.value(reboiler_duty_var)}"
            )
            solver.logger.info(
                " Distillate h2o mole frac: "
                f"{pyo.value(distillate_h2o_comp_var)}"
            )
            solver.logger.info(
                " Stripper top flood frac: "
                f"{pyo.value(stripper_top_flood_frac_var)}"
            )
            solver.logger.info(
                f" TAC obj: {pyo.value(slack_model.fs.tac_obj)}"
            )

            return init_var_val_map

    heatmap_initialization_func = None

    @staticmethod
    def get_flowsheet_results_type():
        """Get flowsheet results type."""
        return MEAFlowsheetEconomicOptResults

    @staticmethod
    def get_model_first_stage_variables(model):
        """Get first-stage variables."""
        return (
            FlowsheetModelData.get_model_first_stage_variables(model)
            + list(model.fs.design_epigraph_vars.values())
        )

    @staticmethod
    def add_pyros_separation_priorities(model, **kwargs):
        """
        Add PyROS separation priorities.
        """
        FlowsheetModelData.add_pyros_separation_priorities(model, **kwargs)
        for epi_con in model.fs.design_epigraph_cons.values():
            model.pyros_separation_priority[epi_con] = 1

    def get_heatmap_expressions_to_evaluate(self):
        """
        Get named expression-like components to evaluate in heatmap
        workflows.
        """
        exprs_to_evaluate = super(
            FlowsheetEconomicOptModelData,
            self,
        ).get_heatmap_expressions_to_evaluate()

        fs = self.original_model.fs
        exprs_to_evaluate.update(
            (var.name, var) for var in fs.design_epigraph_vars.values()
        )
        exprs_to_evaluate.update(
            (expr.name, expr)
            for expr
            in get_costing_expressions_to_evaluate(self.original_model)
        )
        exprs_to_evaluate.update(
            (expr.name, expr)
            for expr in [
                fs.costing_setup.solvent_fill_init[0],
                fs.costing_setup.CO2_capture_rate[0]
            ]
        )
        exprs_to_evaluate.update(
            (expr.name, expr) for expr in fs.ccs_aic_terms.values()
        )
        exprs_to_evaluate.update(
            (expr.name, expr) for expr in fs.ccs_aoc_terms.values()
        )
        exprs_to_evaluate.update(
            (srd_var.name, srd_var) for srd_var in fs.SRD.values()
        )
        exprs_to_evaluate.update(
            (param.name, param)
            for param in self.original_model.ngcc_ccs_costing_output.values()
        )

        return exprs_to_evaluate

    @staticmethod
    def build_square_flowsheet_model(*args, **kwargs):
        """
        Build square flowsheet model.
        """
        fs_model = FlowsheetModelData.build_square_flowsheet_model(
            *args,
            **kwargs,
        )
        fs_model.fs.add_costing()

        return fs_model

    @staticmethod
    def tune_flowsheet_model(fs_model):
        """
        Tune flowsheet optimization model.
        """
        # deactivate proxy objective and associated components
        fs_model.fs.obj.deactivate()
        fs_model.fs.condenser_cw_temperature_outlet_excess_temp.fix()
        fs_model.fs.condenser_cw_temperature_penalty_constraint.deactivate()
        pump_blocks = [
            fs_model.fs.absorber_section.rich_solvent_pump,
            fs_model.fs.absorber_section.lean_solvent_pump,
        ]
        for pump_blk in pump_blocks:
            pump_blk.max_work.fix()
            pump_blk.max_work_con.deactivate()

        distillate_h2o_comp = (
            fs_model
            .fs
            .stripper_section
            .condenser
            .vapor_phase
            .properties_out[0]
            .mole_frac_comp["H2O"]
        )

        fs_model.fs.tac_obj_terms = pyo.Expression(
            ["tac_term", "distillate_h2o_comp_term"],
            initialize={
                "tac_term": fs_model.fs.costing.CCS_TAC * 1e-7,
                "distillate_h2o_comp_term": distillate_h2o_comp * 10,
            },
        )
        fs_model.fs.tac_obj = pyo.Objective(
            expr=sum(fs_model.fs.tac_obj_terms.values()),
            sense=pyo.minimize,
        )
        fs_model.fs.tac_obj.deactivate()

        # set up optimization for TAC per CO2 captured
        # *at the capture target* per train
        flue_gas_co2_mass_flow_per_train = (
            fs_model
            .fs.flue_gas_feed_scaler.properties[0].flow_mass_comp["CO2"]
        ) / fs_model.fs.number_trains.value
        fs_model.fs.co2_capture_target_mass_flow = pyo.Expression(
            expr=(
                (fs_model.fs.co2_capture_target / 100)
                * pyo.units.convert(
                    src=flue_gas_co2_mass_flow_per_train,
                    to_units=pyo.units.tonne / pyo.units.h,
                )
            ),
        )
        fs_model.fs.levelized_cost_of_target_capture = pyo.Expression(
            expr=(
                fs_model.fs.costing.CCS_TAC
                / pyo.units.convert(
                    src=fs_model.fs.co2_capture_target_mass_flow,
                    to_units=pyo.units.tonne / pyo.units.a,
                )
            ),
        )
        fs_model.fs.lcoc_obj = pyo.Objective(
            expr=fs_model.fs.levelized_cost_of_target_capture,
            sense=pyo.minimize,
        )
        fs_model.fs.scaling_factor[fs_model.fs.lcoc_obj] = 0.1

        # add this mainly to prevent solutions with no condenser
        cond_vap_out_temp = (
            fs_model
            .fs
            .stripper_section
            .condenser
            .vapor_phase
            .properties_out[0]
            .temperature
        )
        fs_model.fs.max_cond_temp_out = pyo.Constraint(
            expr=cond_vap_out_temp <= 273.15 + 40,
        )

        _add_ngcc_ccs_costing_output_tracking_components(fs_model)
        add_ccs_capex_cost_epigraph_vars(fs_model)
        add_ccs_capex_cost_term_expressions(fs_model)
        add_ccs_opex_cost_term_expressions(fs_model)

        # upper bound on max reboiler duty to ensure that
        # max steam crossover fraction is not exceeded
        fs_blk = fs_model.fs
        reb_duty_var = fs_blk.stripper_section.reboiler.liquid_phase.heat[0]
        fs_blk.max_reboiler_duty_upper_bound_con = pyo.Constraint(
            expr=(
                fs_blk.design_epigraph_vars["reboiler_heat_duty"]
                * pyo.units.get_units(reb_duty_var)
                * fs_blk.number_trains
                <= pyo.units.convert(
                    src=0.85 * 379.61 * pyo.units.MW,
                    to_units=pyo.units.get_units(reb_duty_var),
                )
            )
        )
        fs_blk.scaling_factor[fs_blk.max_reboiler_duty_upper_bound_con] = (
            SuffixFinder("scaling_factor").find(reb_duty_var)
        )

        # specific reboiler duty is new: initialize it here
        fs = fs_model.fs
        for tidx in fs_model.fs.time:
            srd_var = fs_model.fs.SRD[tidx]
            srd_con = fs_model.fs.calculate_SRD[tidx]
            calculate_variable_from_constraint(srd_var, srd_con)

        fs = fs_model.fs
        design_var_to_oper_expr_map = {
            "rich_solvent_flow_vol": (
                fs.absorber_section.rich_solvent_flow_vol[0]
            ),
            "lean_solvent_flow_vol": (
                fs.stripper_section.lean_solvent_flow_vol[0]
            ),
            "reboiler_heat_duty": (
                fs.stripper_section.reboiler.liquid_phase.heat[0]
            ),
            "CO2_capture_flow_rate": fs.costing_setup.CO2_capture_rate[0],
        }
        for epi_var_idx, oper_expr in design_var_to_oper_expr_map.items():
            fs_model.fs.design_epigraph_vars[epi_var_idx].set_value(
                pyo.value(oper_expr)
            )

        # algorithmic lower bound to avoid numerical issues
        # encountered by some PyROS subproblems
        fs.stripper_section.condenser.area.setlb(100)

        # key upper bounds on column dimensions
        fs.absorber_section.absorber.length_column.setub(36)
        fs.stripper_section.stripper.length_column.setub(36)
        fs.absorber_section.absorber.diameter_column.setub(18)
        fs.stripper_section.stripper.diameter_column.setub(18)

        # LRHX area upper bound
        fs.absorber_section.lean_rich_heat_exchanger.area.setub(25000)

    def solve_deterministic(
            self,
            solver,
            logger=None,
            keep_model_in_results=True,
            *args,
            **kwargs,
            ):
        """
        Solve deterministic flowsheet model.
        """
        # this returns the base results type
        solver_wrapper = self.create_heatmap_solver_wrapper(solver, logger)
        res = solver_wrapper.solve(
            self.original_model,
            load_solutions=False,
            *args,
            costing_tee=True,
            **kwargs,
        )
        if pyo.check_optimal_termination(res):
            logger.info(
                "Solved model to acceptable status. "
                f"Solver results: \n{res.solver}\n"
                "Loading solution to model."
            )

            self.original_model.solutions.load_from(res)
            orig_costing_params = self.original_model.ngcc_ccs_costing_output
            for key, orig_param in orig_costing_params.items():
                orig_param.set_value(res.solver.ngcc_ccs_costing_stats[key])

            # since the solver was invoked on the original model,
            # we need to refresh the scaled model
            self.scaled_model = self._create_scaled_model()
        else:
            logger.warning(
                "Could not solve to acceptable status. "
                f"Solver results: \n{res.solver}"
            )

        # get full set of results
        return self.create_flowsheet_results_object(
            solver_results=res,
            keep_model=True,
        )

    def solve_pyros(
            self,
            uncertainty_set,
            progress_logger,
            keep_model_in_results=True,
            **kwargs,
            ):
        """
        Solve flowsheet model robustly using PyROS.
        """
        pyros_fs_res = super().solve_pyros(
            uncertainty_set=uncertainty_set,
            progress_logger=progress_logger,
            keep_model_in_results=keep_model_in_results,
            **kwargs,
        )
        pyros_solver_res = pyros_fs_res.pyros_solver_results

        acceptable_pyros_terminations = {
            pyros.pyrosTerminationCondition.robust_optimal,
            pyros.pyrosTerminationCondition.robust_feasible,
        }
        pyros_term_cond = pyros_solver_res.pyros_termination_condition
        termination_acceptable = (
            pyros_term_cond in acceptable_pyros_terminations
        )

        run_costing = termination_acceptable and self.with_costing
        if termination_acceptable:
            original_model = pyros_fs_res.model_data.original_model
            temp_costing_model = original_model.clone()
            solver_wrapper_cls = self.get_heatmap_solver_wrapper_type()
            if run_costing:
                progress_logger.info(
                    "Evaluating NGCC + CCS costing metrics "
                    "for PyROS solution... "
                )
                solver_wrapper_cls.add_costing_results(
                    model=temp_costing_model,
                    dof_vars=self.get_model_dof_variables(temp_costing_model),
                    logger=progress_logger,
                )
            temp_model_costing_param = (
                temp_costing_model.ngcc_ccs_costing_output
            )
            for key, temp_param in temp_model_costing_param.items():
                # ensure costing results are reflected in the
                # user-provided model
                original_model.find_component(temp_param).set_value(
                    pyo.value(temp_param)
                )

            # variables of original model may have changed, so we
            # refresh the scaled model to reflect the changes
            self.scaled_model = self._create_scaled_model()

        if not run_costing:
            progress_logger.info(
                "Skipping NGCC + CCS postsolve costing evaluation for "
                "PyROS flowsheet solution."
            )

        return self.create_flowsheet_results_object(
            solver_results=None,
            presolver_results=None,
            pyros_solver_results=pyros_solver_res,
            keep_model=keep_model_in_results,
        )


NGCC_CCS_COSTING_OUTPUTS = {
    "costing_successful": ("Costing Evaluation Was Successful", None),
    "cooling_duty_CCS_compr": ("Total Cooling Duty", "MW"),
    "plant_net_power": ("Plant Net Power", "MW"),
    "IP_LP_crossover_steam_fraction": ("Fraction of Steam Extraction", None),
    "CO2_capture_rate": (
        "Mass Flow Rate of $\\text{CO}_2$ Captured", "tonne/hr",
    ),
    "cost_CO2_avoided": ("Cost of Avoided Carbon", "\\$/tonne"),
    "cost_of_capture": ("Cost of $\\text{CO}_2$ Captured", "\\$/tonne"),
    "FG_Cleanup_TPC": ("FG Cleanup TPC", "MM\\$"),
    "LCOE": ("Levelized Cost of Electricity", "\\$/MWh"),
    "capital_lcoe": ("Levelized Capital Cost of Electricity", "\\$/MWh"),
    "fixed_lcoe": ("Levelized Fixed Cost of Electricity", "\\$/MWh"),
    "variable_lcoe": ("Levelized Variable Cost of Electricity", "\\$/MWh"),
    "transport_lcoe": (
        "Levelized Transporation Cost of Electricity", "\\$/MWh"
    ),
    "co2_emission_cap": (
        "$\\text{CO}_2$ Emission Mass Flow Rate",
        "tonne $\\text{CO}_2/hr$",
    ),
    "capital_cost": ("Capital Cost", "MM\\$/yr"),
    "ngcccap_total_fixed_OM_cost": (
        "Total Fixed Operation and Maintenance Cost",
        "MM\\$/yr",
    ),
    "ngcccap_total_variable_OM_cost": (
        "Total Variable Operation and Maintenance Cost",
        "MM\\$/yr",
    ),
    "ngcccap_transport_cost": ("Transportation Cost", "MM\\$/yr"),
    "ngcccap_capacity_factor": ("Capacity Factor", None),
}


def _add_ngcc_ccs_costing_output_tracking_components(model):
    """
    Add indexed mutable Param component to flowsheet model
    to more easily track ex-post-facto IDAES costing results.

    Quantities tracked are:
    - LCOE: levelized cost of electricity, $/MWh
    - plant_net_power: plant net power, MW
    - cap_factor: plant capacity factor
    - total_annualized_cost: Total annualized cost, MM$/yr
    - ngcccap_total_annualized_cost: NGCC capture plant total
      annualized cost, MM$/yr
    - Cost_of_capture: Cost per metric ton of CO2 captured, $/metric ton
    - Cost_CO2_avoided: Cost of avoided CO2, $/metric ton
    """
    model.ngcc_ccs_costing_output = pyo.Param(
        list(NGCC_CCS_COSTING_OUTPUTS.keys()),
        doc=(
            "Components for tracking NGCC + CCS costing results",
        ),
        within=pyo.Any,
        mutable=True,
        initialize=float("nan"),
    )


def get_ngcc_ccs_costing_output_expr_map(model):
    """
    Get model expressions for NGCC + CCS costing output.
    """
    fs_blk = model.fs
    expr_map = dict(
        costing_successful=None,
        cooling_duty_CCS_compr=pyo.units.convert(
            fs_blk.costing.cooling_duty_CCS_compr,
            to_units=pyo.units.MW,
        ),
        plant_net_power=fs_blk.costing.plant_net_power,
        IP_LP_crossover_steam_fraction=(
            fs_blk.costing_setup.IP_LP_crossover_steam_fraction[0]
        ),
        CO2_capture_rate=pyo.units.convert(
            fs_blk.costing_setup.CO2_capture_rate[0],
            to_units=pyo.units.tonne/pyo.units.hr,
        ),
        cost_CO2_avoided=fs_blk.costing.cost_CO2_avoided,
        cost_of_capture=fs_blk.costing.cost_of_capture,
        FG_Cleanup_TPC=fs_blk.costing.FG_Cleanup_TPC / 1e6,
        LCOE=fs_blk.costing.LCOE,
        capital_lcoe=fs_blk.costing.capital_lcoe,
        fixed_lcoe=fs_blk.costing.fixed_lcoe,
        variable_lcoe=fs_blk.costing.variable_lcoe,
        transport_lcoe=fs_blk.costing.transport_lcoe,
        co2_emission_cap=pyo.units.convert(
            fs_blk.costing.CO2_emission_cap,
            to_units=pyo.units.tonne/pyo.units.hr,
        ),
        capital_cost=fs_blk.costing.capital_cost,
        ngcccap_total_fixed_OM_cost=fs_blk.ngcccap.total_fixed_OM_cost,
        ngcccap_total_variable_OM_cost=(
            fs_blk.ngcccap.total_variable_OM_cost[0]
        ),
        ngcccap_transport_cost=fs_blk.ngcccap.transport_cost,
        ngcccap_capacity_factor=fs_blk.ngcccap.capacity_factor,
    )

    assert expr_map.keys() == NGCC_CCS_COSTING_OUTPUTS.keys()

    return expr_map


class FlowsheetNGCCPlusCCSCostingSolver:
    """
    Wrapper around a Pyomo solver, designed to
    perform post-solve NGCC + CCS costing evaluation
    for a solution to an MEA flowsheet model with CCS
    costing components.
    """

    SOLVERBACKUP_RESULTS_SOLVER_ATTRS = ["ngcc_ccs_costing_stats"]

    def __init__(
            self,
            solver,
            logger,
            flowsheet_model_data_type,
            with_costing=True,
            ):
        """Initialize self (see class docstring).

        """
        if not isinstance(solver, SolverWithBackup):
            # SolverWithBackup helps us track more
            # computational performance metrics
            solver = SolverWithBackup(solver, logger=logger)
        self.solver = solver
        self.logger = logger
        self.flowsheet_model_data_type = flowsheet_model_data_type
        self.with_costing = with_costing

    def available(self, exception_flag=False):
        return True

    def version(self):
        return (0, 0, 0, 0)

    @staticmethod
    def add_costing_results(model, dof_vars, logger=None, tee=False):
        """
        Add NGCC + CCS costing results to flowsheet model,
        in-place.
        """
        if logger is None:
            logger = default_logger

        # cost evaluation expects a simulation model.
        # temporarily fix the DOF variables
        unfixed_dof_vars = []
        for var in dof_vars:
            assert var.model() is model
            if not var.fixed:
                unfixed_dof_vars.append(var)
                var.fix()

        # remove CCS costing components
        # to prevent name clashes with NGCC + CCS
        # costing components
        model.fs.del_component(model.fs.costing)
        model.fs.del_component(model.fs.costing_setup)
        tpc_blocks = [
            blk for blk in model.fs.component_data_objects(
                pyo.Block, active=True, descend_into=False
            )
            if re.fullmatch(r"b[\d]+", blk.local_name) is not None
        ]
        for blk in tpc_blocks:
            model.fs.del_component(blk)

        # setup logging output according to desired configuration
        if tee:
            solver_log_context_mgr = capture_output
            solver_log_context_kwds = dict(
                output=LogStream(level=logging.DEBUG, logger=logger)
            )
        else:
            solver_log_context_mgr = nullcontext
            solver_log_context_kwds = dict()

        with solver_log_context_mgr(**solver_log_context_kwds):
            try:
                MEACombinedFlowsheetDataNGCCPlusCCS.add_costing(
                    self=model.fs,
                )
            except (RuntimeError, ApplicationError, ValueError):
                # if costing fails, exception definitely raised.
                # NOTE: the components added for tracking costing
                #       are set to nan
                logger.exception(
                    "Could not sucessfully solve costing model."
                )
                costing_successful = False
                for key, param in model.ngcc_ccs_costing_output.items():
                    param.set_value(float("nan"))
                model.ngcc_ccs_costing_output["costing_successful"].set_value(
                    costing_successful
                )
            else:
                # costing successful. update model with those stats
                costing_successful = True
                costing_output_expr_map = (
                    get_ngcc_ccs_costing_output_expr_map(model)
                )
                for key, expr in costing_output_expr_map.items():
                    model.ngcc_ccs_costing_output[key].set_value(
                        pyo.value(expr)
                    )
                model.ngcc_ccs_costing_output["costing_successful"].set_value(
                    costing_successful
                )
            finally:
                for var in unfixed_dof_vars:
                    var.unfix()

        return costing_successful

    def solve(
            self,
            model,
            *args,
            tee=False,
            costing_tee=False,
            load_solutions=False,
            **kwargs,
            ):
        """
        Solve an unscaled flowsheet model and perform rigorous
        IDAES-based costing evaluations.
        """
        # use scaling solver to account for scaling
        if not isinstance(self.solver, SolverWithScaling):
            scaling_solver = SolverWithScaling(self.solver, self.logger)

        self.logger.debug(
            f"{type(self).__name__} invoking {scaling_solver}..."
        )
        res = scaling_solver.solve(
            model=model,
            load_solutions=False,
            tee=tee,
            *args,
            **kwargs,
        )
        self.logger.debug(
            f"{type(self).__name__} back from invocation of subordinate "
            "solver. "
            "Preparing final results object..."
        )

        final_res = SolverResults()
        final_res.problem.update(res.problem.items())
        final_res.solver.update(res.solver.items())

        ngcc_ccs_costing_results = {
            key: np.nan for key in model.ngcc_ccs_costing_output.keys()
        }
        if res.solution:
            mdl = model.clone()
            mdl.solutions.load_from(res)

            # store solved model prior to costing evaluation,
            # as the costing evaluation may result in slight
            # changes to the state variables
            mdl.solutions.store_to(
                results=final_res,
                cuid=False,
                skip_stale_vars=False,
            )

            if pyo.check_optimal_termination(res) and self.with_costing:
                if self.with_costing:
                    self.add_costing_results(
                        model=mdl,
                        dof_vars=(
                            self
                            .flowsheet_model_data_type
                            .get_model_dof_variables(mdl)
                        ),
                        logger=self.logger,
                        tee=costing_tee,
                    )

            ngcc_ccs_costing_results.update({
                key: pyo.value(param)
                for key, param in mdl.ngcc_ccs_costing_output.items()
            })

        final_res.solver.ngcc_ccs_costing_stats = ngcc_ccs_costing_results

        self.logger.debug(
            f"{type(self).__name__} done preparing results object."
        )

        if load_solutions:
            self.load_solution(model, final_res)

        return final_res

    def load_solution(self, model, results):
        """
        Load solution from results object to model,
        ensuring that components for tracking the post-solve
        NGCC + CCS costing evaluation are also updated.
        """
        model.solutions.load_from(results)
        # also ensure NGCC + CCS costing output loaded
        for key, param in model.ngcc_ccs_costing_output.items():
            param.set_value(results.solver.ngcc_ccs_costing_stats[key])
        self.logger.debug(
            f"{type(self).__name__} loaded solution to model."
        )


def get_costing_expressions_to_evaluate(model):
    """
    Get costing expressions to evaluate.

    Returns
    -------
    list of Var or ExpressionData
        Costing expressions. Note: all entries of the list
        are named components.
    """
    equipment_costs = [
        # in MM $(2018), not annualized
        model.fs.costing.absorber_column_cost,
        model.fs.costing.absorber_packing_cost,
        model.fs.costing.lean_rich_hex_cost,
        model.fs.costing.rich_solvent_pump_cost,
        model.fs.costing.lean_solvent_pump_cost,
        model.fs.costing.stripper_column_cost,
        model.fs.costing.stripper_packing_cost,
        model.fs.costing.stripper_condenser_cost,
        model.fs.costing.stripper_reboiler_cost,
        model.fs.costing.reboiler_condensate_pot_cost,
        model.fs.costing.stripper_reflux_drum_cost,
        model.fs.costing.stripper_reflux_pump_cost,
        model.fs.costing.solvent_filtration_cost,
        model.fs.costing.solvent_storage_tank_cost,
        model.fs.costing.RemovalSystem_Equip_Adjust,
    ]
    pump_utility_costs = [
        # in USD/h
        model.fs.costing.UC_rich_solvent_pump,
        model.fs.costing.UC_lean_solvent_pump,
    ]
    other_utility_costs = [
        # Other utility costs, in USD/s
        model.fs.costing.UC_reboiler,
        model.fs.costing.UC_condenser,
        model.fs.costing.UC_H2O_makeup,
    ]
    total_equipment_costs = [
        # equipment costs, in USD
        model.fs.costing.CCS_TEC,
        model.fs.costing.CCS_TPC,
    ]
    annualized_costs = [
        # annualized economic metrics, in USD/yr
        model.fs.costing.CCS_AIC,
        model.fs.costing.CCS_AOC,
        model.fs.costing.CCS_TAC,
        model.fs.costing.CCS_TAC_perCO2captured,
    ]

    # immunize against changes to the units
    for puc in pump_utility_costs:
        assert_units_equivalent(
            pyo.units.get_units(puc),
            pyo.units.USD_2018 / pyo.units.h,
        )
    for ouc in other_utility_costs:
        assert_units_equivalent(
            pyo.units.get_units(ouc),
            pyo.units.USD_2018 / pyo.units.s,
        )

    return (
        equipment_costs
        + pump_utility_costs
        + other_utility_costs
        + total_equipment_costs
        + annualized_costs
        + list(model.fs.tac_obj_terms.values())
        + [model.fs.tac_obj]
        + [model.fs.costing_setup.CO2_capture_rate[0]]
        + [model.fs.co2_capture_target_mass_flow]
        + [model.fs.levelized_cost_of_target_capture]
        + [model.fs.lcoc_obj]
    )


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_specific_reboiler_duty(deterministic_results_df, outdir):
    """
    Plot specific reboiler duty vs CO2 capture target.
    """
    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr="Specific Reboiler Duty",
        unitstr="MJ/kg $\\mathrm{CO_2}$",
    ))
    ax.plot(
        acceptable_results_df.index,
        acceptable_results_df[("specific_reboiler_duty",) * 2].to_numpy(),
    )
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )

    fig.savefig(
        os.path.join(outdir, "srd.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_ccs_tac_obj_terms(deterministic_results_df, outdir):
    """
    Plot terms of the total annualized cost objective.
    """
    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, ax = plt.subplots(figsize=(3.1, 3.5))
    for col in acceptable_results_df["tac_obj"].columns:
        ax.plot(
            acceptable_results_df.index,
            acceptable_results_df[("tac_obj", col)].to_numpy(),
            label=f"\\texttt{{{col}}}",
        )

    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str("Objective-Related Quantity"))

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

    fig.savefig(
        os.path.join(outdir, "tac_obj.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def _plot_ccs_equipment_costs(acceptable_results_df, fig, ax):
    """
    Plot CCS equipment costs.
    """
    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr="Annualized Investment Cost Contribution",
        unitstr="MM \\$/yr",
    ))

    # to enhance readability, want the highest costs to
    # be plotted first
    sorted_ccs_aic_results_df = acceptable_results_df.ccs_aic_terms[
        acceptable_results_df
        .ccs_aic_terms
        .mean(axis=0)
        .sort_values(ascending=False)
        .index
    ]
    ax.plot(
        sorted_ccs_aic_results_df.index,
        sorted_ccs_aic_results_df.to_numpy() / 1e6,
        label=[
            col.replace("_", " ")
            for col in sorted_ccs_aic_results_df.columns
        ],
    )
    ax.legend(
        bbox_to_anchor=(0, -0.2),
        loc="upper left",
        ncol=1,
    )
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=sorted_ccs_aic_results_df.index,
    )


def _plot_ccs_operating_costs(acceptable_results_df, fig, ax):
    """
    Plot CCS operating costs.
    """
    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr="Annualized Operating Cost Contribution",
        unitstr="MM \\$/yr",
    ))
    sorted_ccs_aoc_results_df = acceptable_results_df.ccs_aoc_terms[
        acceptable_results_df
        .ccs_aoc_terms
        .mean(axis=0)
        .sort_values(ascending=False)
        .index
    ]
    ax.plot(
        acceptable_results_df.index,
        sorted_ccs_aoc_results_df.to_numpy() / 1e6,
        label=[
            col.replace("_", " ").replace("H2O", "$\\mathrm{H_2O}$")
            for col in sorted_ccs_aoc_results_df.columns
        ],
    )
    ax.legend(
        bbox_to_anchor=(0, -0.2),
        loc="upper left",
        ncol=1,
    )
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=sorted_ccs_aoc_results_df.index,
    )


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def _plot_ccs_tac_summary(acceptable_results_df, fig, ax):
    """
    Plot summary of CCS TAC metrics.
    """
    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr="Annualized Cost",
        unitstr="MM \\$/yr",
    ))

    annualized_cost_cols = [
        "fs.costing.CCS_TAC",
        "fs.costing.CCS_AIC",
        "fs.costing.CCS_AOC",
    ]
    for col in annualized_cost_cols:
        col_label = col.split("fs.costing.")[-1].split("CCS_")[-1]
        ax.plot(
            acceptable_results_df.index,
            (
                acceptable_results_df[("ccs_costing", col)].to_numpy()
                / 1e6  # convert to MM USD
            ),
            label=col_label,
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


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def _plot_ccs_tac_per_co2_captured(acceptable_results_df, fig, ax):
    """Plot TAC per CO2 captured."""
    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr="Levelized Cost of Capture",
        unitstr="\\$/tonne $\\mathrm{CO_2}$",
    ))

    tac_per_co2_col = "fs.costing.CCS_TAC_perCO2captured"
    ax.plot(
        acceptable_results_df.index,
        acceptable_results_df[("ccs_costing", tac_per_co2_col)].to_numpy(),
    )
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_ccs_costing_results(deterministic_results_df, outdir):
    """
    Plot economic CCS costing results.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    fig, (tac_ax, tac_per_co2_ax, equip_ax, oper_ax) = (
        plt.subplots(ncols=4, figsize=(12.4, 6))
    )

    _plot_ccs_tac_summary(acceptable_results_df, fig, tac_ax)
    _plot_ccs_tac_per_co2_captured(acceptable_results_df, fig, tac_per_co2_ax)
    _plot_ccs_equipment_costs(acceptable_results_df, fig, equip_ax)
    _plot_ccs_operating_costs(acceptable_results_df, fig, oper_ax)

    fig.savefig(
        os.path.join(outdir, "ccs_costs.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_ngcc_ccs_lcoe_breakdown_stack(deterministic_results_df, outdir):
    """
    Generate NGCC + CCS LCOE breakdown stackplots.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    lcoe_cols = [
        "capital_lcoe",
        "fixed_lcoe",
        "variable_lcoe",
        "transport_lcoe",
    ]

    fig, ax = plt.subplots(figsize=(3.1, 3.5))

    bottoms = np.zeros(acceptable_results_df.index.size)
    for lcoe_col in lcoe_cols:
        lcoe_vals = (
            acceptable_results_df
            .ngcc_ccs_costing_output[lcoe_col]
            .astype(float)
            .to_numpy()
        )
        ax.plot(
            acceptable_results_df.index,
            bottoms + acceptable_results_df.ngcc_ccs_costing_output[lcoe_col],
        )
        bottoms += lcoe_vals

    ax.stackplot(
        acceptable_results_df.index,
        (
            acceptable_results_df
            .ngcc_ccs_costing_output[lcoe_cols]
            .astype(float)
            .to_numpy()
        ).T,
        alpha=0.5,
        colors=(
            plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(lcoe_cols)]
        ),
        labels=[col.split("_lcoe")[0] for col in lcoe_cols],
    )

    ax.set_xlabel(wrap_quantity_str(
        namestr="$\\mathrm{CO_2}$ Capture Target",
        unitstr="\\%",
    ))
    ax.set_ylabel(wrap_quantity_str(
        namestr=f"NGCC + CCS {NGCC_CCS_COSTING_OUTPUTS['LCOE'][0]}",
        unitstr=NGCC_CCS_COSTING_OUTPUTS["LCOE"][1],
    ))
    set_nonoverlapping_fixed_xticks(
        fig=fig,
        ax=ax,
        locs=acceptable_results_df.index,
    )
    ax.legend(
        bbox_to_anchor=(0, -0.2),
        loc="upper left",
        ncol=1,
    )
    fig.savefig(
        os.path.join(outdir, "ngcc_ccs_costing_lcoe_breakdown.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_postsolve_ngcc_ccs_costing_results(deterministic_results_df, outdir):
    """
    Plot postsolve NGCC CCS costing results.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acceptable_results_df = deterministic_results_df[
        deterministic_results_df[
            ("solver_termination", "termination_condition")
        ].isin({"locallyOptimal", "optimal", "globallyOptimal"})
    ]

    keys_to_plot = [
        key for key in NGCC_CCS_COSTING_OUTPUTS if not key.endswith("_lcoe")
    ]
    for col in keys_to_plot:
        qty_name, qty_units = NGCC_CCS_COSTING_OUTPUTS[col]
        fig, ax = plt.subplots(figsize=(3.1, 2.8))

        ax.plot(
            acceptable_results_df.index,
            (
                acceptable_results_df
                .ngcc_ccs_costing_output[col]
                .astype(float)
                .to_numpy()
            ),
        )
        ax.set_xlabel(wrap_quantity_str(
            namestr="$\\mathrm{CO_2}$ Capture Target",
            unitstr="\\%",
        ))
        ax.set_ylabel(wrap_quantity_str(
            namestr=f"NGCC + CCS {qty_name}",
            unitstr=qty_units,
        ))
        set_nonoverlapping_fixed_xticks(
            fig=fig,
            ax=ax,
            locs=acceptable_results_df.index,
        )

        fig.savefig(
            os.path.join(outdir, f"ngcc_ccs_costing_{col}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


class MEAFlowsheetEconomicOptResults(MEAFlowsheetResults):
    """
    Results object for economic flowsheet optimization.
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
        super().__init__(
            model_data=model_data,
            solver_results=solver_results,
            presolver_results=presolver_results,
            pyros_solver_results=pyros_solver_results,
            keep_model=keep_model,
        )

        self._setup_ccs_costing_results(model_data.original_model)
        self._setup_tac_obj_results(model_data.original_model)
        self._setup_design_epigraph_var_results(model_data.original_model)
        self._setup_solvent_fill_init_results(model_data.original_model)
        self._setup_ccs_aic_term_results(model_data.original_model)
        self._setup_ccs_aoc_term_results(model_data.original_model)
        self._setup_ccs_reboiler_srd_results(model_data.original_model)
        self._setup_ngcc_ccs_costing_output_results(model_data.original_model)

    def _setup_ngcc_ccs_costing_output_results(self, model):
        """
        Setup NGCC + CCS postsolve costing results.
        """
        # keep track of idaes costing stats
        self.ngcc_ccs_costing_output = Bunch(**{
            key: pyo.value(param) for key, param in
            model.ngcc_ccs_costing_output.items()
        })

    def _setup_ccs_reboiler_srd_results(self, model):
        """
        Setup reboiler specific duty result.
        """
        self.specific_reboiler_duty = Bunch(
            specific_reboiler_duty=pyo.value(model.fs.SRD[0]),
        )

    def _setup_ccs_aic_term_results(self, model):
        """
        Set up CCS AIC terms (annual CAPEX cost contributions),
        all in USD (2018)/yr.
        """
        self.ccs_aic_terms = Bunch(
            **{
                key: pyo.value(expr)
                for key, expr in model.fs.ccs_aic_terms.items()
            }
        )

    def _setup_ccs_aoc_term_results(self, model):
        """
        Set up CCS AOC terms (annual OPEX cost contributions),
        all in USD (2018)/yr.
        """
        self.ccs_aoc_terms = Bunch(
            **{
                key: pyo.value(expr)
                for key, expr in model.fs.ccs_aoc_terms.items()
            }
        )

    def _setup_solvent_fill_init_results(self, model):
        """
        Setup solvent initial fill results.

        Bunch with single entry `solvent_fill_init` (in kg).
        """
        self.solvent_fill_init = Bunch(
            solvent_fill_init=pyo.value(
                model.fs.costing_setup.solvent_fill_init[0]
            ),
        )

    def _setup_design_epigraph_var_results(self, model):
        """
        Setup design epigraph variable results.

        Sets up a Bunch attribute with float-type entries:

        rich_solvent_flow_vol
            Rich solvent volumetric flow, m3/s.
        lean_solvent_flow_vol
            Lean solvent volumetric flow, m3/s.
        reboiler_heat_duty
            Reboiler heat duty, W.
        CO2_capture_flow_rate
            CO2 capture flow rate, lb/h.
        solvent_fill_init
            Solvent initial fill, kg.
        """
        self.design_epigraph_vars = Bunch(**{
            name: var.value
            for name, var in model.fs.design_epigraph_vars.items()
        })

    def _setup_ccs_costing_results(self, model):
        """
        Set up CCS costing results.

        Equipment costs are in MM $ (2018). These costs
        have not been annualized.

        Utility costs as follows:
        - UC_rich_solvent_pump
            Rich solvent pump cost, $/h.
        - UC_lean_solvent_pump
            Lean solvent pump cost, $/h.
        - UC_reboiler
            Reboiler cost, $/s.
        - UC_condenser
            Condenser cost, $/s.
        - UC_H2O_makeup
            H2O makeup cost, $/s.

        Annualized costs
        - CCS_TAC
            Total annualized cost, $/yr
        - CCS_AIC
            Annual investment cost, $/yr
        - CCS_AOC
            Annual operating cost, $/yr
        - CCS_TAC_perCO2captured
            TAC per unit CO2 captured, $/tonne

        Other non-annualized costs
        - CCS_TEC
            Total equipment cost? ($)
        - CCS_TPC
            Total plant cost? ($)
        """
        # add Bunch of economic optimization results
        CCS_cost_summary_component_costs = (
            get_costing_expressions_to_evaluate(model)
        )

        self.ccs_costing = Bunch(**{
            costing_comp.name: pyo.value(costing_comp)
            for costing_comp in CCS_cost_summary_component_costs
        })

    def _setup_tac_obj_results(self, model):
        """
        Set up CCS costing TAC objective results.
        """
        self.tac_obj = Bunch(
            tac_obj=pyo.value(model.fs.tac_obj),
            **{
                name: pyo.value(expr)
                for name, expr in model.fs.tac_obj_terms.items()
            },
        )

    def to_dict(self, include_detailed_stream_results=False):
        """
        Cast results stored in self to dict.
        """
        fs_res_dict = super().to_dict(
            include_detailed_stream_results=include_detailed_stream_results,
        )
        fs_res_dict["ccs_costing"] = dict(self.ccs_costing)
        fs_res_dict["tac_obj"] = dict(self.tac_obj)
        fs_res_dict["design_epigraph_vars"] = dict(self.design_epigraph_vars)
        fs_res_dict["solvent_fill_init"] = dict(self.solvent_fill_init)
        fs_res_dict["ccs_aic_terms"] = dict(self.ccs_aic_terms)
        fs_res_dict["ccs_aoc_terms"] = dict(self.ccs_aoc_terms)
        fs_res_dict["specific_reboiler_duty"] = dict(
            self.specific_reboiler_duty
        )
        fs_res_dict["ngcc_ccs_costing_output"] = dict(
            self.ngcc_ccs_costing_output
        )

        return fs_res_dict

    @staticmethod
    def plot_fs_results(fs_results_dict, outdir):
        """
        Plot response of flowsheet optimization result
        to CO2 capture target.
        """
        MEAFlowsheetResults.plot_fs_results(
            fs_results_dict=fs_results_dict,
            outdir=outdir,
        )

        # plot CCS costing metrics
        fs_res_df = MEAFlowsheetEconomicOptResults.create_fs_results_dataframe(
            target_to_fs_res_dict=fs_results_dict,
        )
        plot_ccs_costing_results(fs_res_df, outdir)
        plot_ccs_tac_obj_terms(fs_res_df, outdir)
        plot_specific_reboiler_duty(fs_res_df, outdir)
        plot_postsolve_ngcc_ccs_costing_results(fs_res_df, outdir)
        plot_ngcc_ccs_lcoe_breakdown_stack(fs_res_df, outdir)


def process_solve_results(
        indir,
        pyros_dr_order,
        label_conf_lvls_as_stdevs=False,
        cov_mat_infile="cov_matrix_k_eq_He_05022022.csv",
        output_plot_fmt="png",
        ):
    """
    Process solve results.
    """
    from combined_flowsheet_pyros_base import (
        default_capture_target_sens_plot_data_list,
        default_conf_lvl_sens_plot_data_list,
        default_grouped_conf_lvl_sens_plot_data_list,
        default_stacked_conf_lvl_sens_plot_data_list,
        QuantityPlotData,
        StackedPlotData,
        GroupedPlotDataContainer,
        GroupedPlotData,
        get_all_solve_results,
        process_solve_results_df,
    )

    res_df = get_all_solve_results(indir, pyros_dr_order=pyros_dr_order)

    ccs_costing_plot_data_list = [
        QuantityPlotData(
            qty_name="CCS TAC Objective Value",
            qty_unit=None,
            multiplier=1,
            col_to_plot=("tac_obj", "tac_obj"),
            fname="ccs_tac_obj",
        ),
        QuantityPlotData(
            qty_name="Annual Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_costing", "fs.costing.CCS_AIC"),
            fname="ccs_aic",
            inset_targ_lims=(89.5, 97),
        ),
        QuantityPlotData(
            qty_name="Annual Operating Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_costing", "fs.costing.CCS_AOC"),
            fname="ccs_aoc",
        ),
        QuantityPlotData(
            qty_name="Total Annualized Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_costing", "fs.costing.CCS_TAC"),
            fname="ccs_tac",
        ),
        QuantityPlotData(
            qty_name="Levelized Cost of Capture",
            qty_unit="\\$/metric ton $\\text{CO}_2$",
            multiplier=1,
            col_to_plot=("ccs_costing", "fs.costing.CCS_TAC_perCO2captured"),
            fname="ccs_lcoc",
            inset_targ_lims=(89.5, 97),
        ),
        QuantityPlotData(
            qty_name="Levelized Cost of Target Capture",
            qty_unit="\\$/metric ton $\\text{CO}_2$",
            multiplier=1,
            col_to_plot=("ccs_costing", "fs.levelized_cost_of_target_capture"),
            fname="ccs_lcotc",
            inset_targ_lims=(89.5, 97),
        ),
        QuantityPlotData(
            qty_name="$\\text{CO}_2$ Capture Mass Flow Rate",
            qty_unit="metric ton $\\text{CO}_2$/h",
            multiplier=1 / 2204.62 / 4,  # divide by 4, as there are 4 trains
            col_to_plot=(
                "ccs_costing",
                "fs.costing_setup.CO2_capture_rate[0.0]"
            ),
            fname="co2_capture_mass_flow",
        ),
        QuantityPlotData(
            qty_name="Annualized Absorber Column Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "absorber_column"),
            fname="ccs_absorber_column_cost",
        ),
        QuantityPlotData(
            qty_name="Annualized Absorber Packing Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "absorber_packing"),
            fname="ccs_absorber_packing_cost",
        ),
        QuantityPlotData(
            qty_name="Annualized Stripper Column Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "stripper_column"),
            fname="ccs_stripper_column",
        ),
        QuantityPlotData(
            qty_name="Annualized Stripper Packing Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "stripper_packing"),
            fname="ccs_stripper_packing_cost",
        ),
        QuantityPlotData(
            qty_name="Annualized Reboiler Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "stripper_reboiler"),
            fname="ccs_stripper_reboiler_cost",
        ),
        QuantityPlotData(
            qty_name="Annualized Condenser Investment Cost",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "stripper_condenser"),
            fname="ccs_stripper_condenser_cost",
        ),
        QuantityPlotData(
            qty_name="Annualized Investment Cost of Total Solvent Purchased",
            qty_unit="MM \\$/yr",
            multiplier=1e-6,
            col_to_plot=("ccs_aic_terms", "total_solvent_purchased"),
            fname="ccs_solvent_initial_fill_cost",
        ),
        QuantityPlotData(
            qty_name="Reboiler Utility Cost",
            qty_unit="MM \\$/yr",
            multiplier=31_536_000 / 1e6,
            col_to_plot=("ccs_costing", "fs.costing.UC_reboiler"),
            fname="ccs_uc_reboiler",
        ),
        QuantityPlotData(
            qty_name="Condenser Utility Cost",
            qty_unit="MM \\$/yr",
            multiplier=31_536_000 / 1e6,
            col_to_plot=("ccs_costing", "fs.costing.UC_condenser"),
            fname="ccs_uc_condenser",
        ),
        QuantityPlotData(
            qty_name="$\\text{H_2O}$ Makeup Utility Cost",
            qty_unit="MM \\$/yr",
            multiplier=31_536_000 / 1e6,
            col_to_plot=("ccs_costing", "fs.costing.UC_H2O_makeup"),
            fname="ccs_uc_h2o_makeup",
        ),
        QuantityPlotData(
            qty_name="Rich Solvent Pump Utility Cost",
            qty_unit="MM \\$/yr",
            multiplier=31_536_000 / 60 / 1e6,
            col_to_plot=("ccs_costing", "fs.costing.UC_rich_solvent_pump"),
            fname="ccs_uc_rich_solvent_pump",
        ),
        QuantityPlotData(
            qty_name="Lean Solvent Pump Utility Cost",
            qty_unit="MM \\$/yr",
            multiplier=31_536_000 / 60 / 1e6,
            col_to_plot=("ccs_costing", "fs.costing.UC_lean_solvent_pump"),
            fname="ccs_uc_lean_solvent_pump",
        ),
        QuantityPlotData(
            qty_name="Specific Reboiler Duty",
            qty_unit="MJ/kg",
            multiplier=1,
            col_to_plot=("specific_reboiler_duty",) * 2,
            fname="specific_reboiler_duty",
        ),
    ]

    co2_cap_rate_capacity_name = (
        "design_epigraph_vars", "CO2_capture_rate"
    )
    res_df[co2_cap_rate_capacity_name] = (
        res_df[
            ("design_epigraph_vars", "CO2_capture_flow_rate")
        ].astype(float)
        / 2204.62  # lb to tonne
        / 3600     # per hr to per s
        / (
            res_df[
                ("absorber_column", "vapor_inlet_flow_rate")
            ].astype(float)
            * res_df[
                ("absorber_column", "vapor_inlet_co2_mole_frac")
            ].astype(float)
            * 44.009  # co2 molar mass, kg / kmol
            / 1000    # kg to tonne
            * 4       # account for number of trains
        )
        * 100  # percent
    )
    res_df[("ngcc_ccs_alt_calc", "cost_of_targ_capture")] = (
        res_df[("ngcc_ccs_costing_output", "cost_of_capture")].astype(float)
        * res_df[co2_cap_rate_capacity_name]
        / res_df[("absorber_column", "co2_capture_target")].astype(float)
    )

    ccs_epigraph_plot_data_list = [
        QuantityPlotData(
            qty_name="Maximum Rich Solvent Volumetric Flow Rate",
            qty_unit="$\\text{m}^3$/s",
            multiplier=1,
            col_to_plot=("design_epigraph_vars", "rich_solvent_flow_vol"),
            fname="ccs_epigraph_rich_solvent",
        ),
        QuantityPlotData(
            qty_name="Maximum Lean Solvent Volumetric Flow Rate",
            qty_unit="$\\text{m}^3$/s",
            multiplier=1,
            col_to_plot=("design_epigraph_vars", "lean_solvent_flow_vol"),
            fname="ccs_epigraph_lean_solvent",
        ),
        QuantityPlotData(
            qty_name="Maximum Reboiler Heat Duty",
            qty_unit="MW",
            multiplier=1e-6,
            col_to_plot=("design_epigraph_vars", "reboiler_heat_duty"),
            fname="ccs_epigraph_reboiler_duty",
        ),
        QuantityPlotData(
            qty_name="Maximum $\\text{CO}_2$ Capture Mass Flow Rate",
            qty_unit="metric ton/hr",
            multiplier=1 / 2204.62 / 4,  # divide by number of trains
            col_to_plot=("design_epigraph_vars", "CO2_capture_flow_rate"),
            fname="ccs_epigraph_co2_capture_rate",
        ),
        QuantityPlotData(
            qty_name="Maximum $\\text{CO}_2$ Capture Rate",
            qty_unit="\\%",
            multiplier=1,
            col_to_plot=co2_cap_rate_capacity_name,
            fname="ccs_epigraph_co2_capture_rate_percent",
        ),
        QuantityPlotData(
            qty_name="Maximum Solvent Fill",
            qty_unit="metric ton",
            multiplier=1e-3 / 4,  # divide by 4 as there are 4 trains
            col_to_plot=("design_epigraph_vars", "solvent_fill_init"),
            fname="ccs_epigraph_solvent_fill_init",
        ),
    ]

    ngcc_ccs_costing_plot_data_list = [
        QuantityPlotData(
            qty_name="NGCC + CCS Total Cooling",
            qty_unit="MW",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "cooling_duty_CCS_compr"),
            fname="ngcc_ccs_postsolve_cooling_duty_CCS_compr",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Plant Net Power",
            qty_unit="MW",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "plant_net_power"),
            fname="ngcc_ccs_postsolve_plant_net_power",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS IP/LP Crossover Steam Fraction",
            qty_unit="MW",
            multiplier=1,
            col_to_plot=(
                "ngcc_ccs_costing_output",
                "IP_LP_crossover_steam_fraction"
            ),
            fname="ngcc_ccs_postsolve_IP_LP_crossover_steam_fraction",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Cost of Avoided Carbon",
            qty_unit="\\$/$\\text{metric ton}\\,\\text{CO}_2$",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "cost_CO2_avoided"),
            fname="ngcc_ccs_postsolve_cost_CO2_avoided",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Cost of Capture",
            qty_unit="\\$/$\\text{metric ton}\\,\\text{CO}_2$",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "cost_of_capture"),
            fname="ngcc_ccs_postsolve_lcoc",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Cost of Capture",
            qty_unit="\\$/$\\text{metric ton}\\,\\text{CO}_2$",
            multiplier=1,
            col_to_plot=("ngcc_ccs_alt_calc", "cost_of_targ_capture"),
            fname="ngcc_ccs_postsolve_lcotc",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS FG Cleanup TPC",
            qty_unit="MM\\$",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "FG_Cleanup_TPC"),
            fname="ngcc_ccs_postsolve_fg_cleanup_tpc",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Levelized Cost of Electricity",
            qty_unit="\\$/MWh",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "LCOE"),
            fname="ngcc_ccs_postsolve_lcoe",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Levelized Capital Cost",
            qty_unit="\\$/MWh",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "capital_lcoe"),
            fname="ngcc_ccs_postsolve_lcoe_capital",
        ),
        QuantityPlotData(
            qty_name=(
                "NGCC + CCS Levelized Fixed Operation and Maintenance Cost"
            ),
            qty_unit="\\$/MWh",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "fixed_lcoe"),
            fname="ngcc_ccs_postsolve_lcoe_fixed",
        ),
        QuantityPlotData(
            qty_name=(
                "NGCC + CCS Levelized Variable Operation and Maintenance Cost"
            ),
            qty_unit="\\$/MWh",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "variable_lcoe"),
            fname="ngcc_ccs_postsolve_lcoe_variable",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Levelized $\\text{CO}_2$ Transportation Cost",
            qty_unit="\\$/MWh",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "transport_lcoe"),
            fname="ngcc_ccs_postsolve_lcoe_transport",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Capital Cost",
            qty_unit="MM\\$/yr",
            multiplier=1,
            col_to_plot=("ngcc_ccs_costing_output", "capital_cost"),
            fname="ngcc_ccs_postsolve_capital_cost",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Fixed Operation and Maintenance Cost",
            qty_unit="MM\\$/yr",
            multiplier=1,
            col_to_plot=(
                "ngcc_ccs_costing_output", "ngcccap_total_fixed_OM_cost"
            ),
            fname="ngcc_ccs_postsolve_fixed_om_cost",
        ),
        QuantityPlotData(
            qty_name="NGCC + CCS Fixed Operation and Maintenance Cost",
            qty_unit="MM\\$/yr",
            multiplier=1,
            col_to_plot=(
                "ngcc_ccs_costing_output", "ngcccap_total_variable_OM_cost"
            ),
            fname="ngcc_ccs_postsolve_variable_om_cost",
        ),
    ]

    ccs_aic_stacked_label_col_dict = {
        "absorber_packing": (
            "absorber packing", ("ccs_aic_terms", "absorber_packing")
        ),
        "absorber_column": (
            "absorber column", ("ccs_aic_terms", "absorber_column")
        ),
        "stripper_packing": (
            "stripper packing", ("ccs_aic_terms", "stripper_packing")
        ),
        "stripper_column": (
            "stripper column", ("ccs_aic_terms", "stripper_column"),
        ),
        "stripper_reboiler": (
            "reboiler", ("ccs_aic_terms", "stripper_reboiler")
        ),
        "stripper_condenser": (
            "condenser", ("ccs_aic_terms", "stripper_condenser")
        ),
        "total_solvent_purchased": (
            "solvent", ("ccs_aic_terms", "total_solvent_purchased"),
        ),
    }

    ccs_costing_stacked_plot_data_list = [
        StackedPlotData(
            fname="ccs_tac",
            cols_to_stack=[
                ("ccs_costing", "fs.costing.CCS_AIC"),
                ("ccs_costing", "fs.costing.CCS_AOC"),
            ],
            other_cols_to_stack=[],
            qty_name="Annualized Cost",
            qty_units="MM\\$/yr",
            label_names=["investment", "operating"],
            col_multipliers=[1e-6, 1e-6],
        ),
        StackedPlotData(
            fname="ccs_aic_terms",
            cols_to_stack=[
                col for _, col in ccs_aic_stacked_label_col_dict.values()
            ],
            other_cols_to_stack=[
                ("ccs_aic_terms", col)
                for col in res_df.ccs_aic_terms.columns
                if col not in ccs_aic_stacked_label_col_dict.keys()
            ],
            qty_name="Annualized Investment Cost",
            qty_units="MM\\$/yr",
            label_names=[
                label for label, _ in ccs_aic_stacked_label_col_dict.values()
            ],
            col_multipliers=[1e-6] * res_df.ccs_aic_terms.columns.size,
        ),
        StackedPlotData(
            fname="ccs_aoc_terms",
            cols_to_stack=[
                ("ccs_aoc_terms", "reboiler"),
            ],
            other_cols_to_stack=[
                ("ccs_aoc_terms", "condenser"),
                ("ccs_aoc_terms", "H2O_makeup"),
                ("ccs_aoc_terms", "rich_solvent_pump"),
            ],
            qty_name="Annual Operating Cost",
            qty_units="MM\\$/yr",
            label_names=["steam utility"],
            col_multipliers=[1e-6] * 4,
        ),
    ]
    ccs_costing_grouped_plot_data_list = [
        GroupedPlotDataContainer(
            fname="ccs_annualized_costs",
            qty_name="Annualized Cost",
            qty_unit="MM\\$/yr",
            group_data_list=[
                GroupedPlotData(
                    group_name=f"{cost_type}",
                    multiplier=1e-6,
                    col_to_group=(
                        "ccs_costing",
                        f"fs.costing.CCS_A{cost_type[0].upper()}C"
                    ),
                )
                for cost_type in ["investment", "operating"]
            ],
            legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        ),
        GroupedPlotDataContainer(
            fname="solvent_flow_epigraph",
            qty_name="Maximum Volumetric Flow Rate",
            qty_unit="$\\text{m}^3/\\text{hr}$",
            group_data_list=[
                GroupedPlotData(
                    group_name=f"{solvent_type} solvent",
                    multiplier=1,
                    col_to_group=(
                        "design_epigraph_vars",
                        f"{solvent_type}_solvent_flow_vol",
                    ),
                )
                for solvent_type in ["rich", "lean"]
            ],
            legend_kwargs=dict(loc="lower left", bbox_to_anchor=(0, -0.7)),
        ),
    ]

    return process_solve_results_df(
        res_df=res_df,
        outdir=os.path.join(indir, "solve_results_analysis"),
        capture_target_sens_plot_data_list=(
            default_capture_target_sens_plot_data_list
            + ccs_costing_plot_data_list
            + ccs_epigraph_plot_data_list
            + ngcc_ccs_costing_plot_data_list
        ),
        conf_lvl_sens_plot_data_list=(
            default_conf_lvl_sens_plot_data_list
            + ccs_epigraph_plot_data_list
        ),
        stacked_conf_lvl_sens_plot_data_list=(
            default_stacked_conf_lvl_sens_plot_data_list
            + ccs_costing_stacked_plot_data_list
        ),
        grouped_conf_lvl_sens_plot_data_list=(
            default_grouped_conf_lvl_sens_plot_data_list
            + ccs_costing_grouped_plot_data_list
        ),
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )


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
    Process all heatmap results.
    """
    from combined_flowsheet_pyros_base import (
        default_heatmap_plot_qty_info_list,
        HeatmapQuantityInfo,
        process_all_heatmap_results as _process_all_heatmap_results,
    )

    ccs_costing_heatmap_plot_qty_info_list = [
        HeatmapQuantityInfo(
            qty_expr_col="fs.costing.CCS_TAC",
            qty_mult=1e-6,
            qty_yax_str="Total Annualized Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_tac",
            solve_results_col=("ccs_costing", "fs.costing.CCS_TAC"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.levelized_cost_of_target_capture",
            qty_mult=1,
            qty_yax_str="Levelized Cost of Target Capture",
            qty_yax_unit="\\$/metric ton $\\text{CO}_2$",
            fname="ccs_lcotc",
            solve_results_col=(
                "ccs_costing", "fs.levelized_cost_of_target_capture"
            ),
            solve_results_mult=1,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.costing.CCS_TAC_perCO2captured",
            qty_mult=1,
            qty_yax_str="Levelized Cost of Capture",
            qty_yax_unit="\\$/metric ton $\\text{CO}_2$",
            fname="ccs_tac_perco2captured",
            solve_results_col=(
                "ccs_costing", "fs.costing.CCS_TAC_perCO2captured"
            ),
            solve_results_mult=1,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.costing.CCS_AIC",
            qty_mult=1e-6,
            qty_yax_str="Annualized Investment Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_aic",
            solve_results_col=("ccs_costing", "fs.costing.CCS_AIC"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.costing.CCS_AOC",
            qty_mult=1e-6,
            qty_yax_str="Annual Operating Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_aoc",
            solve_results_col=("ccs_costing", "fs.costing.CCS_AOC"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.ccs_aoc_terms[rich_solvent_pump]",
            qty_mult=1e-6,
            qty_yax_str="Rich Pump Utility Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_uc_rich_solvent_pump",
            solve_results_col=("ccs_aoc_terms", "rich_solvent_pump"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.ccs_aoc_terms[lean_solvent_pump]",
            qty_mult=1e-6,
            qty_yax_str="Lean Pump Utility Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_uc_lean_solvent_pump",
            solve_results_col=("ccs_aoc_terms", "lean_solvent_pump"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.ccs_aoc_terms[reboiler]",
            qty_mult=1e-6,
            qty_yax_str="Reboiler Utility Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_uc_reboiler",
            solve_results_col=("ccs_aoc_terms", "reboiler"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.ccs_aoc_terms[condenser]",
            qty_mult=1e-6,
            qty_yax_str="Condenser Utility Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_uc_condenser",
            solve_results_col=("ccs_aoc_terms", "condenser"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.ccs_aoc_terms[H2O_makeup]",
            qty_mult=1e-6,
            qty_yax_str="$\\text{H}_2\\text{O}$ Makeup Utility Cost",
            qty_yax_unit="MM \\$/yr",
            fname="ccs_uc_h2o_makeup",
            solve_results_col=("ccs_aoc_terms", "H2O_makeup"),
            solve_results_mult=1e-6,
        ),
        HeatmapQuantityInfo(
            qty_expr_col="fs.SRD[0.0]",
            qty_mult=1,
            qty_yax_str="Specific Reboiler Duty",
            qty_yax_unit="MJ/kg\\,$\\text{CO}_2$",
            fname="ccs_specific_reboiler_duty",
            solve_results_col=("specific_reboiler_duty",) * 2,
            solve_results_mult=1,
        ),
    ]

    return _process_all_heatmap_results(
        results_dir=results_dir,
        outdir=outdir,
        cov_mat_infile=cov_mat_infile,
        pyros_dr_order=pyros_dr_order,
        include_deterministic_results=include_deterministic_results,
        include_pyros_results=include_pyros_results,
        include_nominal_point_in_analysis=include_nominal_point_in_analysis,
        export_summary=export_summary,
        heatmap_plot_qty_info_list=(
            default_heatmap_plot_qty_info_list
            + ccs_costing_heatmap_plot_qty_info_list
        ),
        label_conf_lvls_as_stdevs=label_conf_lvls_as_stdevs,
        output_plot_fmt=output_plot_fmt,
    )


def create_workflow_argument_parser():
    """
    Create argument parser for command line invocations of this
    script.
    """
    parser = argparse.ArgumentParser(
        prog="PyROS MEA Flowsheet",
        description=(
            "Optimization workflows for CO2 capture process using "
            "MEA scrubbing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Ensure that GAMS and IPOPT executables are installed on "
            "your system path."
        ),
    )

    def float_or_str(obj):
        try:
            return float(obj)
        except ValueError:
            return str(obj)

    def eval_bool(obj):
        if obj not in {"True", "False"}:
            raise ValueError(f"{obj!r} is not a valid boolean string.")
        return obj == "True"

    parser.add_argument(
        "--workflow",
        required=True,
        choices=[
            "deterministic",
            "deterministic_heatmaps",
            "pyros",
            "solve_results",
            "deterministic_heatmap_results",
            "pyros_heatmap_results",
            "all_heatmap_results",
        ],
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help=(
            "Path for directory "
            "to/from which workflow inputs/outputs are to be read/written. "
            "If the directory is to be used for output, and does not "
            "currently exist, then it is created. "
            "Regardless of the workflow, reserved nonexisting subdirectories "
            "are also created even if the directory already exists. "
        ),
    )
    parser.add_argument(
        "--co2-capture-targets",
        nargs="+",
        default=None,
        type=float_or_str,
        help=(
            "CO2 capture targets (minimum CO2 capture rates) "
            "for which to perform workflow."
        ),
    )
    parser.add_argument(
        "--load-from-json",
        default=None,
        help=(
            "Path from which to read initial square flowsheet model "
            "solution. If no path is provided, then the model is initialized "
            "automatically."
        ),
        type=str,
    )
    parser.add_argument(
        "--save-to-json",
        default=None,
        help=(
            "Path to a 'json.gz' file to which to write initial "
            "square flowsheet model solution. If no path is provided, "
            "then the solution is not written to any such file."
        ),
        type=str,
    )
    parser.add_argument(
        "--with-postsolve-costing",
        type=eval_bool,
        default=False,
        choices=[True, False],
        help=(
            "True to include postsolve IDAES NGCC costing in chosen "
            "optimization workflow, False otherwise."
        ),
    )
    parser.add_argument(
        "--uncertainty-cov-mat-infile",
        type=str,
        default="cov_matrix_k_eq_He_05022022.csv",
        help=(
            "Input file for covariance matrix of 6D ellipsoidal "
            "uncertainty set."
        ),
    )
    parser.add_argument(
        "--heatmap-samples-infile",
        default=None,
        type=str,
        help=(
            "Path from which to read sample points for feasibility "
            "assessment workflows. If no path is provided, then the "
            "samples are pseudorandomly generated. The seed "
            "for the random number generator has been hardcoded."
        ),
    )
    parser.add_argument(
        "--include-nompt-in-heatmap-analysis",
        default=False,
        type=eval_bool,
        choices=[True, False],
        help=(
            "True to include nominal point in heatmap postprocessing summary "
            "statistics evaluations, False otherwise."
        ),
    )

    # PyROS arguments
    parser.add_argument(
        "--pyros-confidence-levels",
        nargs="+",
        default=[90.0, 95.0, 99.0],
        type=float,
        help=(
            "If workflow 'pyros' is chosen, then "
            "the confidence levels for which to solve RO problem for "
            "each entry of option `co2_capture_targets`."
        ),
    )
    parser.add_argument(
        "--pyros-dr-order",
        type=int,
        default=0,
        help=(
            "If workflow 'pyros' is chosen, the decision rule order "
            "with which to solve all RO problems. "
            "If workflow 'solve_results' or '*_heatmap_results' "
            "is chosen, PyROS solutions obtained with only this decision rule "
            "order are considered."
        ),
    )
    parser.add_argument(
        "--pyros-tolerance",
        type=float,
        default=1e-3,
        help=(
            "If workflow 'pyros' is chosen, "
            "the PyROS solver robust feasibility tolerance."
        ),
    )
    parser.add_argument(
        "--include-pyros-heatmaps",
        type=eval_bool,
        default=False,
        choices=[True, False],
        help=(
            "If workflow 'pyros' is chosen, True to invoke heatmap workflows "
            "(with samples of the 99 percent confidence ellipsoid) on "
            "PyROS solutions with acceptable termination statuses, "
            "False otherwise."
        )
    )
    parser.add_argument(
        "--include-detailed-stream-results",
        type=eval_bool,
        default=False,
        choices=[True, False],
        help=(
            "True to include full data "
            "(molar flows, mole fractions, temperatures, pressures) "
            "for all streams "
            "in the serialized flowsheet results "
            "(for deterministic and PyROS workflows), "
            " False otherwise."
        )
    )
    parser.add_argument(
        "--export-nonoptimal-heatmap-models",
        type=eval_bool,
        default=False,
        choices=[True, False],
        help=(
            "True to export models not solved to acceptable "
            "status in heatmap workflows, False otherwise. "
            "NOTE: The exported files may be large."
        )
    )

    parser.add_argument(
        "--label-conf-lvls-as-stdevs",
        default=False,
        type=eval_bool,
        choices=[True, False],
        help=(
            "True to use standard deviation counts in lieu of "
            "confidence levels in plot labels/ticks, False otherwise."
        ),
    )
    parser.add_argument(
        "--output-plot-fmt",
        default="png",
        type=str,
        choices=["png", "pdf"],
        help=(
            "File format for plots generated by the 'solve_results'"
            "and '*_heatmap_results' workflows. "
        ),
    )

    return parser


def get_main_solver_configs():
    return {
        "gams_conopt4": (
            "gams", {
                "solver": "conopt4",
                "add_options": [
                    "option threads=1;",
                    "option reslim=300;",
                ],
            },
        ),
        "gams_conopt3": (
           "gams", {
               "solver": "conopt3",
               "add_options": ["option reslim=300;"],
           },
        ),
        "gams_knitro": (
            "gams", {
                "solver": "knitro",
                "add_options": [
                    "GAMS_MODEL.optfile=1;",
                    "$onecho > knitro.opt",
                    "linsolver 7",
                    "maxit 100",
                    "$offecho",
                ],
            }
        ),
    }


def get_extended_solver_configs():
    return {
        "gams_conopt4": (
            "gams", {
                "solver": "conopt4",
                "add_options": [
                    "option threads=1;",
                    "option reslim=300;",
                ],
            },
        ),
        "gams_conopt3": (
           "gams", {
               "solver": "conopt3",
               "add_options": ["option reslim=300;"],
           },
        ),
        "gams_conopt4_high_tol": (
            "gams", {
                "solver": "conopt4",
                "add_options": [
                    "option reslim = 3.6e3;",
                    "option threads=1;",
                    "GAMS_MODEL.optfile=1;",
                    "$onecho > conopt4.opt",
                    "Tol_Feas_Max 1e-5",
                    "$offecho",
                ],
            },
        ),
        "gams_knitro": (
            "gams", {
                "solver": "knitro",
                "add_options": [
                    "option reslim = 300;",
                    "GAMS_MODEL.optfile=1;",
                    "$onecho > knitro.opt",
                    "algorithm 3",
                    "linsolver 5",
                    # level tolerance with CONOPT default max
                    "infeastol 1e-7",
                    "maxit 1000",
                    "$offecho",
                ],
            }
        ),
        "default_ipopt": (default_solver, default_solver_options),
    }


def main():
    parser = create_workflow_argument_parser()
    args = parser.parse_args()

    main_workflow_dir = args.results_dir

    subdir_names = [
        "deterministic",
        "deterministic_heatmaps",
        "pyros",
        "pyros_heatmaps",
        "logs",
    ]
    if not os.path.exists(main_workflow_dir):
        os.mkdir(main_workflow_dir)

    workflow_subdir_dict = dict()
    for name in subdir_names:
        subdir = os.path.join(main_workflow_dir, name)
        workflow_subdir_dict[name] = subdir
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    if args.workflow == "solve_results":
        process_solve_results(
            indir=main_workflow_dir,
            label_conf_lvls_as_stdevs=args.label_conf_lvls_as_stdevs,
            output_plot_fmt=args.output_plot_fmt,
            pyros_dr_order=args.pyros_dr_order,
        )

    if args.workflow.endswith("_heatmap_results"):
        results_prefix = args.workflow.split("_heatmap_results")[0]
        process_all_heatmap_results(
            results_dir=main_workflow_dir,
            outdir=os.path.join(
                main_workflow_dir, "heatmap_analysis", results_prefix,
            ),
            include_deterministic_results=(
                results_prefix in {"deterministic", "all"}
            ),
            include_pyros_results=results_prefix in {"pyros", "all"},
            cov_mat_infile=args.uncertainty_cov_mat_infile,
            pyros_dr_order=args.pyros_dr_order,
            include_nominal_point_in_analysis=(
                args.include_nompt_in_heatmap_analysis
            ),
            label_conf_lvls_as_stdevs=args.label_conf_lvls_as_stdevs,
            output_plot_fmt=args.output_plot_fmt,
        )

    if args.workflow == "deterministic":
        # instantiate logger
        det_logger = logging.getLogger("mea_pyros")
        det_logger.setLevel(logging.DEBUG)
        det_logger.handlers.clear()

        # add handlers
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        fh = logging.FileHandler(os.path.join(
            workflow_subdir_dict["logs"],
            "deterministic_runs.log",
        ))
        det_logger.addHandler(ch)
        det_logger.addHandler(fh)

        log_script_invocation(det_logger, args)

        if args.co2_capture_targets is None:
            co2_capture_targets = [
                85.0, 87.5, 90.0, 92.5, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0
            ]
        else:
            co2_capture_targets = args.co2_capture_targets

        generate_deterministic_results_spreadsheet(
            progress_logger=det_logger,
            co2_capture_targets=co2_capture_targets,
            outdir=workflow_subdir_dict["deterministic"],
            keep_state_var_bounds=True,
            include_plots=True,
            include_presolve=False,
            tee=True,
            solver="gams",
            solver_options={"solver": "conopt4"},
            load_from_json=args.load_from_json,
            save_to_json=args.save_to_json,
            include_detailed_stream_results=(
                args.include_detailed_stream_results
            ),
            flowsheet_model_data_type=FlowsheetEconomicOptModelData,
            flowsheet_process_block_constructor=MEACombinedFlowsheet,
            flowsheet_model_data_kwargs=dict(
                with_costing=args.with_postsolve_costing,
            ),
        )

    if args.workflow == "deterministic_heatmaps":
        heatmap_logger = logging.getLogger("pyros_mea")
        heatmap_logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        fh = logging.FileHandler(
            os.path.join(
                workflow_subdir_dict["logs"],
                f"deterministic_heatmaps_"
                f"{float_to_str(args.co2_capture_targets[0])}capture.log"
            )
        )
        heatmap_logger.addHandler(ch)
        heatmap_logger.addHandler(fh)

        log_script_invocation(heatmap_logger, args)

        heatmap_workflow(
            capture_targets={
                targ: [targ] for targ in args.co2_capture_targets
            },
            outdir=workflow_subdir_dict["deterministic_heatmaps"],
            samples_infile=args.heatmap_samples_infile,
            cov_mat_infile=args.uncertainty_cov_mat_infile,
            solvers_and_options=list(get_main_solver_configs().values()),
            logger=heatmap_logger,
            tee=False,
            load_from_json=args.load_from_json,
            save_to_json=args.save_to_json,
            flowsheet_process_block_constructor=MEACombinedFlowsheet,
            flowsheet_model_data_type=FlowsheetEconomicOptModelData,
            flowsheet_model_data_kwargs=dict(
                with_costing=args.with_postsolve_costing,
            ),
            export_nonoptimal_models=args.export_nonoptimal_heatmap_models,
        )

    if args.workflow == "pyros":
        # set up PyROS progress logger
        pyros_logger = logging.getLogger("mea_flowsheet_pyros")
        pyros_logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        pyros_logger.addHandler(ch)

        local_solvers = [
            get_solver(name, options)
            for name, options in get_main_solver_configs().values()
        ]

        # use metasolver as the PyROS local solver so that
        # backup solvers are invoked on all subproblems as needed
        local_metasolver = SolverWithBackup(
            *local_solvers,
            logger=pyros_logger,
        )

        for co2_capture_target in args.co2_capture_targets:
            for conf_lvl in args.pyros_confidence_levels:
                # update logfile
                logfile_name = os.path.join(
                    workflow_subdir_dict["logs"],
                    (
                        "pyros"
                        f"_{get_co2_capture_str(None, co2_capture_target)}"
                        f"_{float_to_str(conf_lvl)}-conf"
                        f"_dr{args.pyros_dr_order}_log.log"
                    ),
                )
                fh = logging.FileHandler(os.path.join(logfile_name))
                pyros_logger.addHandler(fh)

                log_script_invocation(pyros_logger, args)

                pyros_workflow_dir = os.path.join(
                    workflow_subdir_dict["pyros"],
                    (
                        f"{float_to_str(co2_capture_target)}"
                        f"capture_{float_to_str(conf_lvl)}_conf"
                    ),
                )
                pyros_subproblem_dir = os.path.join(
                    pyros_workflow_dir,
                    "subproblems",
                )
                if not os.path.exists(pyros_workflow_dir):
                    os.mkdir(pyros_workflow_dir)
                if not os.path.exists(pyros_subproblem_dir):
                    os.mkdir(pyros_subproblem_dir)

                try:
                    pyros_workflow(
                        co2_capture_target=co2_capture_target,
                        confidence_level=conf_lvl,
                        cov_mat_infile=args.uncertainty_cov_mat_infile,
                        include_plots=True,
                        progress_logger=pyros_logger,
                        outdir=pyros_workflow_dir,
                        capture_targets_for_heatmaps=(
                            [co2_capture_target]
                            if args.include_pyros_heatmaps
                            else None
                        ),
                        heatmap_samples_infile=args.heatmap_samples_infile,
                        heatmap_solvers=local_solvers,
                        heatmap_confidence_level=99.0,
                        heatmap_output_dir=workflow_subdir_dict[
                            "pyros_heatmaps"
                        ],
                        heatmap_tee=False,
                        export_nonoptimal_heatmap_models=(
                            args.export_nonoptimal_heatmap_models
                        ),
                        include_deterministic_solution_heatmap=False,
                        include_detailed_stream_results=(
                            args.include_detailed_stream_results
                        ),
                        deterministic_build_and_solve_kwargs=dict(
                            keep_state_var_bounds=True,
                            solver="gams",
                            solver_options={
                                "solver": "conopt4",
                                "add_options": ["option reslim=3.6e3"],
                            },
                            tee=True,
                            load_from_json=args.load_from_json,
                            save_to_json=args.save_to_json,
                            flowsheet_process_block_constructor=(
                                MEACombinedFlowsheet
                            ),
                        ),
                        pyros_options=dict(
                            local_solver=local_metasolver,
                            global_solver=local_metasolver,
                            decision_rule_order=args.pyros_dr_order,
                            robust_feasibility_tolerance=args.pyros_tolerance,
                            solve_master_globally=False,
                            bypass_global_separation=True,
                            objective_focus=pyros.ObjectiveType.nominal,
                            symbolic_solver_labels=True,
                            tee=True,
                            keepfiles=True,
                            subproblem_file_directory=pyros_subproblem_dir,
                            subproblem_format_options={
                                "gams": {"symbolic_solver_labels": True},
                            },
                        ),
                        prioritize_absorber_ub_flood_idxs=range(1, 22),
                        prioritize_stripper_ub_flood_idxs=[1],
                        prioritize_stripper_lb_flood_idxs=[40],
                        flowsheet_model_data_type=(
                            FlowsheetEconomicOptModelData
                        ),
                        flowsheet_model_data_kwargs=dict(
                            with_costing=args.with_postsolve_costing,
                        ),
                    )
                except (
                        ApplicationError,
                        ValueError,
                        RuntimeError,
                        ZeroDivisionError,
                        ):
                    pyros_logger.exception(
                        msg="PyROS failed; see exception",
                    )
                pyros_logger.info("/" * 100)
                pyros_logger.removeHandler(fh)


if __name__ == "__main__":
    main()
