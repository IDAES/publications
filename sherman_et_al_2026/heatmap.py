"""
Methods for evaluating robust feasibilility of the design
variables of a model subject to parametric uncertainty
and for visualizing the results.
"""


from collections import namedtuple
import itertools
import logging
import os
import textwrap

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from confidence_ellipsoid import (
    calc_conf_lvl,
    gaussian_pdf,
    ellipsoid_hypervolume,
    get_conf_lvl_plot_label,
)

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.timing import TicTocTimer
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr import identify_variables
import pyomo.contrib.pyros as pyros
import pyomo.environ as pyo
from pyomo.opt import SolverResults, TerminationCondition

from model_utils import SolverWithBackup, get_co2_capture_str, float_to_str
from confidence_ellipsoid import (
    ellipsoid_probability,
    mag_factor,
)
from plotting_utils import (
    DEFAULT_MPL_RC_PARAMS,
    AX_LABEL_TEXTWIDTH,
    set_nonoverlapping_fixed_xticks,
    wrap_quantity_str,
    set_lightness,
)


default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)
default_logger.addHandler(logging.StreamHandler())

DEFAULT_ACCEPTABLE_TERMINATIONS = {
    TerminationCondition.optimal,
    TerminationCondition.locallyOptimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.feasible,
}


def initialize_model(
        model,
        first_stage_variables,
        uncertain_params,
        solver,
        ):
    """
    Initialize model by solving slack variable feasibility
    problem.
    """
    from pyomo.core import TransformationFactory
    tol = 1e-4

    slack_model = model.clone()
    violated_cons = []
    for con in slack_model.component_data_objects(pyo.Constraint, active=True):
        if con.lslack() < -tol or con.uslack() < - tol:
            violated_cons.append(con)

    if violated_cons:
        TransformationFactory("core.add_slack_variables").apply_to(
            model=slack_model,
            targets=violated_cons,
        )
        res = solver.solve(
            slack_model,
            load_solutions=False,
        )
        if pyo.check_optimal_termination(res):
            slack_model.solutions.load_from(res)
            new_initial_point_map = ComponentMap()
            for var in model.component_data_objects(pyo.Var):
                new_initial_point_map[var] = (
                    slack_model.find_component(var).value
                )
            return new_initial_point_map


def load_solution(solver, model, results):
    """
    Load solution to a model, using, if present,
    the `load_solution` method of the provided solver.

    Parameters
    ----------
    solver : Pyomo solver
        Optimizer used to obtain `results`.
    model : ConcreteModel
        Model to which to load solution.
    results : pyomo.opt.results.results_.SolverResults
        Results object containing solution to be loaded.
    """
    if hasattr(solver, "load_solution"):
        solver.load_solution(model, results)
    else:
        model.solutions.load_from(results)


def evaluate_design_heatmap(
        model,
        first_stage_variables,
        uncertain_params,
        solver,
        uncertain_parameter_samples,
        cov_mat,
        model_solutions,
        initialization_func=None,
        initialization_func_kwargs=None,
        nominal_parameter_values=None,
        output_dir=None,
        backup_solvers=None,
        progress_logger=None,
        fixed_con_feas_tol=1e-4,
        exprs_to_evaluate=None,
        tee=False,
        export_nonoptimal_models=False,
        ):
    """
    Evaluate robust feasibility (i.e. heatmap) of a design
    for a model with ellipsoidal parametric uncertainty.

    Parameters
    ----------
    model : ConcreteModel
        Model of interest.
    first_stage_variables : list of VarData
        First-stage (design) variables.
    uncertain_params : (N,) list of {VarData, ParamData}
        Uncertain model parameters.
        All entries of type VarData should have attribute
        `fixed=True`.
    solver : Pyomo solver type
        Main optimizer for model instances.
    uncertain_parameter_samples : (M, N) array_like
        Uncertain parameter samples for which to check feasibility.
    cov_mat : (N, N) array_like or None
        Covariance matrix for ellipsoid of parametric uncertainty.
        This is used solely for evaluating sample probability
        masses/densities for each sample.
        If `None` is passed, then all
        samples are assigned a probability weight of 1.
        Otherwise, each sample point is assigned the value
        of the multivariate normal PDF at that point.
    model_solutions : dict of ComponentMap
        Mapping from strings giving name/identifier of solutions
        to evaluate to `ComponentMap` objects specifying the solution
        variable values. Each `ComponentMap` should at least contain
        entries for the first-stage variables of `model`.
    nominal_parameter_values : (N,) array_like or None, optional
        Nominal uncertain parameter values. If `None` is
        passed, then these are obtained by evaluating each
        param object in `uncertain_params` at the current state
        of `model`.
    output_dir : path-like or None, optional
        Directory to which to output results.
        If `None` is passed, then results are not serialized.
        Otherwise, results are serialized to directory with
        the following subdirectory structure:

        - dataframes/ : detailed optimization results (.csv files)
        - results/ : summary of the results for each solution
          in `model_solutions` (.csv files)

    backup_solvers : list of Pyomo solver type, optional
        Backup optimizers, used in the event of failure of
        `solver` for a given problem instance.
    progress_logger : logging.Logger, optional
        Progress logger.
    fixed_con_feas_tol : float, optional
        Absolute tolerance for checking whether constraints
        with fixed or first-stage variables are violated.
    exprs_to_evaluate : dict or None, optional
        Expressions to evaluate after each control
        problem is solved.
        If a dict is passed, then each entry should map a str
        to an expression in components of `model`.
        If `None` is passed, then an empty dict is used.

    Returns
    -------
    heatmap_res_list : list of dict
        Summary of evaluation results for each solution in
        `model_solutions`.
    heatmap_res_list : list of dict
        Detailed evaluation results for each solution in
        `model_solutions`.
    """
    # standardize progress_logger
    if progress_logger is None:
        progress_logger = default_logger

    # log progress using TicTocTimer instead of logger directly,
    # to get sense of time required in output logs
    tt_timer = TicTocTimer(logger=progress_logger)
    tt_timer.tic(msg="Starting heatmap")

    # create output dir, if specified
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=False)

        # subdirectories
        df_subdir = os.path.join(output_dir, "dataframes")
        os.mkdir(df_subdir)
        res_subdir = os.path.join(output_dir, "results")
        os.mkdir(res_subdir)
        if export_nonoptimal_models:
            model_subdir = os.path.join(output_dir, "models")
            os.mkdir(model_subdir)

    # assemble iterable of solvers (including backup solvers)
    if backup_solvers is None:
        all_solvers = [solver]
    else:
        all_solvers = [solver] + list(backup_solvers)
    heatmap_solver = SolverWithBackup(*all_solvers, logger=progress_logger)

    # satisfactory termination conditions.
    # useful for determining whether to use backup solver(s)
    # for given problem instance
    acceptable_terminations = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
        TerminationCondition.feasible,
    }

    # type check uncertain params
    uncertain_params = list(uncertain_params)
    for param in uncertain_params:
        if isinstance(param, VarData):
            if not param.fixed:
                raise ValueError(
                    f"Uncertain parameter with name {param.name!r} "
                    f"of type {VarData} is unfixed (`fixed=False`). "
                    "Ensure all VarData-type entries of `uncertain_params` "
                    "are such that `fixed=True`."
                )
        elif not isinstance(param, ParamData):
            raise TypeError(
                f"Uncertain parameter entry {param!r} is of type "
                f"{type(param).__name__!r}, but should be of type "
                "VarData or ParamData."
            )

    # standardize nominal parameter values
    if nominal_parameter_values is None:
        nominal_parameter_values = np.array(
            [pyo.value(param) for param in uncertain_params]
        )
    else:
        nominal_parameter_values = np.array(nominal_parameter_values)

    # determine index of nominal parameter realization
    # among sequence of sampled points, if present
    isclose_arr = np.array([
        np.allclose(nominal_parameter_values, sample)
        for sample in uncertain_parameter_samples
    ])
    if isclose_arr.size > 0:
        nom_sample_idx = np.arange(isclose_arr.size)[isclose_arr][0]
    else:
        nom_sample_idx = None

    exprs_to_evaluate = (
        dict() if exprs_to_evaluate is None else exprs_to_evaluate
    )
    exprs_to_eval = pyo.Expression(
        list(exprs_to_evaluate.keys()),
        initialize=exprs_to_evaluate,
    )
    exprs_to_eval_attr_name = unique_component_name(
        model,
        "exprs_to_eval",
    )
    model.add_component(exprs_to_eval_attr_name, exprs_to_eval)

    # column names for detailed results DataFrame
    expr_col_names = [
        f"Value({expr_name})" for expr_name in exprs_to_evaluate
    ]

    # cast uncertain parameter samples to array
    uncertain_parameter_samples = np.array(uncertain_parameter_samples)

    # copy model; fix first-stage variables
    mdl = model.clone()
    for first_stg_var in first_stage_variables:
        mdl.find_component(first_stg_var).fix()

    # keep original state of model variables
    orig_var_val_map = ComponentMap()
    for var in mdl.component_data_objects(pyo.Var):
        orig_var_val_map[var] = pyo.value(var, exception=False)

    # assemble list of constraints containing only fixed vars
    # (e.g. constraints with first-stage variables only)
    # and deactivate all such constraints
    activated_cons_with_fixed_vars = [
        con for con in
        mdl.component_data_objects(pyo.Constraint, active=True)
        if all(var.fixed for var in identify_variables(con.body))
    ]
    for con in activated_cons_with_fixed_vars:
        # we will check these constraints are satisfied ourselves,
        # instead of passing them to the solver,
        # as the presolvers for some solvers have zero
        # tolerance for violation of these constraints
        # and will raise Exceptions
        con.deactivate()

    if initialization_func_kwargs is None:
        initialization_func_kwargs = dict()

    heatmap_res_list = []
    heatmap_df_list = []
    for sol_name, sol_map in model_solutions.items():
        # initialize dataframe for recording results
        heatmap_df = pd.DataFrame(
            index=[idx for idx in range(uncertain_parameter_samples.shape[0])],
            columns=[
                "Parameter Values",
                "Weight",
                "Termination Condition",
                "Solve Time (s)",
                "Solver Message",
            ] + expr_col_names,
        )

        # generate paths for exporting detailed and summary results
        if output_dir is not None:
            output_df_path = os.path.join(
                df_subdir,
                f"results_{sol_name}.csv",
            )
            output_res_path = os.path.join(
                res_subdir,
                f"results_{sol_name}.csv",
            )

        # evaluate design for this solution at the sampled
        # parameter points
        for pt_idx, pt in enumerate(uncertain_parameter_samples):
            # might hamper performance, but done to prevent
            # any issues incurred from loading solutions
            # across iterations over the parameter samples
            active_model = mdl.clone()

            # to ensure initial point is not dependent on
            # solutions loaded from prior iterations
            for var, val in orig_var_val_map.items():
                active_model.find_component(var).set_value(val)

            # load solution to model
            # note: first-stage variables have already been fixed
            for sol_var, val in sol_map.items():
                active_model.find_component(sol_var).set_value(val)

            tt_timer.toc(
                msg=(
                    "Evaluating design at point "
                    f"{pt} ({pt_idx + 1} of "
                    f"{len(uncertain_parameter_samples)})..."
                ),
                level=logging.INFO,
                delta=False,
            )

            active_model_uncertain_params = [
                active_model.find_component(param)
                for param in uncertain_params
            ]
            for param, val in zip(active_model_uncertain_params, pt):
                if isinstance(param, ParamData):
                    param.set_value(val)
                else:
                    param.set_value(val * pyo.units.get_units(param))

            if initialization_func is not None:
                tt_timer.toc(
                    "Initializing model...",
                    delta=False,
                    level=logging.DEBUG,
                )
                initialization_clone = active_model.clone()
                initial_pt_map = initialization_func(
                    model=initialization_clone,
                    first_stage_variables=[
                        initialization_clone.find_component(var)
                        for var in first_stage_variables
                    ],
                    uncertain_params=[
                        initialization_clone.find_component(param)
                        for param in uncertain_params
                    ],
                    solver=solver,
                    **initialization_func_kwargs,
                )

                # all first-stage variables are included in this set,
                # since the first-stage variables were fixed
                # prior to cloning
                active_model_fixed_var_set = ComponentSet(
                    active_model.find_component(var)
                    for var in active_model.component_data_objects(pyo.Var)
                    if var.fixed
                )

                if initial_pt_map is not None:
                    for initvar, val in initial_pt_map.items():
                        active_var = active_model.find_component(initvar)
                        # ensure fixed variables are not updated,
                        # as we consider fixed variables to be effectively
                        # first-stage (regardless of whether user included
                        # them in first-stage variables list or not).
                        if active_var not in active_model_fixed_var_set:
                            active_var.set_value(val)

            # check constraints with fixed variables.
            # if any of them violated, then problem is infeasible.
            # set up results object with infeasible termination status.
            # should help circumvent issues with GAMS interfaces.
            infeasible_fixed_var_cons = [
                active_model.find_component(con)
                for con in activated_cons_with_fixed_vars
                if con.lslack() < -fixed_con_feas_tol
                or con.uslack() < -fixed_con_feas_tol
            ]
            if infeasible_fixed_var_cons:
                tt_timer.toc(
                    "Constraints "
                    f"{[con.name for con in infeasible_fixed_var_cons]}, "
                    "whose expressions contain only "
                    "either first-stage or fixed variables, "
                    "are not satisfied. Generating SolverResults object with "
                    "infeasible status.",
                    level=logging.DEBUG,
                    delta=False,
                )
                res = SolverResults()
                res.solver.termination_condition = (
                    TerminationCondition.infeasible
                )
                setattr(res.solver, SolverWithBackup.TOTAL_WALL_TIME_ATTR, 0)
                res.solver.message = (
                    "Constraints "
                    f"{[con.name for con in infeasible_fixed_var_cons]}, "
                    "whose expressions contain only "
                    "either first-stage or fixed variables, "
                    "are not satisfied. Generating SolverResults object with "
                    "infeasible status."
                )
            else:
                res = heatmap_solver.solve(
                    active_model,
                    tee=tee,
                    load_solutions=False,
                    symbolic_solver_labels=True,
                )

            # evaluate sample probability weight
            if cov_mat is None:
                weight = 1
            else:
                weight = ellipsoid_probability(
                    x=pt,
                    mean=nominal_parameter_values,
                    cov_mat=cov_mat,
                )

            heatmap_df.loc[
                pt_idx, [
                    "Parameter Values",
                    "Weight",
                    "Termination Condition",
                    "Solve Time (s)",
                    "Solver Message",
                ]
            ] = {
                "Parameter Values": pt,
                "Weight": weight,
                "Termination Condition": res.solver.termination_condition,
                "Solve Time (s)": getattr(
                    res.solver,
                    SolverWithBackup.TOTAL_WALL_TIME_ATTR,
                ),
                "Solver Message": res.solver.message,
            }

            # include models solved to feasible, rather than optimal,
            # status.
            # note: we serialize model before loading solution so that
            # initial point is included in the serialized file
            export_model = (
                not pyo.check_optimal_termination(res)
                and export_nonoptimal_models
            )
            if export_model:
                active_model.write(
                    os.path.join(
                        model_subdir,
                        (
                            f"sol_{sol_name}_model_{pt_idx}_"
                            f"{res.solver.termination_condition.name}.gms"
                        ),
                    ),
                    io_options=dict(symbolic_solver_labels=True),
                )

            # evaluate expressions
            term_cond = res.solver.termination_condition
            if term_cond in acceptable_terminations:
                load_solution(solver, active_model, res)
                mdl_exprs_to_eval = active_model.find_component(
                    exprs_to_eval_attr_name
                )
                expr_values = [
                    pyo.value(expr) for expr in mdl_exprs_to_eval.values()
                ]
                heatmap_df.loc[pt_idx, expr_col_names] = expr_values

            # write dataframe to file
            if output_dir is not None:
                heatmap_df.to_csv(output_df_path)

        # summarize results for this solution
        heatmap_res = {
            "Total Solve Time (s)": heatmap_df["Solve Time (s)"].sum(),
            "Total Elapsed Time": tt_timer.toc(msg=None),
            "Total num points sampled": uncertain_parameter_samples.shape[0],
        }

        # count points by termination condition
        for term_cond in TerminationCondition:
            df_with_term_cond = heatmap_df[
                heatmap_df["Termination Condition"] == term_cond
            ]
            heatmap_res[f"Points {term_cond.name}"] = {
                "count": df_with_term_cond.index.size,
                "weight": (
                    df_with_term_cond["Weight"].sum()
                    / heatmap_df["Weight"].sum()
                )
            }

        heatmap_res["First-stage Variable Values"] = {
            var.name: mdl.find_component(var).value
            for var in first_stage_variables
        }

        feasible_df = heatmap_df[
            heatmap_df["Termination Condition"].isin(acceptable_terminations)
        ]

        # take note of whether nominal uncertain parameter realization:
        # (1) present in list of sampled points
        # (2) admits feasible solution to the model
        nom_val_found = (
            nom_sample_idx is not None
            and nom_sample_idx in feasible_df.index
        )

        # evaluate expression stats
        for expr_name, col_name in zip(exprs_to_evaluate, expr_col_names):
            # get expression value under nominal realization
            nom_val = np.nan
            if nom_val_found:
                nom_val = feasible_df.loc[nom_sample_idx, col_name]

            # evaluate basic stats
            mean = feasible_df[col_name].mean()
            stdev = feasible_df[col_name].std()
            min_val = feasible_df[col_name].min()
            max_val = feasible_df[col_name].max()

            # take weighted averages
            if not feasible_df["Weight"].index.empty:
                weighted_mean = (
                    (feasible_df[col_name] * feasible_df["Weight"]).sum()
                    / feasible_df["Weight"].sum()
                )
                weighted_stdev = np.sqrt(
                    (
                        (feasible_df[col_name] - weighted_mean) ** 2
                        * feasible_df["Weight"]
                    ).sum()
                    / feasible_df["Weight"].sum()
                )
            else:
                weighted_mean = np.nan
                weighted_stdev = np.nan

            expr_stats_dict = {
                "nom_val": nom_val,
                "mean": mean,
                "stdev": stdev,
                "min": min_val,
                "max": max_val,
                "weighted_mean": weighted_mean,
                "weighted_stdev": weighted_stdev,
            }
            heatmap_res[f"Expr stats {expr_name}"] = expr_stats_dict

        # serialize summary of results
        if output_dir is not None:
            pd.Series(heatmap_res).to_csv(output_res_path)

        heatmap_res_list.append(heatmap_res)
        heatmap_df_list.append(heatmap_df)

    tt_timer.toc("Done evaluating heatmap", level=logging.INFO, delta=False)

    return heatmap_res_list, heatmap_df_list


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


def process_heatmap_results(
        heatmap_proc_data_list,
        param_cov_mat,
        feasible_termination_conditions=None,
        ignore_termination_conditions=None,
        include_nominal_point_in_analysis=False,
        ):
    """
    Process heatmap results.
    """
    from confidence_ellipsoid import (
        calc_conf_lvl,
        ellipsoid_hypervolume,
        gaussian_pdf,
        get_pyros_ellipsoidal_set,
    )

    if feasible_termination_conditions is None:
        feasible_termination_conditions = {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal,
            TerminationCondition.feasible,
        }
    if ignore_termination_conditions is None:
        ignore_termination_conditions = {
            tc for tc in TerminationCondition
            if tc not in feasible_termination_conditions
            and tc != TerminationCondition.infeasible
        }

    feasible_termination_str_set = {
        str(tc) for tc in feasible_termination_conditions
    }

    def get_pt_conf_lvl(x, mean, cov_mat):
        x = np.array(x)
        offset = x - mean
        return calc_conf_lvl(
            r=np.sqrt(
                offset[np.newaxis] @ np.linalg.inv(param_cov_mat) @ offset
            ),
            n=param_cov_mat.shape[-1],
        )

    overall_results_dict = dict()
    for heatmap_proc_data in heatmap_proc_data_list:
        heatmap_df = heatmap_proc_data.heatmap_df
        design_target = heatmap_proc_data.design_target
        design_conf_lvl = heatmap_proc_data.design_conf_lvl
        eval_target = heatmap_proc_data.eval_target
        eval_conf_lvl = heatmap_proc_data.eval_conf_lvl
        heatmap_summary = heatmap_proc_data.heatmap_summary_series

        default_logger.info(
            "Processing heatmap results for: "
            f"{design_target=}, {design_conf_lvl=}, "
            f"{eval_target=}, {eval_conf_lvl=}"
        )

        # record first-stage variable values
        first_stage_var_vals = eval(
            heatmap_summary["First-stage Variable Values"]
        )

        # ascertain samples in confidence region of interest,
        # removing samples with termination conditions to be ignored

        # note: this assumes the evaluation results for the nominal point
        # were recorded in the first row of the dataframe.
        nominal_row_idx = 0
        include_in_analysis = (
            heatmap_df.index != nominal_row_idx
            | include_nominal_point_in_analysis
        )
        param_mean = heatmap_df["Parameter Values"][nominal_row_idx]
        is_nominally_feasible = (
            heatmap_df.loc[nominal_row_idx]["Termination Condition"]
            in feasible_termination_str_set
        )
        design_conf_ellipsoid = get_pyros_ellipsoidal_set(
            mean=param_mean,
            cov_mat=param_cov_mat,
            level=design_conf_lvl / 100,
        )
        eval_conf_ellipsoid = get_pyros_ellipsoidal_set(
            mean=param_mean,
            cov_mat=param_cov_mat,
            level=eval_conf_lvl / 100,
        )
        is_in_design_ellipsoid = (
            heatmap_df["Parameter Values"].apply(
                func=lambda pt: design_conf_ellipsoid.point_in_set(pt),
            )
        )
        is_in_eval_ellipsoid = (
            heatmap_df["Parameter Values"].apply(
                func=lambda pt: eval_conf_ellipsoid.point_in_set(pt),
            )
        )
        is_feasible = heatmap_df["Termination Condition"].isin(
            feasible_termination_str_set
        )
        include_in_eval = is_in_eval_ellipsoid & include_in_analysis
        include_in_design_check = is_in_design_ellipsoid & include_in_analysis
        heatmap_df_eval_samples = heatmap_df[include_in_eval]
        if heatmap_df_eval_samples.index.size == 0:
            continue

        # importance sampling diagnostics
        # estimating gaussian expectations with uniform trial PDF
        parameter_samples = np.array(
            [sample for sample in heatmap_df_eval_samples["Parameter Values"]]
        )
        weights_arr_for_eval = heatmap_df_eval_samples["Weight"].to_numpy()
        absolute_weights = (
            (100 / eval_conf_lvl)
            * gaussian_pdf(parameter_samples, param_mean, param_cov_mat)
            * ellipsoid_hypervolume(
                param_mean, param_cov_mat, eval_conf_lvl / 100
            )
        )
        eval_diagnostic_stats = dict(
            num_samples=weights_arr_for_eval.size,
            avg_norm_weight=absolute_weights.mean(),
            max_weight_ratio=(
                weights_arr_for_eval.max() / weights_arr_for_eval.sum()
            ),
            mean_ess=(
                weights_arr_for_eval.sum() ** 2
                / (weights_arr_for_eval ** 2).sum()
            ),
            var_ess=(
                (weights_arr_for_eval ** 2).sum() ** 2
                / (weights_arr_for_eval ** 4).sum()
            ),
            skewness_ess=(
                (weights_arr_for_eval ** 2).sum() ** 3
                / (weights_arr_for_eval ** 3).sum() ** 2
            ),
        )

        is_eval_feasible = is_feasible[include_in_eval]
        is_design_feasible = is_feasible[
            include_in_eval & include_in_design_check
        ]

        # among final samples, report for feasibility estimate
        # probability, variance, and ESS
        feas_prob_avg = eval_conf_lvl * (
            weights_arr_for_eval[is_eval_feasible].sum()
            / weights_arr_for_eval.sum()
        )
        feas_prob_std = np.sqrt(
            (
                weights_arr_for_eval ** 2
                * (eval_conf_lvl * is_eval_feasible - feas_prob_avg) ** 2
            ).sum()
            / weights_arr_for_eval.sum() ** 2
        )
        feas_prob_custom_weights = (
            is_eval_feasible * weights_arr_for_eval
            / (is_eval_feasible * weights_arr_for_eval).sum()
        )
        feas_prob_ess = 1 / (feas_prob_custom_weights ** 2).sum()
        feas_prob_stats = dict(
            is_nominally_feasible=is_nominally_feasible,
            nominal_included_in_analysis=include_nominal_point_in_analysis,
            num_design_samples=include_in_design_check.sum(),
            num_eval_samples=weights_arr_for_eval.size,
            num_feas_samples=(
                weights_arr_for_eval[is_eval_feasible].size
            ),
            num_feas_design_samples=is_design_feasible.sum(),
            expected_val=feas_prob_avg,
            std=feas_prob_std,
            ess=feas_prob_ess,
        )

        # for each expression evaluated, report:
        # - importance sampling estimate of
        # - average, standard deviation, ESS
        # - among samples with non-ignored termination conditions
        expr_eval_cols = [
            col for col in heatmap_df_eval_samples.columns
            if col.startswith("Value(")
        ]
        expr_evals_res_dict = dict()
        feasible_weights_for_eval = weights_arr_for_eval[
            is_eval_feasible
        ]
        for col in expr_eval_cols:
            feasible_expr_vals = heatmap_df_eval_samples[
                is_eval_feasible
            ][col].astype(float)
            nominal_val = (
                heatmap_df.loc[nominal_row_idx, col]
                if is_nominally_feasible else np.nan
            )
            min_val = feasible_expr_vals.min()
            max_val = feasible_expr_vals.max()
            expected_val = (
                (feasible_expr_vals * feasible_weights_for_eval).sum()
                / feasible_weights_for_eval.sum()
            )
            abs_expected_over_nom_val = (
                abs(expected_val / nominal_val)
                if is_nominally_feasible and abs(nominal_val) > 0
                else np.nan
            )
            std = np.sqrt(
                (
                    feasible_weights_for_eval ** 2
                    * (feasible_expr_vals - expected_val) ** 2
                ).sum()
                / feasible_weights_for_eval.sum() ** 2
            )
            std_over_abs_expected_val = (
                std / abs(expected_val) if abs(expected_val) > 0 else np.nan
            )
            if abs(feasible_expr_vals.max()) < 1e-16:
                ess = np.nan
            else:
                custom_weights = (
                    abs(feasible_expr_vals) * feasible_weights_for_eval
                    / (
                        abs(feasible_expr_vals) * feasible_weights_for_eval
                    ).sum()
                )
                ess = 1 / (custom_weights ** 2).sum()
            abs_range = feasible_expr_vals.max() - feasible_expr_vals.min()
            range_over_nom_val = (
                (feasible_expr_vals.max() - feasible_expr_vals.min())
                / abs(nominal_val)
            ) if abs(nominal_val) > 0 else np.nan
            range_over_expected_val = (
                (feasible_expr_vals.max() - feasible_expr_vals.min())
                / abs(expected_val)
            ) if abs(expected_val) > 0 else np.nan
            expr_evals_res_dict[col] = dict(
                num_feas_samples=feasible_weights_for_eval.size,
                nominal_val=nominal_val,
                min_val=min_val,
                max_val=max_val,
                expected_val=expected_val,
                std=std,
                range=abs_range,
                ess=ess,
                abs_expected_over_nom_val=abs_expected_over_nom_val,
                std_over_abs_expected_val=std_over_abs_expected_val,
                range_over_nom_val=range_over_nom_val,
                range_over_expected_val=range_over_expected_val,
            )

        # assemble series
        sampling_diagnostics_series_idx = pd.MultiIndex.from_product([
            ["sampling_diagnostics"],
            eval_diagnostic_stats.keys(),
        ])
        feas_prob_series_idx = pd.MultiIndex.from_product([
            ["feasibility_evaluation"],
            list(feas_prob_stats.keys()),
        ])
        first_stage_var_val_series_idx = pd.MultiIndex.from_product([
            ["first_stage_var_vals"],
            first_stage_var_vals.keys(),
        ])
        expr_evals_series_idx = pd.MultiIndex.from_product([
            list(expr_evals_res_dict.keys()),
            list(next(iter(expr_evals_res_dict.values())).keys()),
        ])

        final_series_idx = pd.MultiIndex.from_tuples(
            sampling_diagnostics_series_idx.tolist()
            + feas_prob_series_idx.tolist()
            + first_stage_var_val_series_idx.tolist()
            + expr_evals_series_idx.tolist()
        )

        final_series = pd.Series(index=final_series_idx, dtype=object)
        for key, val in feas_prob_stats.items():
            final_series[("feasibility_evaluation", key)] = float(val)
        final_series.first_stage_var_vals = pd.Series(first_stage_var_vals)
        final_series.sampling_diagnostics = pd.Series(eval_diagnostic_stats)
        for colname, eval_stats_dict in expr_evals_res_dict.items():
            final_series[colname] = pd.Series(eval_stats_dict)

        overall_results_dict[(
            design_target,
            design_conf_lvl,
            eval_target,
            eval_conf_lvl,
        )] = final_series

    overall_results_df = pd.DataFrame(overall_results_dict).transpose()
    overall_results_df.index.names = [
        "design_target",
        "design_conf_lvl",
        "eval_target",
        "eval_conf_lvl",
    ]
    overall_results_df.sort_index(inplace=True)

    default_logger.info(
        "Done processing heatmap results."
    )
    return overall_results_df


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_heatmap_feasibility_target_response(
        heatmap_proc_data_list,
        heatmap_summary_df,
        outdir,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        label_out_of_sample_portions=False,
        ):
    """
    Plot response of feasibility evaluation to CO2
    capture target.
    """
    feas_eval_results = heatmap_summary_df.feasibility_evaluation

    conf_lvl_groups = feas_eval_results.groupby("design_conf_lvl")
    num_conf_lvl_groups = len(conf_lvl_groups)
    bar_width = 1 / (1.5 * num_conf_lvl_groups)

    design_target_to_pos_map = {
        target: idx
        for idx, target in enumerate(feas_eval_results.index.levels[0])
    }

    are_all_conf_lvls_int = all(
        clvl == int(clvl) for clvl, _ in conf_lvl_groups
    )

    num_samples_fig, num_samples_ax = plt.subplots(
        figsize=(
            6.2,
            2.7 + 0.1 * (
                len(conf_lvl_groups) if label_out_of_sample_portions
                else int(len(conf_lvl_groups) / 4)
            )
        ),
    )
    feas_prob_fig, feas_prob_ax = plt.subplots(figsize=(6.2, 3.5))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    det_label = "det." if label_out_of_sample_portions else "deterministic"

    for plot_mode in ["in_sample", "out_sample", "h_lines"]:
        for grp_idx, (conf_lvl, conf_subdf) in enumerate(conf_lvl_groups):
            bar_x_positions = [
                design_target_to_pos_map[targ]
                + bar_width * (grp_idx + 0.5 - num_conf_lvl_groups / 2)
                for targ in conf_subdf.index.get_level_values("design_target")
            ]
            # at most one row per design target per confidence level
            assert len(bar_x_positions) == len(set(bar_x_positions))

            plot_color = color_cycle[grp_idx % len(color_cycle)]

            bar_label = (
                det_label if conf_lvl == 0
                else get_conf_lvl_plot_label(
                    conf_lvl=conf_lvl,
                    n=6,
                    decimals=int(not (are_all_conf_lvls_int)),
                    as_mag_factor=label_conf_lvls_as_stdevs,
                )
            )
            num_feas_samples_arr = (
                conf_subdf.num_feas_samples.astype(float).values
            )
            num_feas_design_samples_arr = (
                conf_subdf.num_feas_design_samples.astype(float).values
            )

            if plot_mode == "out_sample":
                num_samples_ax.bar(
                    bar_x_positions,
                    num_feas_samples_arr - num_feas_design_samples_arr,
                    bar_width * 0.9,
                    bottom=num_feas_design_samples_arr,
                    label=(
                        f"{bar_label} (out-of-sample)"
                        if label_out_of_sample_portions else None
                    ),
                    edgecolor=plot_color,
                    color="none",
                    hatch="///////",
                    linewidth=0.6,
                    zorder=1.1,
                )
            elif plot_mode == "in_sample":
                num_samples_ax.bar(
                    bar_x_positions,
                    num_feas_design_samples_arr,
                    bar_width * 0.9,
                    label=f"{bar_label}",
                    color=plot_color,
                    edgecolor=plot_color,
                    zorder=1.2,
                )
            elif plot_mode == "h_lines":
                num_all_design_samples = conf_subdf.iloc[0].num_design_samples
                num_samples_ax.axhline(
                    num_all_design_samples,
                    color=plot_color,
                    linestyle="dashed",
                    zorder=1,
                )
                feas_prob_ax.bar(
                    bar_x_positions,
                    conf_subdf.expected_val.astype(float).values,
                    bar_width * 0.9,
                    label=bar_label,
                )
            else:
                raise ValueError

    num_eval_samples = (
        heatmap_summary_df.iloc[0].feasibility_evaluation.num_eval_samples
    )
    num_samples_ax.set_ylim((None, num_eval_samples * 1.1))

    ax_ylab_zip = [
        (
            num_samples_ax,
            "Number of Scenarios for Which Feasibility Reported",
            None,
        ),
        (
            feas_prob_ax,
            "Cumulative Gaussian Probability of Feasibility",
            "\\%",
        ),
    ]
    for ax, ylab, yunit in ax_ylab_zip:
        ax.set_xlabel(wrap_quantity_str(
            namestr="$\\text{CO}_2$ Capture Target",
            unitstr="\\%",
        ))
        ax.set_ylabel(wrap_quantity_str(
            namestr=ylab,
            unitstr=yunit,
            width=30,
        ))
        legend_bbox_y_pos = 1 + (
            0.125 * len(conf_lvl_groups) if label_out_of_sample_portions
            else 0.2 * int(len(conf_lvl_groups) / 4)
        )
        ax.legend(
            bbox_to_anchor=(0, legend_bbox_y_pos),
            loc="upper left",
            ncol=2 if label_out_of_sample_portions else 4,
        )
        ax.set_xticks(
            list(design_target_to_pos_map.values()),
            [str(targ) for targ in design_target_to_pos_map.keys()],
        )
        ax.grid(axis="x")

    num_samples_fig.savefig(
        os.path.join(outdir, f"plot_num_feas_samples.{output_plot_fmt}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    feas_prob_fig.savefig(
        os.path.join(outdir, f"plot_feas_prob.{output_plot_fmt}"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )


def plot_heatmap_feasibility_results(
        heatmap_proc_data_list,
        param_cov_mat,
        outdir=None,
        feasible_termination_conditions=None,
        ignore_termination_conditions=None,
        include_nominal_point_in_analysis=False,
        output_plot_fmt="png",
        ):
    """
    Visualize heatmap feasibility results.
    """

    if outdir is not None and not os.path.exists(outdir):
        os.mkdir(outdir)

    default_logger.info("Generating heatmap feasibility barplots...")

    for procdata in heatmap_proc_data_list:
        df = procdata.heatmap_df
        mean = df["Parameter Values"][0]
        default_logger.info(
            "Generating barplot for: "
            f"{procdata.design_target=}, {procdata.design_conf_lvl=}, "
            f"{procdata.eval_target=}, {procdata.eval_conf_lvl=}"
        )

        # exclude nominal realization from the analysis
        off_nominal_df = df[df.index != 0]
        param_values_arr = np.stack(
            off_nominal_df["Parameter Values"].to_numpy()
        )
        mahalanobis_distances = np.sqrt(
            (
                (param_values_arr - mean)
                @ np.linalg.inv(param_cov_mat)
                * (param_values_arr - mean)
            ).sum(axis=-1)
        )
        gaussian_pdf_vals = (
            procdata.eval_conf_lvl
            * gaussian_pdf(param_values_arr, mean, param_cov_mat)
        )
        conf_lvls = np.array([
            calc_conf_lvl(dist, row.size)
            for dist, row in zip(mahalanobis_distances, param_values_arr)
        ])
        is_feasible = off_nominal_df["Termination Condition"].isin(
            {"locallyOptimal", "optimal", "globallyOptimal"}
        ).to_numpy()
        is_infeasible = off_nominal_df["Termination Condition"].isin(
            {"infeasible"}
        ).to_numpy()
        is_other = np.logical_not(np.logical_or(is_feasible, is_infeasible))

        bin_lvls = list(range(0, 90, 20)) + [90, 95, 99]

        sliver_hypervolumes = []
        feasible_sample_counts = []
        feasible_pdf_sums = []
        infeasible_sample_counts = []
        infeasible_pdf_sums = []
        other_sample_counts = []
        other_pdf_sums = []
        for lvl, prevlvl in zip(bin_lvls[1:], bin_lvls[:-1]):
            sliver_hypervolumes.append(
                ellipsoid_hypervolume(mean, param_cov_mat, lvl / 100)
                - ellipsoid_hypervolume(mean, param_cov_mat, prevlvl / 100)
            )
            is_in_sliver = np.logical_and(
                conf_lvls < lvl / 100,
                conf_lvls >= prevlvl / 100,
            )
            feasible_gaussian_sum_in_sliver = gaussian_pdf_vals[
                np.logical_and(is_feasible, is_in_sliver)
            ].sum()
            infeasible_gaussian_sum_in_sliver = gaussian_pdf_vals[
                np.logical_and(is_infeasible, is_in_sliver)
            ].sum()
            other_gaussian_sum_in_sliver = gaussian_pdf_vals[
                np.logical_and(is_other, is_in_sliver)
            ].sum()

            feasible_points_in_sliver = param_values_arr[
                np.logical_and(is_in_sliver, is_feasible)
            ]
            infeasible_points_in_sliver = param_values_arr[
                np.logical_and(is_in_sliver, is_infeasible)
            ]
            other_points_in_sliver = param_values_arr[
                np.logical_and(is_in_sliver, is_other)
            ]

            feasible_sample_counts.append(
                feasible_points_in_sliver.shape[0]
            )
            infeasible_sample_counts.append(
                infeasible_points_in_sliver.shape[0]
            )
            other_sample_counts.append(
                other_points_in_sliver.shape[0]
            )

            feasible_pdf_sums.append(
                feasible_gaussian_sum_in_sliver / gaussian_pdf_vals.sum()
            )
            infeasible_pdf_sums.append(
                infeasible_gaussian_sum_in_sliver / gaussian_pdf_vals.sum()
            )
            other_pdf_sums.append(
                other_gaussian_sum_in_sliver / gaussian_pdf_vals.sum()
            )

        with plt.rc_context(DEFAULT_MPL_RC_PARAMS):
            fig, [ax3, ax1, ax2] = plt.subplots(ncols=3, figsize=(9, 3))

            label_to_heights_map = {
                "feasible": (
                    feasible_sample_counts, feasible_pdf_sums, "steelblue"
                ),
                "infeasible": (
                    infeasible_sample_counts,
                    infeasible_pdf_sums,
                    "lightsteelblue",
                ),
                "other": (other_sample_counts, other_pdf_sums, "orange"),
            }

            counts_bottom = np.zeros(len(bin_lvls[1:]))
            pdf_sums_bottom = np.zeros(len(bin_lvls[1:]))
            for label, val in label_to_heights_map.items():
                sample_counts, pdf_sums, plot_color = val
                ax1.bar(
                    bin_lvls[:-1],
                    sample_counts,
                    label=label,
                    width=[
                        (curr - prev)
                        for curr, prev in zip(bin_lvls[1:], bin_lvls[:-1])
                    ],
                    bottom=counts_bottom,
                    align="edge",
                    edgecolor="white",
                    color=plot_color,
                )
                ax2.bar(
                    bin_lvls[:-1],
                    pdf_sums,
                    label=label,
                    width=[
                        (curr - prev)
                        for curr, prev in zip(bin_lvls[1:], bin_lvls[:-1])
                    ],
                    bottom=pdf_sums_bottom,
                    align="edge",
                    edgecolor="white",
                    color=plot_color,
                )
                counts_bottom += np.array(sample_counts)
                pdf_sums_bottom += np.array(pdf_sums)

            ax1.set_xlabel("Confidence Level (\\%)")
            ax1.set_ylabel("Number of Samples in Homoeoid")
            ax1.set_ylim(0, counts_bottom.max() * 1.05)
            ax1.set_xticks(ticks=bin_lvls, labels=bin_lvls)
            ax1.legend()

            ax2.set_xlabel("Confidence Level (\\%)")
            ax2.set_ylabel("Self-Normalized Gaussian Weight\nin Homoeoid")
            ax2.set_ylim(0, pdf_sums_bottom.max() * 1.05)
            ax2.set_xticks(ticks=bin_lvls, labels=bin_lvls)
            ax2.legend()

            ax3.bar(
                bin_lvls[:-1],
                sliver_hypervolumes,
                width=[
                    (curr - prev)
                    for curr, prev in zip(bin_lvls[1:], bin_lvls[:-1])
                ],
                align="edge",
                edgecolor="white",
                color="dimgray",
            )
            ax3.set_xlabel("Confidence Level (\\%)")
            ax3.set_ylabel("Homoeoid Hypervolume")
            ax3.set_ylim(0, max(sliver_hypervolumes) * 1.05)
            ax3.set_xticks(ticks=bin_lvls, labels=bin_lvls)

            set_nonoverlapping_fixed_xticks(fig, ax1, bin_lvls)
            set_nonoverlapping_fixed_xticks(fig, ax2, bin_lvls)
            set_nonoverlapping_fixed_xticks(fig, ax3, bin_lvls)

            if outdir is not None:
                plt.savefig(
                    os.path.join(
                        outdir,
                        (
                            f"{float_to_str(procdata.design_target)}"
                            "_capture_solution_"
                            f"{float_to_str(procdata.eval_target)}"
                            "_capture_problem_"
                            f"{float_to_str(procdata.design_conf_lvl)}"
                            "_conf_solution_"
                            f"{float_to_str(procdata.design_conf_lvl)}"
                            "_conf_problem"
                            f".{output_plot_fmt}"
                        )
                    ),
                    bbox_inches="tight",
                    dpi=200,
                )
            else:
                plt.show()

            plt.close(fig)


class HeatmapQuantityInfo:
    def __init__(
            self,
            qty_expr_col,
            qty_mult,
            qty_yax_str,
            qty_yax_unit,
            fname,
            pre_mult_offset=0,
            post_mult_offset=0,
            solve_results_col=None,
            solve_results_mult=None,
            solve_results_pre_mult_offset=0,
            solve_results_post_mult_offset=0,
            ):
        """
        Initialize self (see class docstring).
        """
        self.qty_expr_name = qty_expr_col
        self.qty_mult = qty_mult
        self.qty_yax_str = qty_yax_str
        self.qty_yax_unit = qty_yax_unit
        self.pre_mult_offset = pre_mult_offset
        self.post_mult_offset = post_mult_offset
        self.fname = fname
        self.solve_results_col = solve_results_col
        self.solve_results_mult = solve_results_mult
        self.solve_results_pre_mult_offset = solve_results_pre_mult_offset
        self.solve_results_post_mult_offset = solve_results_post_mult_offset

    def apply_mult_and_offsets(self, obj):
        return (
            (obj + self.pre_mult_offset)
            * self.qty_mult
            + self.post_mult_offset
        )

    def apply_solve_results_mult_and_offsets(self, obj):
        return (
            (obj + self.solve_results_pre_mult_offset)
            * self.solve_results_mult
            + self.solve_results_post_mult_offset
        )


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_quantity_distributions(
        heatmap_proc_data_list,
        outdir,
        heatmap_df_col_info_list,
        heatmap_summary_df,
        solver_results_df,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Plot distributions of selected quantities
    (using violin and/or box plots).
    """
    acceptable_termination_str_set = {
        "feasible",
        "optimal",
        "locallyOptimal",
        "globallyOptimal",
    }
    acceptable_pyros_termination_str_set = {
        f"{pyros.pyrosTerminationCondition.__name__}.{ptc.name}"
        for ptc in [
            pyros.pyrosTerminationCondition.robust_optimal,
            pyros.pyrosTerminationCondition.robust_feasible,
        ]
    }
    acceptable_solver_results_df = solver_results_df[
        solver_results_df.solver_termination.termination_condition.isin(
            acceptable_termination_str_set
        )
        | (
            solver_results_df
            .pyros_solver_termination
            .pyros_termination_condition
            .isin(
                acceptable_pyros_termination_str_set
            )
        )
    ]

    are_all_conf_lvls_int = all(
        clvl == int(clvl)
        for clvl in heatmap_summary_df.index.get_level_values(1)
    )

    red_rgb = mpl.colors.ColorConverter.to_rgb("tab:red")
    blue_rgb = mpl.colors.ColorConverter.to_rgb("tab:blue")

    # group data list by design target
    groupby_iter = itertools.groupby(
        heatmap_proc_data_list,
        key=lambda procdata: procdata.design_target,
    )
    for design_target, proc_data_iter in groupby_iter:
        proc_data_sublist = list(proc_data_iter)
        co2_cap_str = get_co2_capture_str(None, design_target)
        for qty_info in heatmap_df_col_info_list:
            default_logger.info(
                "Plotting distributions for "
                f"design target {design_target} "
                f"and quantity {qty_info.qty_yax_str}..."
            )

            dataset = [
                qty_info.apply_mult_and_offsets(
                    procdata.heatmap_df[
                        procdata.heatmap_df["Termination Condition"].isin(
                            acceptable_termination_str_set
                        )
                    ][f"Value({qty_info.qty_expr_name})"].to_numpy()
                )
                for procdata in proc_data_sublist
            ]
            plt.rcParams["font.size"] = 12

            fig, ax = plt.subplots(figsize=(3.1, 3.5))
            ax.grid(axis="x")

            # should box and violin plots include nominally
            # evaluated point?
            ax.set_ylabel(wrap_quantity_str(
                namestr=qty_info.qty_yax_str,
                unitstr=qty_info.qty_yax_unit,
            ))
            ax.set_xlabel(
                "Magnification Factor"
                if label_conf_lvls_as_stdevs
                else "Confidence Level (\\%)"
            )
            violin_parts = ax.violinplot(
                dataset,
                showmedians=False,
                showextrema=False,
            )
            for pc in violin_parts["bodies"]:
                pc.set_edgecolor(set_lightness(blue_rgb, 0.6))
                pc.set_facecolor(set_lightness(blue_rgb, 0.9))
                pc.set_linewidth(0.6)
                pc.set_alpha(0.5)

            ax.boxplot(
                dataset,
                widths=0.2,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="black"),
                boxprops=dict(facecolor="none", color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(markersize=4),
                capprops=dict(color="black"),
            )

            nominal_vals = qty_info.apply_mult_and_offsets(
                heatmap_summary_df
                .loc[design_target, f"Value({qty_info.qty_expr_name})"]
                .nominal_val
                .to_numpy()
            )
            ax.scatter(
                range(1, len(dataset) + 1),
                nominal_vals,
                color="None",
                edgecolor=set_lightness(blue_rgb, 0.3),
                zorder=3.1,
                s=24,
                marker="^",
                label="nominal (re-optimized)",
            )

            if qty_info.solve_results_col is not None:
                solver_vals = qty_info.apply_solve_results_mult_and_offsets(
                    acceptable_solver_results_df.loc[
                        design_target, qty_info.solve_results_col
                    ].astype(float).to_numpy()
                )
                ax.scatter(
                    range(1, len(dataset) + 1),
                    solver_vals,
                    edgecolor=set_lightness(red_rgb, 0.5),
                    facecolor="none",
                    zorder=3,
                    s=36,
                    marker="*",
                    label="nominal (PyROS DR)",
                )

            ax.set_xticks(range(1, 1 + len(proc_data_sublist)))
            xtick_labels = []
            for pdata in proc_data_sublist:
                if pdata.design_conf_lvl == 0:
                    label = "0 (det.)"
                else:
                    label = get_conf_lvl_plot_label(
                        conf_lvl=pdata.design_conf_lvl,
                        n=6,
                        as_mag_factor=label_conf_lvls_as_stdevs,
                        conf_lvl_suffix="",
                        decimals=int(not are_all_conf_lvls_int),
                    )
                xtick_labels.append(label)
            ax.set_xticklabels(xtick_labels)

            ax.legend(
                bbox_to_anchor=(0.5, 1.05),
                loc="lower center",
                ncol=1,
            )

            fig.savefig(
                os.path.join(
                    outdir,
                    (
                        f"distribution_{qty_info.fname}_{co2_cap_str}"
                        f".{output_plot_fmt}"
                    ),
                ),
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
            plt.close(fig)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_confidence_level_sensitivity(
        heatmap_proc_data_list,
        heatmap_summary_df,
        outdir,
        param_cov_mat,
        heatmap_df_col_info_list,
        confidence_interval=0.99,
        output_plot_fmt="png",
        ):
    """
    Plot sensitivity to confidence level.
    """
    # NOTE: this assumes that for all results,
    #       the same sample set was used
    first_heatmap_df = next(iter(heatmap_proc_data_list)).heatmap_df
    all_samples = np.stack(first_heatmap_df["Parameter Values"].to_numpy())
    mean, eval_samples = all_samples[0], all_samples[1:]
    mahalanobis_distances = np.sqrt((
        (eval_samples - mean)
        @ np.linalg.inv(param_cov_mat)
        * (eval_samples - mean)
    ).sum(axis=-1))
    sample_conf_lvls = np.array([
        100 * calc_conf_lvl(dist, all_samples.shape[-1])
        for dist in mahalanobis_distances
    ])

    for design_target in heatmap_summary_df.index.levels[0]:
        fig, (ax_gaussian_prob, ax_num_feas) = plt.subplots(
            ncols=2, figsize=(6, 3)
        )
        feasibility_eval = (
            heatmap_summary_df.loc[design_target].feasibility_evaluation
        )
        proc_data_conf_lvls = feasibility_eval.index.get_level_values(0)

        default_logger.info(
            "Plotting confidence level sensitivity for "
            f"design target {design_target}..."
        )

        feas_probs = feasibility_eval.expected_val
        feas_stdevs = feasibility_eval["std"]
        ax_gaussian_prob.plot(proc_data_conf_lvls, feas_probs)
        ax_gaussian_prob.fill_between(
            np.array(proc_data_conf_lvls).astype(float),
            (feas_probs - feas_stdevs).to_numpy().astype(float),
            (feas_probs + feas_stdevs).to_numpy().astype(float),
            alpha=0.3,
        )
        ax_gaussian_prob.set_xlabel("Design Confidence Level (\\%)")
        ax_gaussian_prob.set_ylabel(textwrap.fill(
            "Feasible Gaussian Probability Mass (\\%)", AX_LABEL_TEXTWIDTH
        ))

        ax_num_feas.plot(
            proc_data_conf_lvls,
            feasibility_eval.num_feas_samples,
            label=f"feasible in {confidence_interval * 100:.2f}\\% ellipsoid",
        )
        ax_num_feas.plot(
            proc_data_conf_lvls,
            [
                sample_conf_lvls[sample_conf_lvls <= lvl].size
                for lvl in proc_data_conf_lvls
            ],
            label="located in design ellipsoid",
        )
        ax_num_feas.set_xlabel("Design Confidence Level (\\%)")
        ax_num_feas.set_ylabel("Number of Samples")
        ax_num_feas.legend()
        ax_num_feas.legend(
            bbox_to_anchor=(0, -0.2),
            loc="upper left",
            ncol=1,
        )

        co2_cap_str = get_co2_capture_str(None, design_target)
        fig.savefig(
            os.path.join(
                outdir,
                f"summary_feasibility_{co2_cap_str}.{output_plot_fmt}"
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

        for qty_info in heatmap_df_col_info_list:
            fig_cost, (conflvl_ax, size_ax) = plt.subplots(
                ncols=2,
                figsize=(6, 2.5),
            )
            qty_subdf = heatmap_summary_df.loc[design_target][
                f"Value({qty_info.qty_expr_name})"
            ]
            expected_vals = qty_info.apply_mult_and_offsets(
                qty_subdf.expected_val
            )
            stdevs = qty_subdf["std"] * abs(qty_info.qty_mult)
            t_factors = sp.stats.t.ppf(
                # assume two-tailed, so halve alpha
                1 - (1 - confidence_interval) / 2,
                qty_subdf.num_feas_samples.to_numpy(dtype=int),
            )
            conflvl_ax.plot(
                proc_data_conf_lvls,
                expected_vals,
                color="black",
            )
            conflvl_ax.fill_between(
                np.array(proc_data_conf_lvls).astype(float),
                (expected_vals - t_factors * stdevs).to_numpy().astype(float),
                (expected_vals + t_factors * stdevs).to_numpy().astype(float),
                alpha=0.3,
                facecolor="black",
                edgecolor=None,
            )
            conflvl_ax.set_xlabel("Design Confidence Level (\\%)")
            conflvl_ax.set_ylabel(wrap_quantity_str(
                namestr=f"Expected {qty_info.qty_yax_str}",
                unitstr=qty_info.qty_yax_unit,
            ))

            conflvl_ax_twin = conflvl_ax.twinx()
            conflvl_ax_twin.tick_params(axis="y", colors="red")
            conflvl_ax_twin.yaxis.label.set_color("red")
            conflvl_ax_twin.set_ylabel("Number of Standard Deviations")

            num_stdevs_list = [
                mag_factor(lvl / 100, all_samples.shape[-1])
                for lvl in proc_data_conf_lvls
            ]
            conflvl_ax_twin.plot(
                proc_data_conf_lvls,
                num_stdevs_list,
                color="red",
            )

            size_ax.set_xlabel("Number of Standard Deviations")
            size_ax.set_ylabel(wrap_quantity_str(
                namestr=f"Expected {qty_info.qty_yax_str}",
                unitstr=qty_info.qty_yax_unit,
            ))
            size_ax.plot(num_stdevs_list, expected_vals),
            size_ax.fill_between(
                np.array(num_stdevs_list).astype(float),
                (expected_vals - t_factors * stdevs).to_numpy().astype(float),
                (expected_vals + t_factors * stdevs).to_numpy().astype(float),
                alpha=0.3,
            )

            fig_cost.savefig(
                os.path.join(
                    outdir,
                    (
                        f"summary_{qty_info.fname}_{co2_cap_str}."
                        f"{output_plot_fmt}"
                    ),
                ),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig_cost)


@mpl.rc_context(DEFAULT_MPL_RC_PARAMS)
def plot_capture_target_sensitivity(
        heatmap_proc_data_list,
        heatmap_summary_df,
        outdir,
        param_cov_mat,
        heatmap_df_col_info_list,
        confidence_interval=0.99,
        label_conf_lvls_as_stdevs=False,
        output_plot_fmt="png",
        ):
    """
    Plot sensitivity of expected value, variance,
    and range to the CO2 capture target.
    """
    assert heatmap_proc_data_list
    assert 0 <= confidence_interval <= 1

    for qty_info in heatmap_df_col_info_list:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(6, 5),
        )
        (ax_nom_val, ax_expected_val), (ax_percent_cv, ax_percent_rr) = (
            axs.tolist()
        )
        ax_nom_val.set_ylabel(wrap_quantity_str(
            f"Nominal {qty_info.qty_yax_str}",
            qty_info.qty_yax_unit,
            width=AX_LABEL_TEXTWIDTH,
        ))
        ax_expected_val.set_ylabel(wrap_quantity_str(
            f"Expected {qty_info.qty_yax_str}",
            qty_info.qty_yax_unit,
            width=AX_LABEL_TEXTWIDTH,
        ))
        ax_percent_cv.set_ylabel(wrap_quantity_str(
            f"\\%CV {qty_info.qty_yax_str}",
            None,
            width=AX_LABEL_TEXTWIDTH,
        ))
        ax_percent_rr.set_ylabel(wrap_quantity_str(
            f"\\%RR {qty_info.qty_yax_str}",
            None,
            width=AX_LABEL_TEXTWIDTH,
        ))
        for ax in axs.flatten():
            ax.set_xlabel(wrap_quantity_str(
                namestr="Design $\\text{CO}_2$ Capture Target",
                unitstr="\\%",
                width=AX_LABEL_TEXTWIDTH,
            ))

        for design_conf_lvl in heatmap_summary_df.index.levels[1]:
            default_logger.info(
                f"Plotting capture target sensitivity for confidence level "
                f"{design_conf_lvl} and quantity {qty_info.qty_yax_str!r}..."
            )
            conf_lvl_subdf = heatmap_summary_df.xs(
                design_conf_lvl, level=1,
            )
            design_targets_for_conf_lvl = (
                conf_lvl_subdf.index.get_level_values(0)
            )

            conf_lvl_subdf_vals = (
                conf_lvl_subdf[f"Value({qty_info.qty_expr_name})"]
            )

            nominal_costs = qty_info.apply_mult_and_offsets(
                conf_lvl_subdf_vals.nominal_val
            )
            expected_costs = qty_info.apply_mult_and_offsets(
                conf_lvl_subdf_vals.expected_val
            )
            percent_cv_vals = 100 * (
                conf_lvl_subdf_vals.std_over_abs_expected_val
            )
            percent_rr_vals = 100 * conf_lvl_subdf_vals.range_over_expected_val
            stdevs = conf_lvl_subdf_vals["std"].copy() * abs(qty_info.qty_mult)
            t_factors = sp.stats.t.ppf(
                # assume two-tailed, so halve alpha
                1 - (1 - confidence_interval) / 2,
                conf_lvl_subdf_vals.num_feas_samples.to_numpy(
                    dtype=int
                ),
            )

            # continuous error bars.
            # note: fill_between specifically requires compatible
            # ndarray-type arguments
            ax_expected_val.fill_between(
                np.array(design_targets_for_conf_lvl).astype(float),
                (expected_costs - t_factors * stdevs).to_numpy().astype(float),
                (expected_costs + t_factors * stdevs).to_numpy().astype(float),
                alpha=0.3,
            )

            plot_label = (
                ("det." if label_conf_lvls_as_stdevs else "deterministic")
                if design_conf_lvl == 0
                else get_conf_lvl_plot_label(
                    conf_lvl=design_conf_lvl,
                    n=6,
                    as_mag_factor=label_conf_lvls_as_stdevs,
                )
            )

            ax_to_yvals_zip = (
                (ax_nom_val, nominal_costs),
                (ax_expected_val, expected_costs),
                (ax_percent_cv, percent_cv_vals),
                (ax_percent_rr, percent_rr_vals),
            )
            for ax, yvals in ax_to_yvals_zip:
                ax.plot(
                    design_targets_for_conf_lvl,
                    yvals,
                    label=plot_label,
                )

        for ax in axs.flatten():
            ax.legend()
            set_nonoverlapping_fixed_xticks(
                fig, ax, design_targets_for_conf_lvl
            )

        fig.savefig(
            os.path.join(
                outdir,
                f"design_target_sens_{qty_info.fname}.{output_plot_fmt}",
            ),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close(fig)


def plot_heatmap_scatter_results(
        heatmap_proc_data_list,
        param_cov_mat,
        heatmap_df_col_info_list,
        outdir,
        projection_idx_pairs=None,
        savefig_kwds=None,
        include_only_designs=None,
        output_plot_fmt="png",
        ):
    """
    Generate heatmapped (hence the name) scatter plots
    visualizing the objective value and feasibility of
    model solutions.
    """
    from itertools import combinations
    from confidence_ellipsoid import (
        get_pyros_ellipsoidal_set, plot_confidence_ellipsoid_projection
    )

    acceptable_termination_str_set = {
        "feasible",
        "optimal",
        "locallyOptimal",
        "globallyOptimal",
    }
    uncertain_param_quantity_unit_pairs = [
        ("$\\text{HCO}_3^{-}$ Equilibrium Constant Coeff. 2", "K"),
        # carbamate representation used as in reaction (i) of
        # https://doi.org/10.1021/je980290n
        ("$\\text{MEACOO}^{-}$ Equilibrium Constant Coeff. 2", "K"),
        ("$\\text{CO}_2$ Henry's Law Constant Coeff. 1", None),
        ("$\\text{CO}_2$ Henry's Law Constant Coeff. 2", "K${}^{-1}$"),
        ("$\\text{CO}_2$ Henry's Law Constant Coeff. 3", "K${}^{-2}$"),
        ("$\\text{CO}_2$ Henry's Law Constant Coeff. 4", None),
    ]

    param_dim = param_cov_mat.shape[0]
    if projection_idx_pairs is None:
        projection_idx_pairs = list(combinations(range(param_dim), 2))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # group by capture target; apply filter, if provided
    cap_target_to_proc_data_map = dict()
    for proc_data in heatmap_proc_data_list:
        design_target = proc_data.design_target
        include_proc_data = (
            include_only_designs is None
            or (
                design_target in include_only_designs.keys()
                and proc_data.design_conf_lvl
                in include_only_designs[design_target]
            )
        )
        if include_proc_data:
            design_target_data_list = cap_target_to_proc_data_map.setdefault(
                design_target,
                [],
            )
            design_target_data_list.append(proc_data)

    cap_target_to_proc_data_map = dict(sorted({
        dtarg: sorted(pd_list, key=lambda pdata: pdata.design_conf_lvl)
        for dtarg, pd_list in cap_target_to_proc_data_map.items()
    }.items()))

    heatmap_df_col_info_list.insert(
        0,
        HeatmapQuantityInfo(
            qty_expr_col="feasibility",
            qty_mult=1,
            qty_yax_str="feasibility",
            qty_yax_unit=None,
            fname="feasibility",
        ),
    )

    for design_target, proc_data_list in cap_target_to_proc_data_map.items():
        pyros_ellipsoid = get_pyros_ellipsoidal_set(
            mean=(
                proc_data_list[0].heatmap_df["Parameter Values"].iloc[0]
            ),
            cov_mat=param_cov_mat,
            level=proc_data_list[0].eval_conf_lvl / 100,
        )

        # get sorted heatmap results
        rc_params = DEFAULT_MPL_RC_PARAMS.copy()
        rc_params.update(
            {"figure.constrained_layout.h_pad": 25 / 72, "axes.grid": False}
        )
        with plt.rc_context(rc_params):
            for qty_info in heatmap_df_col_info_list:
                if qty_info.qty_expr_name != "feasibility":
                    min_col_val = np.nanmin([
                        np.nanmin(
                            qty_info.apply_mult_and_offsets(
                                data.heatmap_df[
                                    f"Value({qty_info.qty_expr_name})"
                                ].to_numpy()
                            )
                        )
                        for data in proc_data_list
                    ])
                    max_col_val = np.nanmax([
                        np.nanmax(
                            qty_info.apply_mult_and_offsets(
                                data.heatmap_df[
                                    f"Value({qty_info.qty_expr_name})"
                                ].to_numpy()
                            )
                        )
                        for data in proc_data_list
                    ])
                else:
                    min_col_val = 0
                    max_col_val = 1

                default_logger.info(
                    f"Generating heatmap scatter plots for design target "
                    f"{design_target} and quantity "
                    f"{qty_info.qty_expr_name!r}..."
                )
                fig, ax_arr = plt.subplots(
                    ncols=len(projection_idx_pairs),
                    nrows=len(proc_data_list),
                    figsize=(
                        # guesstimate of a good aspect ratio
                        1.5 * 6.2,
                        (
                            1.5 * (
                                7.0
                                if qty_info.qty_expr_name == "feasibility"
                                else 6.0
                            )
                            * len(proc_data_list)
                            / len(projection_idx_pairs)
                        ),
                    ),
                    sharex="col",
                    sharey=False,
                    squeeze=False,
                )
                for ax_row, htmp_data in zip(ax_arr, proc_data_list):
                    for ax, proj_pair in zip(ax_row, projection_idx_pairs):
                        ax.set_ylabel(wrap_quantity_str(
                            *uncertain_param_quantity_unit_pairs[
                                proj_pair[1]
                            ],
                            width=28,
                        ))
                        heatmap_df = htmp_data.heatmap_df
                        is_feasible = heatmap_df["Termination Condition"].isin(
                            acceptable_termination_str_set
                        )
                        feasible_rows = heatmap_df[is_feasible]
                        feasible_param_points = np.array([
                            pt for pt in feasible_rows["Parameter Values"]
                        ])

                        # want marker area ('s' parameter)
                        # to be 15 for 401 samples, 10 for 1001 samples,
                        # so as not to ruin visibility among
                        # overlapping points
                        marker_size = (
                            15
                            + (10 - 15) / (1001 - 401)
                            * (heatmap_df.index.size - 401)
                        )

                        if qty_info.qty_expr_name == "feasibility":
                            feasible_costs = is_feasible
                        else:
                            feasible_costs = qty_info.apply_mult_and_offsets(
                                feasible_rows[
                                    f"Value({qty_info.qty_expr_name})"
                                ].to_numpy()
                            )

                        # note: there may be points for which
                        #       termination was neither feasible
                        #       nor infeasible, due to, for example,
                        #       numerical errors.
                        #       we loosely consider these infeasible.
                        infeasible_rows = heatmap_df[~is_feasible]
                        infeasible_param_points = np.array([
                            pt for pt in infeasible_rows["Parameter Values"]
                        ])

                        # scatterplots are produced here
                        if feasible_param_points.size > 0:
                            plot = ax.scatter(
                                feasible_param_points[:, proj_pair[0]],
                                feasible_param_points[:, proj_pair[1]],
                                c=(
                                    "steelblue"
                                    if qty_info.qty_expr_name == "feasibility"
                                    else feasible_costs
                                ),
                                s=marker_size,
                                alpha=0.7,
                                cmap=(
                                    None
                                    if qty_info.qty_expr_name == "feasibility"
                                    else "plasma"
                                ),
                                vmin=(
                                    None
                                    if qty_info.qty_expr_name == "feasibility"
                                    else min_col_val
                                ),
                                vmax=(
                                    None
                                    if qty_info.qty_expr_name == "feasibility"
                                    else max_col_val
                                ),
                                label=(
                                    "feasible"
                                    if qty_info.qty_expr_name == "feasibility"
                                    else None
                                ),
                            )

                        infeas_plot_kwds = dict(
                            s=marker_size,
                            facecolor="none",
                            edgecolor=mpl.colors.colorConverter.to_rgba(
                                "black",
                                alpha=1,
                            ),
                            linewidth=1,
                            label=(
                                "infeasible"
                                if qty_info.qty_expr_name == "feasibility"
                                else None
                            ),
                            marker="^",
                        )
                        if infeasible_param_points.size > 0:
                            infeas_scatterplot = ax.scatter(
                                infeasible_param_points[:, proj_pair[0]],
                                infeasible_param_points[:, proj_pair[1]],
                                **infeas_plot_kwds,
                            )
                        else:
                            infeas_scatterplot = ax.scatter(
                                [],
                                [],
                                **infeas_plot_kwds,
                            )

                        # boundary
                        ellipsoid_label = (
                            f"{htmp_data.eval_conf_lvl}\\% ellipsoid"
                        )
                        ellipsoid = plot_confidence_ellipsoid_projection(
                            mean=pyros_ellipsoid.center,
                            cov_mat=pyros_ellipsoid.shape_matrix,
                            level=proc_data_list[0].eval_conf_lvl / 100,
                            ax=ax,
                            plane_idxs=proj_pair,
                            samples=0,
                            color="black",
                            linestyle="dashed",
                            linewidth=1,
                            marker="none",
                            label=ellipsoid_label,
                        )

                        sbs = ax.get_subplotspec()
                        if sbs.is_first_row() and sbs.is_last_col():
                            legend_pos_args = dict(
                                bbox_to_anchor=(1.02, 1.4),
                                loc="upper right",
                                ncol=1,
                            )
                            if qty_info.qty_expr_name == "feasibility":
                                ax.legend(**legend_pos_args)
                            else:
                                ax.legend(
                                    [infeas_scatterplot, *ellipsoid],
                                    ["infeasible", ellipsoid_label],
                                    **legend_pos_args,
                                )

                        x_param_bounds = pyros_ellipsoid.parameter_bounds[
                            proj_pair[0]
                        ]

                        # prevent overlapping tick labels
                        ax.set_xticks(
                            np.linspace(*x_param_bounds, num=101)[[1, 50, -2]]
                        )
                        ax.set_xlabel(wrap_quantity_str(
                            *uncertain_param_quantity_unit_pairs[
                                proj_pair[0]
                            ],
                            width=28,
                        ))

                if qty_info.qty_expr_name != "feasibility":
                    # need colorbar for quantities
                    fig.colorbar(
                        plot,
                        ax=ax_arr,
                        label=wrap_quantity_str(
                            namestr=qty_info.qty_yax_str,
                            unitstr=qty_info.qty_yax_unit,
                            width=AX_LABEL_TEXTWIDTH * len(proc_data_list),
                        ),
                    )

                # finally, serialize the figure for this target
                outfile = os.path.join(
                    outdir,
                    (
                        f"scatter_plots_{float_to_str(design_target)}_"
                        f"capture_{qty_info.fname}.{output_plot_fmt}"
                    ),
                )
                if savefig_kwds is None:
                    savefig_kwds = dict()
                fig.savefig(outfile, **savefig_kwds)
                plt.close(fig)

    default_logger.info("Done generating heatmap scatter plots.")
