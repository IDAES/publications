"""
Utility functions and solver wrappers for MEA model PyROS workflows.
"""

from contextlib import nullcontext, contextmanager
import copy
import datetime
import logging
import os
import platform
import psutil
import subprocess
import sys
import textwrap

import numpy as np
from scipy.linalg import svd

from pyomo.common.collections import ComponentSet
from pyomo.common.errors import ApplicationError
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import ParamData, VarData
from pyomo.core.expr.visitor import (
    identify_mutable_parameters,
    identify_variables,
    replace_expressions,
)
from pyomo.util.vars_from_expressions import get_vars_from_components
import pyomo.environ as pyo
from pyomo.opt import SolverResults, TerminationCondition, Solution

from idaes.core.base.var_like_expression import VarLikeExpressionData
from pyomo.common.tee import capture_output
from pyomo.common.log import LogStream

import idaes.core.util.scaling as iscale


default_logger = logging.getLogger(__name__)


def get_uniform_grid(nfe):
    # Finite element list in the spatial domain
    x_nfe_list = [i / nfe for i in range(nfe + 1)]
    return x_nfe_list


def pairwise(iterable):
    """
    Equivalent implementation of Python's standard
    ``itertools.pairwise`` iterator.

    See also https://docs.python.org/3/library/itertools.html
    """
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


class ModuleInfo:
    """
    Module information.
    """

    def __init__(self, module, title=None, remote_name="origin"):
        """
        Initialize self (see class docstring).
        """
        info_dict = self.get_module_version_info(
            module,
            remote_name=remote_name,
        )

        self.title = info_dict["name"] if title is None else title
        self.name = info_dict["name"]
        self.version = info_dict["version"]
        self.git_branch_local = info_dict["git_branch_local_name"]
        self.git_commit_hash = info_dict["git_commit_hash"]
        self.git_remote_name = remote_name
        self.git_branch_remote = info_dict["git_branch_remote_name"]
        self.git_remote_url = info_dict["git_remote_url"]
        self.git_diff = info_dict["git_diff"]

    def __str__(self):
        """String representation of self."""
        title_line = f"Module titled: {self.title}"
        name_line = f"Name: {self.name}"
        version_line = f"Version: {self.version}"
        local_branch_line = f"Git branch local name: {self.git_branch_local}"
        remote_branch_line = (
            f"Git branch remote name: {self.git_branch_remote}"
        )
        commit_line = f"Git commit short hash: {self.git_commit_hash}"
        remote_line = f"Git remote local name: {self.git_remote_name}"
        url_line = f"Git remote url: {self.git_remote_url}"

        indented_diff_str = textwrap.indent(self.git_diff, " " * 4)
        git_diff_lines = f"Git diff:\n {indented_diff_str}"

        return "\n ".join((
            title_line,
            name_line,
            version_line,
            local_branch_line,
            remote_branch_line,
            commit_line,
            remote_line,
            url_line,
            git_diff_lines,
        ))

    @staticmethod
    def get_module_version_info(module, remote_name="origin"):
        """
        Get git commit hash for given directory.
        """
        module_dir = os.path.join(*os.path.split(module.__file__)[:-1])
        module_info = {
            "name": module.__name__,
            "version": getattr(module, "__version__", "unknown"),
        }

        git_args_dict = {
            "git_branch_local_name": [
                "git",
                "-C",
                f"{module_dir}",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ],
            "git_commit_hash": [
                "git",
                "-C",
                f"{module_dir}",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            "git_remote_url": [
                "git",
                "-C",
                f"{module_dir}",
                "remote",
                "get-url",
                # NOTE: doesn't get url of remote branch
                #       being tracked by local branch
                f"{remote_name}",
            ],
            "git_branch_remote_name": [
                "git",
                "-C",
                f"{module_dir}",
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{u}",
            ],
            # NOTE: HEAD means compare both staged and unstaged changes
            #       to last commit. untracked files are not reported.
            "git_diff": [
                "git",
                "-C",
                f"{module_dir}",
                "diff",
                "HEAD",
            ],
        }

        # NOTE: excludes flag for whether there were
        #       unstaged changes
        for key, command_args in git_args_dict.items():
            try:
                command_output = (
                    subprocess
                    .check_output(command_args)
                    .decode("utf-8")
                    .strip()
                )
            except subprocess.CalledProcessError:
                command_output = "unknown"

            module_info[key] = command_output

        return module_info


def get_main_dependency_module_info():
    """
    Get module version information for main dependencies
    (Pyomo, IDAES, and the present package).
    """
    import pyomo
    import idaes
    import sys
    dependency_modules = {
        "pyomo": pyomo,
        "idaes-pse": idaes,
        "mea-flowsheet": sys.modules[__name__],
    }

    return [
        ModuleInfo(module, title=title)
        for title, module in dependency_modules.items()
    ]


def get_pip_list_output():
    """
    Get Python dependency versions through ``pip list``.
    """
    try:
        command_output = (
            subprocess
            .check_output(["pip", "list"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        command_output = "unknown"

    return command_output


def log_main_dependency_module_info(logger, level=logging.INFO):
    """
    Log main dependency module info.
    """
    logger.info(" MAIN DEPENDENCY VERSIONS ".center(78, "="))
    module_info_list = get_main_dependency_module_info()
    for mod_info in module_info_list:
        logger.log(msg=str(mod_info), level=logging.INFO)
        logger.info("-" * 78)
    logger.info("=" * 78)

    logger.info(" PIP LIST".center(78, "="))
    logger.info(get_pip_list_output())
    logger.info("=" * 78)


def get_platform_info():
    """
    Get system information.
    """
    return platform.uname()._asdict()


def get_basic_cpu_info():
    """
    Get system hostname.
    """
    basic_keys = [
        "Architecture",
        "CPU(s)",
        "Thread(s) per core",
        "Core(s) per socket",
        "Socket(s)",
        "Model name",
        "CPU max MHz",
        "CPU min MHz",
    ]
    info_str = (
        subprocess
        .check_output("lscpu", shell=True)
        .decode("ascii")
        .strip()
    )
    cpu_info_lines = info_str.strip().split("\n")
    cpu_info_dict = dict()
    for entry in cpu_info_lines:
        entry_args = entry.split(":")
        key = entry_args[0]
        val = ":".join(entry_args[1:]).strip()

        cpu_info_dict[key] = val

    return {cpukey: cpu_info_dict[cpukey] for cpukey in basic_keys}


def get_basic_ram_info():
    """
    Get basic information about system memory.
    """
    return {
        "RAM": str(round(psutil.virtual_memory().total / 1024 ** 3)) + "GB"
    }


def log_script_invocation(logger, args):
    """
    Log script invocation details.
    """
    platform_info_str = "System platform info:\n " + "\n ".join(
        f"{key} : {val}" for key, val in get_platform_info().items()
    )
    cpu_info_str = "Basic system CPU info:\n " + "\n ".join(
        f"{key} : {val}" for key, val in get_basic_cpu_info().items()
    )
    ram_info_str = "Basic system RAM info:\n " + "\n ".join(
        f"{key} : {val}" for key, val in get_basic_ram_info().items()
    )
    logger.info(f"Invoking script: {sys.argv[0]}")
    logger.info(f"\nPython version: {sys.version}")
    logger.info(f"\nInvoked at UTC: {datetime.datetime.utcnow().isoformat()}")
    logger.info("\n" + platform_info_str)
    logger.info("\n" + cpu_info_str)
    logger.info("\n" + ram_info_str)
    logger.info("\nScript CLI call:")
    logger.info("python " + " ".join(sys.argv))
    logger.info("\nScript arguments after parsing:")
    logger.info(
        "    " + "\n    ".join(
            f"{key}={val!r}" for key, val in args._get_kwargs()
        )
    )


def generate_solver_executable_validator(solver_name):
    """
    Create method that validates that the executable matched
    to a given solver is compatible with instantation of
    a corresponding Pyomo solver interface object.
    """
    def check_solver_executable(executable):
        assert os.path.exists(executable)
        solver = pyo.SolverFactory(solver_name, executable=executable)
        assert solver.available()

        return executable

    return check_solver_executable


def get_path_to_solver_command(solver_command):
    """
    Get path to solver executable using shell command
    ``which {solver_command}``.
    """
    path_output = (
        subprocess
        .check_output(["which", solver_command])
        .decode("ascii")
        .strip()
    )
    if not path_output:
        return None
    return path_output


def get_solver(solver=None, solver_options=None, writer_config=None):
    """
    Wrapper around IDAES `get_solver` function that allows custom
    executable path to be specified as entry 'executable' of solver
    options.
    """
    from idaes.core.solvers import get_solver as idaes_get_solver

    executable = None
    if solver_options is not None:
        executable = solver_options.pop("executable", None)
    solver = idaes_get_solver(
        solver=solver,
        solver_options=solver_options,
        writer_config=writer_config,
    )
    if executable is not None:
        solver.set_executable(executable)

    return solver


def deactivate_scaling_factors(model):
    """
    Deactivate the scaling factor Suffix objects of a model.
    """
    scaling_factors = [
        c for c in model.component_data_objects(pyo.Suffix, active=True)
        if "scaling_factor" in c.name
    ]
    for suffix in scaling_factors:
        suffix.deactivate()


def get_state_vars(model, dof_vars, include_fixed=False):
    """
    Obtain state variables of a model. Any unfixed variable
    which is not a degree-of-freedom variable is considered
    a state variable.

    Yields
    ------
    var : Var
        A state variable.
    """
    dof_var_set = ComponentSet(dof_vars)
    seen_vars = ComponentSet()

    for var in model.component_data_objects(pyo.Var, active=True):
        if var not in seen_vars:
            seen_vars.add(var)
            dof_filter = var not in dof_var_set
            fixed_filter = include_fixed or not var.fixed
            if dof_filter and fixed_filter:
                yield var


def _get_state_vars_in_expr(
        expr,
        dof_var_set,
        include_fixed=False,
        exclude_vars=None,
        ):
    """
    Get state variables in an expression.
    """
    if exclude_vars is None:
        exclude_vars = ComponentSet()

    for var in identify_variables(expr):
        if var not in exclude_vars:
            exclude_vars.add(var)
            dof_filter = var not in dof_var_set
            fixed_filter = include_fixed or not var.fixed
            if dof_filter and fixed_filter:
                yield var


def get_state_vars_in_active_cons(
        model,
        dof_vars,
        equality_only=False,
        include_fixed=False,
        include_vars_in_active_obj=True,
        ):
    """
    Obtain state variables of a model participating in
    the model's active Constraint components, and optionally, the
    variables in the model's active Objective components.
    Any unfixed variable which is not a degree-of-freedom variable
    is considered a state variable.

    Yields
    ------
    Var
        A state variable participating in a model
        Constraint with `active=True`.
    """
    dof_var_set = ComponentSet(dof_vars)
    seen_vars = ComponentSet()

    for con in model.component_data_objects(
            pyo.Constraint,
            active=True,
            descend_into=True,
            ):
        if not equality_only or con.equality:
            yield from _get_state_vars_in_expr(
                expr=con.body,
                dof_var_set=dof_var_set,
                include_fixed=include_fixed,
                exclude_vars=seen_vars,
            )

    if include_vars_in_active_obj:
        for obj in model.component_data_objects(
                pyo.Objective, active=True, descend_into=True
                ):
            yield from _get_state_vars_in_expr(
                expr=obj.expr,
                dof_var_set=dof_var_set,
                include_fixed=include_fixed,
                exclude_vars=seen_vars,
            )


def get_eqs_with_var(var, blk=None, **cdata_kwargs):
    """
    Get equality constraints in a block in which a variable
    participates.

    Parameters
    ----------
    var : VarData
        Variable of interest
    blk : BlockData, optional
        Block from which the desired constraints are descended.
        Passed to ``get_cons_with_component()``.
    **cdata_kwargs : dict, optional
        Keyword arguments to ``blk.component_data_objects()``,
        excluding 'ctype'.

    Returns
    -------
    list
        Equality constraints in which the variables participate.
    """
    return list(
        get_cons_with_component(var, blk=blk, equality=True, **cdata_kwargs)
    )


def get_cons_with_component(cdata, blk=None, equality=None, **cdata_kwargs):
    """
    Get constraints in whose expressions a component data object
    (e.g., Var or Param) appears.

    Parameters
    ----------
    cdata : ComponentData
        Component data object of interest.
    blk : BlockData, optional
        Block from which the desired constraints are descended.
        Passed to ``
    equality : bool, optional
        True to consider only equality constraints,
        False to consider only inequality constraints
        (including ranged ones),
        None to ignore consideration of constraint type.
    **cdata_kwargs : dict, optional
        Keyword arguments to ``blk.component_data_objects()``,
        excluding 'ctype'.

    Yields
    ------
    ConstraintData
        A constraint descended from `blk`
        in whose expression the variable appears.
    """
    for con in get_comps_with_component(cdata, pyo.Constraint, **cdata_kwargs):
        con_filter = (
            equality is None
            or (equality and con.equality)
            or not (equality or con.equality)
        )
        if con_filter:
            yield con


def get_exprs_with_component(cdata, blk=None, **cdata_kwargs):
    """
    Get named expressions in which a variable participates.

    Parameters
    ----------
    cdata : ComponentData
        Component data object of interest.
    blk : BlockData
        Block from which the named expressions are descended.
    **cdata_kwargs : dict, optional
        Keyword arguments to ``blk.component_data_objects()``,
        excluding 'ctype'.

    Yields
    ------
    ExpressionData
        An expression descended from `blk` in which `cdata` appears.
    """
    yield from get_comps_with_component(
        cdata=cdata,
        ctype=pyo.Expression,
        blk=None,
        **cdata_kwargs,
    )


def get_comps_with_component(cdata, ctype, blk=None, **cdata_kwargs):
    """
    Get component objects of a given type in whose expressions
    a specified Var or Param data object appears.

    Parameters
    ----------
    cdata : VarData or ParamData
        Component data object of interest.
    ctype : type
        Pyomo component type of which instances have an attribute
        named 'expr'.
    blk : BlockData, optional
        Block from which the desired constraints are descended.
        If None is passed, then ``cdata.model()`` is used.
    **cdata_kwargs : dict, optional
        Keyword arguments to ``blk.component_data_objects()``,
        excluding 'ctype'.

    Yields
    ------
    ComponentData
        A component object in whose expression
        the specified Var or Param data object appears.

    Raises
    ------
    TypeError
        If `cdata` is not a `VarData` or `ParamData` object.
    """
    if isinstance(cdata, VarData):
        identify_comps = identify_variables
    elif isinstance(cdata, ParamData):
        identify_comps = identify_mutable_parameters
    else:
        raise TypeError(
            f"Component of type {type(cdata).__name__!r} not supported."
        )

    if blk is None:
        blk = cdata.model()

    for expr in blk.component_data_objects(ctype, **cdata_kwargs):
        if cdata in ComponentSet(identify_comps(expr.expr)):
            yield expr


def get_vars_in_con(con, include_fixed=False):
    """
    Get variables participating in the expression of a
    constraint data object.

    Parameters
    ----------
    con : ConstraintData
        Constraint of interest.
    include_fixed : bool, optional
        True to include fixed variables participating in
        the expression, False otherwise.

    Returns
    -------
    list
        Variables found in the constraint expression.
    """
    return [
        var for var in identify_variables(con.body)
        if include_fixed or not var.fixed
    ]


def get_vars_in_cons(cons, include_fixed=True):
    """
    Get variables participating in the expresssions of a given
    sequence of constraint data objects.

    Parameters
    ----------
    cons : iterable of _ConstraintData
        Constrains of interest.
    include_fixed : bool, optional
        True to include fixed variables found in the
        constraint expressions, False otherwise.

    Returns
    -------
    list
        Variables participating in the constraint expressions
        of interest.
    """
    from itertools import chain
    return list(ComponentSet(chain(*tuple(
        get_vars_in_con(con, include_fixed=include_fixed)
        for con in cons
    ))))


def strip_state_var_bounds(
        model,
        dof_vars,
        vars_in_active_cons_only=True,
        include_fixed=False,
        ):
    """
    Strip state variable bounds.
    This also involves changing domains of the variable to
    `Reals`.
    """
    if vars_in_active_cons_only:
        state_vars = get_state_vars_in_active_cons(
            model=model,
            dof_vars=dof_vars,
            include_fixed=include_fixed,
            equality_only=False,
            include_vars_in_active_obj=True,
        )
    else:
        state_vars = get_state_vars(
            model=model,
            dof_vars=dof_vars,
            include_fixed=include_fixed,
        )

    for var in state_vars:
        var.setlb(None)
        var.setub(None)
        var.domain = pyo.Reals


def get_unused_state_vars(
        model,
        dof_vars,
        equality_only=False,
        include_fixed=True,
        ):
    """
    Obtain a model's 'unused' variables---that is,
    model state variables which do not participate
    in the model's equality coinstraints.

    Returns
    -------
    ComponentSet of VarData
        Unused state variables.
    """
    state_var_set = ComponentSet(
        get_state_vars(
            model,
            dof_vars,
            include_fixed=include_fixed,
        )
    )
    state_vars_in_active_con_set = ComponentSet(get_state_vars_in_active_cons(
        model,
        dof_vars,
        equality_only=equality_only,
        include_fixed=include_fixed,
    ))

    return state_var_set - state_vars_in_active_con_set


def cons_with_fixed_vars(model, equality_only=True):
    """
    Get constraints of model with no unfixed variables
    in their expressions.

    Return
    ------
    cons : list of ConstraintData
        Constraints of interest.
    """
    cons = []
    for con in model.component_data_objects(pyo.Constraint, active=True):
        if not equality_only or con.equality:
            unfixed_vars_in_con = [
                var
                for var in identify_variables(con.body, include_fixed=False)
            ]
            if not unfixed_vars_in_con:
                cons.append(con)

    return cons


def substitute_uncertain_params(
        m,
        substitution_map,
        include_var_like_exprs=True,
        logger=default_logger,
        **cdata_kwargs,
        ):
    """
    Perform expression replacement (substitution) in Constraint and
    Expression components.

    Arguments
    ---------
    m : ConcreteModel
        Model of interest.
    substitution_map: ComponentMap
        Mapping from the objects (mutable params) to be substituted
        to the objects to be replaced (Var objects or list of such).
    include_var_like_exprs : bool, optional
        True to force updates to named expressions of type
        `idaes.core.base.var_like_expression.VarLikeExpressionData`,
        False otherwise (in which case, the expressions are not updated).
    **kwargs : dict, optional
        Keyword arguments to `m.component_data_objects()`,
        excluding the `ctype` argument.
    """
    logger.debug(f"Invoked {substitute_uncertain_params.__name__}.")

    logger.debug("Assembling substitution dict...")
    vars_to_replace = []
    expr_replacement_map = dict()
    for dest, srcs in substitution_map.items():
        if not isinstance(srcs, (list, tuple)):
            srcs = [srcs]
        expr_replacement_map.update({id(src): dest for src in srcs})
        vars_to_replace.extend(srcs)

    logger.debug("Casting to component set")
    vars_to_replace_set = ComponentSet(vars_to_replace)
    # need uniqueness to ensure substitution is correct
    assert len(vars_to_replace) == len(vars_to_replace_set)

    # substitute for fixed vars in named expressions
    logger.debug("Iterating over named expressions...")
    for edata in m.component_data_objects(pyo.Expression, **cdata_kwargs):
        expr_has_vars_to_replace = (
            ComponentSet(identify_variables(edata.expr, include_fixed=True))
            & vars_to_replace_set
        )
        if expr_has_vars_to_replace:
            logger.debug(
                "Performing substitution in expression object with name "
                f"{edata.name!r}..."
            )
            new_expr = replace_expressions(
                expr=edata.expr,
                substitution_map=expr_replacement_map,
                descend_into_named_expressions=True,
                remove_named_expressions=False,
            )
            if not isinstance(edata, VarLikeExpressionData):
                edata.set_value(new_expr)
            elif include_var_like_exprs:
                # account for IDAES Var-like expressions
                edata.set_value(new_expr, force=True)

    # substitute params for fixed vars in all constraint expressions
    con_obj_types = (pyo.Constraint, pyo.Objective)
    logger.debug("Iterating over constraints and objectives...")
    for cdata in m.component_data_objects(con_obj_types, **cdata_kwargs):
        cdata_has_vars_to_replace = (
            ComponentSet(identify_variables(cdata.expr, include_fixed=True))
            & vars_to_replace_set
        )
        if cdata_has_vars_to_replace:
            logger.debug(
                f"Performing substitution in {type(cdata).__name__} object "
                f"with name {cdata.name!r}..."
            )
            new_expr = replace_expressions(
                expr=cdata.expr,
                substitution_map=expr_replacement_map,
                descend_into_named_expressions=True,
                remove_named_expressions=False,
            )
            cdata.set_value(new_expr)

    default_logger.debug("All done with substitutions.")


def validate_coordinate_grid(axial_coordinate_grid):
    """
    Check axial coordinate grid is such that:

    - all values between 0 and 1 inclusive.
    - all values unique

    Raises
    ------
    ValueError
        If coordinate grid does not specify conditions specified
        above.
    """
    if not all(0 <= val <= 1 for val in axial_coordinate_grid):
        raise ValueError(
            "Axial coordinate grid values should be all be in [0, 1]."
        )

    unique_coordinate_vals = np.unique(axial_coordinate_grid)
    if len(unique_coordinate_vals) != len(axial_coordinate_grid):
        raise ValueError(
            "Axial coordinate grid values should all be unique"
        )


def get_active_inequality_constraints(model, tol=1e-4):
    """
    Get active (in the mathematical sense) model inequality constraints.

    Parameters
    ----------
    model : BlockData
        Model of interest.
    tol : float, optional
        Tolerance. A constraint is considered active if
        its slack (`lslack()` or `uslack()`) is less than
        the tolerance.

    Returns
    -------
    active_lb_cons : list of ConstraintData
        Active lower bound constraints.
    active_ub_cons
        Active upper bound constraints.
    """
    active_lb_cons = []
    active_ub_cons = []
    for con in model.component_data_objects(pyo.Constraint, active=True):
        if not con.equality:
            if con.lb is not None and con.lslack() < tol:
                active_lb_cons.append(con)
            if con.ub is not None and con.uslack() < tol:
                active_ub_cons.append(con)

    return active_lb_cons, active_ub_cons


def get_violated_equality_constraints(blk, tol=1e-4):
    """
    Get equality constraints violated beyond a specified
    absolute tolerance.

    Parameters
    ----------
    blk : BlockData
        Block of which the constraints are to be checked.
    tol : 1e-4
        Feasibility tolerance.

    Returns
    -------
    violated_cons : list of ConstraintData
        Violated equality constraints.
    """
    violated_cons = []
    for con in blk.component_data_objects(pyo.Constraint, active=True):
        if con.equality and (con.lslack() < -tol or con.uslack() < -tol):
            violated_cons.append(con)

    return violated_cons


def get_active_var_bounds(
        model,
        vars_in_active_comps_only=True,
        tol=1e-4,
        include_fixed=False,
        ):
    """
    Get variables near bounds.

    Parameters
    ----------
    model : BlockData
        Model from which to retrieve variables.
    vars_in_active_comps_only : bool, optional
        True if only variables participating in active components
        are to be considered, False if all variables declared on
        the model (and its subblocks) are to be considered.
    tol : float, optional
        Tolerance. A variable is considered near its lower [upper]
        bound if value - lower bound [upper bound - value]
        is less than `tol`.
    include_fixed : bool
        True if fixed variables should be included, False otherwise.

    Returns
    -------
    active_lb_vars : list of VarData
        Variables near their lower bounds.
    active_ub_vars : list of VarData
        Variables near their upper bounds.
    """

    if vars_in_active_comps_only:
        vars = get_vars_from_components(
            block=model,
            ctype=(pyo.Objective, pyo.Constraint),
            active=True,
            include_fixed=include_fixed,
        )
    else:
        vars = (
            var for var in model.component_data_objects(pyo.Var)
            if not include_fixed or not var.fixed
        )

    active_lb_vars = []
    active_ub_vars = []
    for var in vars:
        lb_active = (
            var.lb is not None
            and var.value is not None
            and var.value - var.lb < tol
        )
        if lb_active:
            active_lb_vars.append(var)

        ub_active = (
            var.ub is not None
            and var.value is not None
            and var.ub - var.value < tol
        )
        if ub_active:
            active_ub_vars.append(var)

    return active_lb_vars, active_ub_vars


def compute_and_log_infeasibilities(model, logger=None, level=logging.DEBUG):
    """
    Log model infeasibilities. Variable bound, inequality constraint,
    and equality constraint violations are reported separately.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    max_var_infeas = max(
        (
            max(
                0,
                (var.lb or float("-inf")) - var.value,
                var.value - (var.ub or float("inf")),
            )
            for var in model.component_data_objects(pyo.Var)
            if not var.fixed
        ),
        default=0,
    )
    max_ineq_infeas = max(
        (
            max(0, -con.lslack(), -con.uslack())
            for con in model.component_data_objects(
                pyo.Constraint, active=True,
            )
            if not con.equality
        ),
        default=0,
    )
    max_eq_infeas = max(
        (
            max(0, -con.lslack(), -con.uslack())
            for con in model.component_data_objects(
                pyo.Constraint, active=True,
            )
            if con.equality
        ),
        default=0,
    )
    logger.log(level, msg=f" Max var. bound infeas. {max_var_infeas:.4e}")
    logger.log(level, msg=f" Max ineq. con. infeas. {max_ineq_infeas:.4e}")
    logger.log(level, msg=f" Max   eq. con. infeas. {max_eq_infeas:.4e}")

    return max_var_infeas, max_ineq_infeas, max_eq_infeas


def get_vars_with_close_bounds(
        model,
        vars_in_active_comps_only=True,
        tol=1e-4,
        include_fixed=False,
        ):
    """
    Get Vars with bounds that are equal (or near equal) in value.

    Parameters
    ----------
    model : BlockData
        Model from which to retrieve variables.
    vars_in_active_comps_only : bool, optional
        True if only variables participating in active components
        are to be considered, False if all variables declared on
        the model (and its subblocks) are to be considered.
    tol : float, optional
        Tolerance. The bounds are considered near equal
        if they differ by no more than `tol`.
    include_fixed : bool
        True if fixed variables should be included, False otherwise.

    Returns
    -------
    active_lb_vars : list of VarData
        Variables near their lower bounds.
    active_ub_vars : list of VarData
        Variables near their upper bounds.
    """
    if vars_in_active_comps_only:
        vars = get_vars_from_components(
            block=model,
            ctype=(pyo.Objective, pyo.Constraint),
            active=True,
            include_fixed=include_fixed,
        )
    else:
        vars = (
            var for var in model.component_data_objects(pyo.Var)
            if not include_fixed or not var.fixed
        )

    vars_with_close_bounds = []
    for var in vars:
        has_close_bounds = (
            var.lb is not None
            and var.ub is not None
            and abs(pyo.value(var.lb) - pyo.value(var.ub)) < tol
        )
        if has_close_bounds:
            vars_with_close_bounds.append(var)

    return vars_with_close_bounds


def float_to_str(val):
    """
    Convert floating point value to string.
    """
    return str(val).replace(".", "pt")


def get_co2_capture_str(model, min_capture_param):
    """
    Get a string corresponding to the minimum CO2 capture specification.
    """
    return (
        f"{float_to_str(pyo.value(min_capture_param))}"
        "_percent_capture"
    )


@contextmanager
def time_code(timer, code_block_name):
    """
    Start/stop timer around a code block.
    """
    timer.start(code_block_name)
    try:
        yield
    finally:
        timer.stop(code_block_name)


class ColoredFormatter(logging.Formatter):
    """
    Formatter for customizing console output colors
    at different logging levels.
    See https://misc.flogisoft.com/bash/tip_colors_and_formatting
    for a guide on customizing console colors.
    """

    grey = "\x1b[38;20m"
    dim_grey = "\x1b[2m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: dim_grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SolverWithBackup:
    """
    Metasolver which attempts to
    to solve a model with a sequence of subordinate optimizers
    until one subordinate optimizer returns an optimal termination.

    Parameters
    ----------
    *solvers : tuple of Pyomo solver type
        Subordinate solvers.
    logger : logging.Logger
        Progress logger.

    Attributes
    ----------
    solvers
    logger
    last_unsolved_model : None or Pyomo ConcreteModel
        Last model not solved to acceptable level.
    tt_timer : TicTocTimer
        Wallclock timer, used for timing subordinate solvers.
    """
    TIC_TOC_TIME_ATTR = "tic_toc_time"
    TOTAL_WALL_TIME_ATTR = "total_solve_wall_time"

    def __init__(self, *solvers, logger):
        """Initialize self (see class docstring).

        """
        # solvers
        self.solvers = list(solvers)
        assert self.solvers

        # set up logger and tic toc timer
        self.logger = logger
        self.tt_timer = TicTocTimer(logger=self.logger)

        # we want to keep track of last unsolved model
        self.last_unsolved_model = None

    def available(self, exception_flag=False):
        """
        Return true for availability check.
        """
        return True

    def version(self):
        return (0, 0, 0, 0)

    def _rate_termination_condition(self, termination_condition):
        """
        Assign an integer rating to a Pyomo termination condition.
        """
        is_globally_optimal = termination_condition in {
            termination_condition.globallyOptimal,
        }
        is_optimal = termination_condition in {
            termination_condition.optimal,
        }
        is_locally_optimal = termination_condition in {
            termination_condition.locallyOptimal,
        }
        is_feasible = termination_condition in {
            termination_condition.feasible,
        }
        if is_globally_optimal or is_optimal or is_locally_optimal:
            return 1
        elif is_feasible:
            return 2
        elif termination_condition in TerminationCondition:
            return 3
        else:
            raise TypeError(
                f"Argument `termination_condition` should be an instance"
                f"of Pyomo {TerminationCondition.__name__!r}, but got value "
                f"{termination_condition} of type "
                f"{type(termination_condition).__name__}."
            )

    def sort_termination_conditions(self, termination_conditions):
        """
        Sort iterable of TerminationCondition by rating
        and insertion order.
        """
        def rate_tc_idx_tup(tc_idx_tup):
            tc = tc_idx_tup[0]
            idx = tc_idx_tup[1]
            return self._rate_termination_condition(tc), idx

        return sorted(
            ((tc, idx) for idx, tc in enumerate(termination_conditions)),
            key=rate_tc_idx_tup,
        )

    def solve(
            self,
            model,
            acceptable_terminations=None,
            keep_model_if_not_solved=False,
            tee=False,
            load_solutions=True,
            **kwargs,
            ):
        """
        Solve model.

        Parameters
        ----------
        model : Pyomo _BlockData object
            Model of interest.
        acceptable_terminations : set or None, optional
            Acceptable solver termination conditions.
            If `None` passed, then termination is deemed
            acceptable provided that invoking
            ``pyomo.environ.check_optimal_termination()``
            on a solver results object returns True.
        keep_model_if_not_solved : bool, optional
            If termination not acceptable, True to clone model and assign
            to ``self.last_unsolved_model``, False otherwise.
        tee : bool, optional
            True to output subsolver logs as DEBUG-level logging
            messages through ``self.logger``, False to not output
            the subsolver logs at all.
        load_solutions : bool, optional
            True to load final solution(s) to model, False otherwise.
        **kwargs : dict, optional
            Additional optional arguments to the solver.

        Returns
        -------
        SolverResults
            Solver results.
        """
        if tee:
            solver_log_context_mgr = capture_output
            solver_log_context_kwds = dict(
                output=LogStream(level=logging.DEBUG, logger=self.logger)
            )
        else:
            solver_log_context_mgr = nullcontext
            solver_log_context_kwds = dict()

        exception_types = (
            ApplicationError, ValueError, RuntimeError, ArithmeticError,
        )

        last_unsolved_model_updated = False
        all_solve_results = []
        for idx, solver in enumerate(self.solvers):
            self.logger.debug(
                f"{self.__class__.__name__} invoking "
                f"solver {solver} (version {solver.version()}) "
                f"({idx + 1} of {len(self.solvers)})"
            )

            self.tt_timer.tic(msg=None)
            with solver_log_context_mgr(**solver_log_context_kwds):
                try:
                    res = solver.solve(
                        model,
                        tee=tee,
                        # solution loading is handled afterwards
                        load_solutions=False,
                        **kwargs,
                    )
                except exception_types as exc:
                    # Specialized exception handling. In particular:
                    # - ApplicationError may be due to IPOPT evaluation errors
                    # - ValueError, RuntimeError may be due to GAMS evaluation
                    #   errors or infeasibilities found during GAMS
                    #   preprocessing.
                    # - ArithmeticError may occur due to division by 0,
                    #   overflows, etc.
                    # Take note, and move to next solver
                    res = SolverResults()
                    res.solver.termination_condition = (
                        TerminationCondition.internalSolverError
                    )
                    res.solver.message = (
                        f"Encountered exception of type {type(exc).__name__} "
                        f"with message {str(exc)!r}"
                    )
                    self.logger.exception(
                        f"Solver {solver} ({idx + 1} of {len(self.solvers)}) "
                        "encountered exception: "
                    )

            setattr(
                res.solver, self.TIC_TOC_TIME_ATTR, self.tt_timer.toc(msg=None)
            )
            self.logger.debug(
                "Termination stats for solver "
                f"{solver} ({idx + 1} of {len(self.solvers)}): {res.solver}"
            )

            all_solve_results.append(res)

            # check termination status
            termination_acceptable = (
                pyo.check_optimal_termination(res)
                if acceptable_terminations is None
                else (
                    res.solver.termination_condition
                    in acceptable_terminations
                )
            )
            if termination_acceptable:
                break
            elif keep_model_if_not_solved and not last_unsolved_model_updated:
                # retain copy of model in state of self,
                # if desired
                self.last_unsolved_model = model.clone()
                last_unsolved_model_updated = True

        # as there may be multiple results objects (one per solver),
        # we rank them by termination condition rating and insertion
        # order, and choose the solution stored in the highest-ranked
        # results object as the 'final' one.
        ranked_res_tc_idx_pairs = self.sort_termination_conditions([
            result.solver.termination_condition
            for result in all_solve_results
        ])
        highest_ranked_res_idx = ranked_res_tc_idx_pairs[0][1]
        final_res = self._prepare_final_res(
            all_solve_results,
            final_res_idx=highest_ranked_res_idx,
        )
        self._shallow_copy_subsolver_results_attrs(
            final_res=final_res,
            subsolver_res=all_solve_results[highest_ranked_res_idx],
            subsolver=self.solvers[highest_ranked_res_idx],
        )

        if load_solutions:
            model.solutions.load_from(final_res)

        self.logger.debug(f"Done. Overall solve stats: {final_res.solver}")

        return final_res

    def _shallow_copy_subsolver_results_attrs(
            self,
            final_res,
            subsolver_res,
            subsolver,
            ):
        """
        Shallow copy desired attributes of `SolveResults`
        derived from invoking a subsolver of `self` on a model
        to `final_res`.
        """
        if hasattr(subsolver, "SOLVERBACKUP_RESULTS_SOLVER_ATTRS"):
            for attr_name in subsolver.SOLVERBACKUP_RESULTS_SOLVER_ATTRS:
                setattr(
                    final_res.solver,
                    attr_name,
                    copy.copy(getattr(subsolver_res.solver, attr_name)),
                )

    def _prepare_final_res(self, all_results, final_res_idx):
        """
        Prepare final results object.

        Parameters
        ----------
        all_results : list of SolverResults
            Solve results for each subordinate solver called.
        final_res_idx : int
            Index of results object in `all_results` for which
            solution and termination status are to be ultimately
            reported.

        Returns
        -------
        final_res : SolverResults
            Final solver results.
        """
        basis_res = all_results[final_res_idx]

        # get final results, copy over symbol maps
        final_res = SolverResults()
        final_res._smap = getattr(basis_res, "_smap", None)

        # prepare `solver` attribute
        # final_res.solver.all_results = all_results
        final_res.solver.termination_condition = (
            basis_res.solver.termination_condition
        )
        setattr(
            final_res.solver,
            self.TOTAL_WALL_TIME_ATTR,
            sum(
                getattr(sr.solver, self.TIC_TOC_TIME_ATTR)
                for sr in all_results
            ),
        )
        final_res.solver.status = basis_res.solver.status
        final_res.solver.message = basis_res.solver.message
        final_res.solver.subsolver_termination_conditions = [
            res.solver.termination_condition
            for res in all_results
        ]
        final_res.solver.subsolver_tic_toc_times = [
            getattr(res.solver, self.TIC_TOC_TIME_ATTR)
            for res in all_results
        ]

        # prepare `solution` attribute
        for basis_sol in basis_res.solution:
            final_sol = Solution()
            final_sol.status = basis_sol.status
            if hasattr(basis_sol, "_cuid"):
                final_sol._cuid = basis_sol._cuid
            for name in ["variable", "objective", "constraint", "problem"]:
                final_sol[name] = basis_sol[name].copy()
            final_res.solution.insert(final_sol)

        return final_res


class SolverWithScaling:
    """
    Wrapper around Pyomo solver that applies the scaling
    transformation to (a clone of) the model to be solved
    before invoking the optimizer.

    Parameters
    ----------
    solver : Pyomo solver object
        Subordinate solver.
    logger : logging.Logger
        Progress logger.
    """

    def __init__(self, solver, logger):
        """Initialize self (see class docstring)."""
        self.solver = solver
        self.logger = logger

    def version(self):
        return (0, 0, 0, 0)

    def solve(self, model, load_solutions=False, *args, **kwargs):
        """
        Solve a model.

        This method clones the model, applies the scaling
        transformation to the cloned model,
        solves the scaled model, inverse scales the solution,
        stores the inverse scaled solution in a results object,
        and, optionally, loads the solution to the original model.

        Parameters
        ----------
        model : Pyomo _BlockData object
            Model of interest.
        load_solutions : bool, optional
            True to load final solution(s) to model, False otherwise.
        *args : tuple, optional
            Positional arguments to ``self.solver.solve()``.
        **kwargs : dict, optional
            Keyword arguments to ``self.solver.solve()``.

        Returns
        -------
        SolverResults
            Solver results.
        """
        scaling_transformation = pyo.TransformationFactory("core.scale_model")

        self.logger.info(f"{type(self).__name__!r} cloning and scaling")
        model_clone = model.clone()
        scaled_model = scaling_transformation.create_using(
            model_clone,
            rename=False,
        )

        # solve the scaled model
        self.logger.info(
            f"{type(self).__name__!r} solving scaled model "
            f"with subordinate optimizer {self.solver} "
            f"(version {self.solver.version()})..."
        )
        scaled_results = self.solver.solve(
            scaled_model,
            load_solutions=False,
            *args,
            **kwargs,
        )

        # prepare a results object that is compatible with the
        # original model
        self.logger.info(
            f"{type(self).__name__!r} setting up final results object..."
        )
        final_results = SolverResults()
        final_results.problem.update(scaled_results.problem.copy())
        # roundabout, but this is a simple working method
        # to ensure the `solver` container is properly copied
        for key, val in dict(scaled_results.solver[0]).items():
            final_results.solver.declare(key)
            setattr(final_results.solver, key, val)

        if scaled_results.solution:
            # for now, we only support results with a single solution
            assert len(scaled_results.solution) == 1
            scaled_model.solutions.load_from(scaled_results)

            # inverse transform the scaled solution
            scale_transform = pyo.TransformationFactory("core.scale_model")
            for idx, sol in enumerate(scaled_model.solutions):
                scale_transform.propagate_solution(
                    scaled_model=scaled_model,
                    original_model=model_clone,
                )

            # finally, store unscaled solution.
            # NOTE: This does not work if `model_clone.solutions`
            #       is empty, but that is not applicable to the
            #       practical use cases throughout this codebase.
            try:
                model_clone.solutions.store_to(final_results)
            except ZeroDivisionError:
                self.logger.error(
                    "Could not load solution due to division by zero "
                    "evaluation error. This solution will not be stored "
                    "to the final results object."
                )

        if load_solutions:
            self.logger.info(
                f"{type(self).__name__!r} loading solution..."
            )
            model.solutions.load_from(final_results)

        self.logger.info(f"{type(self).__name__!r} all done.")

        return final_results
