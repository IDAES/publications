import enum
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr.visitor import identify_variables
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface


def consolidate_category_dicts(cat1, cat2, union, intersect):
    """
    I have not set up categorization to accept multiple sets, because
    it is not clear what this would mean (look for diff wrt either set
    or both?). This function combines two category dicts by taking
    the union of their differential/derivative components and the
    intersection of their algebraic components.
    """
    category_dict = dict()
    accounted_for = ComponentSet()
    for categ in union:
        # combine differential/derivative/discretization components
        category_dict[categ] = cat1[categ] + cat2[categ]
        accounted_for.update(category_dict[categ])

    for categ in intersect:
        alg_set1 = ComponentSet(cat1[categ])
        category_dict[categ] = [comp for comp in cat2[categ]
                if comp in alg_set1 and comp not in accounted_for]

    return category_dict


class VariableCategory(enum.Enum):
    DIFFERENTIAL = 1
    ALGEBRAIC = 2
    DERIVATIVE = 3
    INPUT = 4
    FIXED = 5
    SCALAR = 6
    UNUSED = 7
    DISTURBANCE = 8
    MEASUREMENT = 9


class ConstraintCategory(enum.Enum):
    DIFFERENTIAL = 1
    ALGEBRAIC = 2
    DISCRETIZATION = 3
    INPUT = 4
    INITIAL = 5
    TERMINAL = 6
    SENSITIVITY = 7
    SCALAR = 8
    UNUSED = 9
    INEQUALITY = 10


def _is_derivative_wrt(var, time):
    parent = var.parent_component()
    if not isinstance(parent, DerivativeVar):
        return False
    else:
        return time in ComponentSet(parent.get_continuousset_list())


def _get_state_vardata(deriv):
    parent = deriv.parent_component()
    index = deriv.index()
    return parent.get_state_var()[index]


DAE_DISC_SUFFIX = "_disc_eq"


def _get_disc_eq(deriv):
    parent = deriv.parent_component()
    parent_block = parent.parent_block()
    index = deriv.index()
    name = parent.local_name + DAE_DISC_SUFFIX
    disc = parent_block.component(name)
    return disc[index]


def _identify_derivative_if_differential(condata, wrt, include_fixed=False):
    parent = condata.parent_component()
    if parent.local_name.endswith("_disc_eq"):
        return False, None
    deriv = None
    for var in identify_variables(condata.expr, include_fixed=include_fixed):
        if _is_derivative_wrt(var, wrt):
            if deriv is None:
                deriv = var
            else:
                raise RuntimeError(
                "Categorization currently requires only one differential "
                "variable to be present in each differential equation. "
                "Got %s and %s in constraint %s."
                % (deriv.name, var.name, condata.name)
                )
    is_diff = False if deriv is None else True
    return is_diff, deriv


def categorize_dae_variables_and_constraints(
        model,
        dae_vars,
        dae_cons,
        time,
        index=None,
        input_vars=None,
        disturbance_vars=None,
        input_cons=None,
        active_inequalities=None,
        ):
    VC = VariableCategory
    CC = ConstraintCategory
    # Index that we access when we need to work with a specific data
    # object. This would be less necessary if constructing CUIDs was
    # efficient, or if we could do the equivalent of `identify_variables`
    # in a templatized constraint.
    if index is not None:
        t1 = index
    else:
        # Use the first non-initial time point as a "representative
        # index." Don't use get_finite_elements so this will be valid
        # for general ordered sets.
        t1 = time[2]

    if input_vars is None:
        input_vars = []
    if input_cons is None:
        input_cons = []
    if disturbance_vars is None:
        disturbance_vars = []
    if active_inequalities is None:
        active_inequalities = []

    # We will check these sets to determine which components
    # are inputs and disturbances.
    #
    # NOTE: Specified input vars/cons and disturbance vars should be
    # in the form of components indexed only by time. The user can
    # accomplish this easily with the `Reference` function.
    #
    input_var_set = ComponentSet(inp[t1] for inp in input_vars)
    disturbance_var_set = ComponentSet(dist[t1] for dist in disturbance_vars)
    input_con_set = ComponentSet(inp[t1] for inp in input_cons)
    active_inequality_set = ComponentSet(con[t1] for con in active_inequalities)

    # Filter vars and cons for duplicates.
    #
    # Here we assume that if any two components refer to the same
    # data object at our "representative index" t1, they are
    # effectively "the same" components, and do not need to both
    # be included.
    #
    visited = set()
    filtered_vars = []
    duplicate_vars = []
    for var in dae_vars:
        _id = id(var[t1])
        if _id not in visited:
            visited.add(_id)
            filtered_vars.append(var)
        else:
            duplicate_vars.append(var)
    filtered_cons = []
    duplicate_cons = []
    for con in dae_cons:
        _id = id(con[t1])
        if _id not in visited:
            visited.add(_id)
            filtered_cons.append(con)
        else:
            duplicate_cons.append(con)
    dae_vars = filtered_vars
    dae_cons = filtered_cons

    # Filter out inputs and disturbances. These are "not variables"
    # for the sake of having a square DAE model.
    dae_vars = [var for var in dae_vars if var[t1] not in input_var_set
            and var[t1] not in disturbance_var_set]
    dae_cons = [con for con in dae_cons if con[t1] not in input_con_set
            and (con[t1].equality or con[t1] in active_inequality_set)]

    dae_map = ComponentMap()
    dae_map.update(
            (var[t1], var) for var in dae_vars
            )
    dae_map.update(
            (con[t1], con) for con in dae_cons
            )

    diff_eqn_map = ComponentMap()
    for con in dae_cons:
        condata = con[t1]
        is_diff, deriv = _identify_derivative_if_differential(condata, time)
        if is_diff:
            diff_eqn_map[deriv] = condata

    potential_deriv = []
    potential_diff_var = []
    potential_disc = []
    potential_diff_eqn = []
    for var in dae_vars:
        vardata = var[t1]
        if vardata in diff_eqn_map:
            # This check ensures that vardata is differential wrt time
            # and participates in exactly one non-discretization equation.
            # This equation is that derivative's "differential equation."
            diff_vardata = _get_state_vardata(vardata)
            if diff_vardata in dae_map:
                # May not be the case if diff_var is an input...
                potential_diff_var.append(dae_map[diff_vardata])
                potential_diff_eqn.append(dae_map[diff_eqn_map[vardata]])
                potential_deriv.append(var)
                potential_disc.append(dae_map[_get_disc_eq(vardata)])

    # PyNumero requires exactly one objective on the model.
    dummy_obj = False
    if len(list(model.component_objects(Objective, active=True))) == 0:
        dummy_obj = True
        model._temp_dummy_obj = Objective(expr=0)

    igraph = IncidenceGraphInterface()
    variables = [var[t1] for var in dae_vars]
    constraints = [con[t1] for con in dae_cons]

    present_cons = [con for con in constraints if con.active]
    active_var_set = ComponentSet(var for con in present_cons
            for var in identify_variables(con.expr, include_fixed=False))
    present_vars = [var for var in variables if var in active_var_set]

    # Filter out fixed vars and inactive constraints.
    # We could do this check earlier (before constructing igraph)
    # by just checking var.fixed and con.active...
    #_present_vars = [var for var in variables if not var.fixed]
    #_present_cons = [con for con in constraints if con.active]
    #present_vars = [var for var in variables if var in _nlp._vardata_to_idx]
    #present_cons = [con for con in constraints if con in _nlp._condata_to_idx]

    var_block_map, con_block_map = igraph.block_triangularize(
            present_vars,
            present_cons,
            )
    derivdatas = []
    diff_vardatas = []
    discdatas = []
    diff_condatas = []
    for deriv, disc, diff_var, diff_con in zip(potential_deriv, potential_disc,
            potential_diff_var, potential_diff_eqn):
        derivdata = deriv[t1]
        discdata = disc[t1]
        diff_vardata = diff_var[t1]
        diff_condata = diff_con[t1]
        # Check:
        if (
                # a. Variables are actually used (not fixed), and
                #    constraints are active
                derivdata in var_block_map and
                diff_vardata in var_block_map and
                discdata in con_block_map and
                diff_condata in con_block_map and
                # b. The diff var can be matched with the disc eqn and 
                #    the deriv var can be matched with the diff eqn.
                (var_block_map[diff_vardata] == con_block_map[discdata]) and
                (var_block_map[derivdata] == con_block_map[diff_condata])
                ):
            # Under these conditions, assuming the Jacobian of the diff eqns
            # with respect to the derivatives is nonsingular, a sufficient
            # condition for nonsingularity (of the submodel with fixed inputs
            # at t1) is that the Jacobian of algebraic variables with respect
            # to algebraic equations is nonsingular.
            derivdatas.append(derivdata)
            diff_vardatas.append(diff_vardata)
            discdatas.append(discdata)
            diff_condatas.append(diff_condata)

    derivs = [dae_map[vardata] for vardata in derivdatas]
    diff_vars = [dae_map[vardata] for vardata in diff_vardatas]
    discs = [dae_map[condata] for condata in discdatas]
    diff_cons = [dae_map[condata] for condata in diff_condatas]

    not_alg_set = ComponentSet(derivdatas+diff_vardatas+discdatas+diff_condatas)
    alg_vars = []
    unused_vars = []
    for vardata in variables:
        var = dae_map[vardata]
        if vardata not in var_block_map:
            unused_vars.append(var)
        elif vardata not in not_alg_set:
            alg_vars.append(var)
        # else var is differential, derivative, input, or disturbance
    alg_cons = []
    unused_cons = []
    for condata in constraints:
        con = dae_map[condata]
        if condata not in con_block_map:
            unused_cons.append(con)
        elif condata not in not_alg_set:
            alg_cons.append(con)
        # else con is differential or discretization (or a constraint
        # on inputs)

    if dummy_obj:
        model.del_component(model._temp_dummy_obj)

    var_category_dict = {
            VC.INPUT: input_vars,
            VC.DIFFERENTIAL: diff_vars,
            VC.DERIVATIVE: derivs,
            VC.ALGEBRAIC: alg_vars,
            VC.DISTURBANCE: disturbance_vars,
            VC.UNUSED: unused_vars,
            }
    con_category_dict = {
            CC.INPUT: input_cons,
            CC.DIFFERENTIAL: diff_cons,
            CC.DISCRETIZATION: discs,
            CC.ALGEBRAIC: alg_cons,
            CC.UNUSED: unused_cons,
            }
    return var_category_dict, con_category_dict
