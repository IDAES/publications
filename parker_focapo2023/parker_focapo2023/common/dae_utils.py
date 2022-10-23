import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
#from pyomo.core.expr.relational_expr import EqualityExpression
# relational_expr doesn't exist in Pyomo 6.4.2
from pyomo.core.expr.current import EqualityExpression

from pyomo.dae.flatten import (
    slice_component_along_sets,
    flatten_dae_components,
)

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.interface import (
    _generate_variables_in_constraints,
)


DAE_DISC_SUFFIX = "_disc_eq"


def _get_count_of_components(components):
    count = ComponentMap((c, 0) for c in components)
    for c in components:
        count[c] += 1
    return count


# FIXME:
# Having is_derivative return a tuple is a little weird.
# In this way, bool(is_derivative(var)) is always True,
# and we need bool(is_derivative(var)[0]).
# Should probably separate the "is_derivative" functionality
# from the "get_wrt_sets" functionality.
def is_derivative(var, wrt=None, require_all=True):
    """
    Utility function for checking whether a component is a DerivativeVar,
    possibly with respect to any or all specified sets.

    """
    if not isinstance(var, dae.DerivativeVar):
        # If we're not a DerivativeVar, return False without doing
        # any work.
        return False, tuple()

    if wrt is None:
        wrt = tuple()
    elif not isinstance(wrt, tuple):
        wrt = (wrt,)
    set_count = _get_count_of_components(wrt)

    csets = var.get_continuousset_list()
    wrt_sets = tuple(s for s in csets if s in set_count)
    wrt_set_count = _get_count_of_components(wrt_sets)

    if (len(set_count) == len(wrt_set_count)
            and all(set_count[s] <= wrt_set_count[s] for s in wrt_sets)):
        # We are differential wrt all the sets the user asked for, at
        # least as many times as the user specified.
        # This covers the case of wrt=None.
        return True, wrt_sets

    elif wrt_sets and not require_all:
        # We are differential wrt at least one of the sets the user
        # specified.
        return True, wrt_sets

    else:
        # Either require_all is True and we have not matched every
        # set provided, or require_all is False and we have not
        # matched any set
        return False, wrt_sets


def is_derivative_data(vardata, wrt=None, require_all=True):
    """
    Utility function for checking whether a data object "is a derivative,"
    possibly with respect to any or all specified sets.

    """
    var = vardata.parent_component()
    return is_derivative(var, wrt=wrt, require_all=require_all)


def contains_derivative_data(
        var, wrt=None, require_all_sets=True, require_all=False, identical=True
        ):
    # TODO: Naming of require_all_sets and require_all might be confusing.
    # - require_all here refers to whether all data objects should be
    #   "derivative data" objects
    # - require_all_sets refers to whether we require the derivative data
    #   objects to be derivatives with respect to all specified sets
    """
    """
    found_non_deriv = False
    cached_wrt_sets = None
    for data in var.values():
        is_deriv, wrt_sets = is_derivative_data(
            data, wrt=wrt, require_all=require_all_sets
        )
        if is_deriv:
            if not require_all:
                return is_deriv, wrt_sets
            elif identical:
                # Return False if the "wrt sets" have changed
                if cached_wrt_sets is None:
                    cached_wrt_sets = _get_count_of_components(wrt_sets)
                elif cached_wrt_sets != _get_count_of_components(wrt_sets):
                    return False, tuple()
        elif require_all:
            return is_deriv, tuple()
        else:
            found_non_deriv = True
    # Getting to the end of the loop means either we found only derivatives
    # and require_all, or we found no derivatives and not require_all.
    if found_non_deriv:
        return False, tuple()
    elif identical:
        return True, wrt_sets
    else:
        return True, tuple()


def generate_discretization_components_along_set(m, set_, active=True):
    # What is a "discretization equation with respect to 's'"?
    # Is it the component itself, or reference-to-slices of the
    # component along set s?
    #
    # If we consider a "variable" to be something that maps time
    # to some data, then a "differential variable" or "discretization
    # equation" should also map time to some data...
    for var in m.component_objects(pyo.Var, active=active):
        if (isinstance(var, dae.DerivativeVar)
                and set_ in ComponentSet(var.get_continuousset_list())
                ):
            block = var.parent_block()
            # NOTE: Making some assumptions about the name and location
            # of discretization equations.
            con = block.find_component(var.local_name + DAE_DISC_SUFFIX)
            var_map = dict(
                    (idx, pyo.Reference(slice_))
                    for idx, slice_ in slice_component_along_sets(var, (set_,))
                    )
            con_map = dict(
                    (idx, pyo.Reference(slice_))
                    for idx, slice_ in slice_component_along_sets(con, (set_,))
                    )
            if not active or con.active:
                # We do not check the individual condata objects for activity...
                for idx, var in var_map.items():
                    yield idx, con_map[idx], var


def generate_diff_deriv_disc_components_along_set(m, set_, active=True):
    for var in m.component_objects(pyo.Var, active=active):
        if (isinstance(var, dae.DerivativeVar)
                and set_ in ComponentSet(var.get_continuousset_list())
                ):
            block = var.parent_block()
            con = block.find_component(var.local_name + DAE_DISC_SUFFIX)
            state = var.get_state_var()
            deriv_map = dict(
                (idx, pyo.Reference(slice_))
                for idx, slice_ in slice_component_along_sets(var, (set_,))
            )
            disc_map = dict(
                (idx, pyo.Reference(slice_))
                for idx, slice_ in slice_component_along_sets(con, (set_,))
            )
            state_map = dict(
                (idx, pyo.Reference(slice_))
                for idx, slice_ in slice_component_along_sets(state, (set_,))
            )
            if not active or con.active:
                for idx, deriv in deriv_map.items():
                    yield state_map[idx], deriv, disc_map[idx]


def _filter_duplicates(objs):
    seen = set()
    for obj in objs:
        id_ = id(obj)
        if id_ not in seen:
            seen.add(id_)
            yield obj


class DifferentialHelper(object):

    def __init__(self, model, time, indices=None, active=True):
        t0 = time.first()
        t1 = time.next(t0)
        tf = time.last()

        if indices is None:
            # By default use t1 as the "reference index" for descending
            # into indexed blocks. This is because (unfortunately) components
            # are often skipped at the boundaries.
            indices = ComponentMap([(time, t1)])

        scalar_vars, dae_vars = flatten_dae_components(
            model, time, pyo.Var, indices=indices
        )
        scalar_cons, dae_cons = flatten_dae_components(
            model, time, pyo.Constraint, indices=indices
        )
        self._scalar_vars = scalar_vars
        self._dae_vars = dae_vars
        self._scalar_cons = scalar_cons
        self._dae_cons = dae_cons

        diff_deriv_disc_list = list(
            generate_diff_deriv_disc_components_along_set(
                model, time, active=active
            )
        )
        self._diff_deriv_disc_list = diff_deriv_disc_list

    def get_subsystem_at_time(self, t, include_inequality=False):
        """
        Returns the subsystem of VarData and (equality) ConstraintData
        objects at a particular point in time. The variables are only those
        unfixed and contained in active constraints.

        """
        constraints = list(_filter_duplicates(
            con[t] for con in self._dae_cons
            if t in con and con[t].active
            and (
                include_inequality or
                isinstance(con[t].expr, EqualityExpression)
            )
        ))
        var_set = ComponentSet(
            _generate_variables_in_constraints(constraints, include_fixed=False)
        )
        variables = list(_filter_duplicates(
            var[t] for var in self._dae_vars if var[t] in var_set
        ))
        return variables, constraints

    def get_valid_diff_deriv_disc_at_time(self, t):
        """
        Returns a list of tuples of time-indexed differential
        variables, derivative variables, and discretization equations.
        These must be valid in the sense that discretization equations
        can be matched with differential variables in a maximum matching
        of variables and equations. If a differential variable does not
        satisfy this criteria, it should not be fixed as an initial condition
        and we do not consider it "valid."

        Note that the returned components are time-indexed. Their "validity"
        is only checked at the point in time specified.

        """
        variables, constraints = self.get_subsystem_at_time(t)
        return self._get_valid_diff_deriv_disc(t, variables, constraints)

    def _get_valid_diff_deriv_disc(self, t, variables, constraints):
        # We could potentially avoid repeating work at each time point
        # if we cache the "valid differential components" for a given
        # structural incidence matrix.
        # This would be analagous to identifying which time points have
        # subsystems with a different structure. (Would want to use the
        # row and col arrays of a COO matrix as keys.)
        igraph = IncidenceGraphInterface()

        var_dmp, con_dmp = igraph.dulmage_mendelsohn(variables, constraints)

        var_block_map, con_block_map = igraph.block_triangularize(
            var_dmp.square, con_dmp.square
        )

        valid = []
        for state, deriv, disc in self._diff_deriv_disc_list:
            if (
                state[t] in var_block_map
                and deriv[t] in var_block_map
                and disc[t] in con_block_map
            ):
                # All three components are part of the square subsystem.
                # We can make some statement about whether they can be
                # matched with each other.
                #
                # TODO: What if they are in the underconstrained subsystem?
                if var_block_map[state[t]] == con_block_map[disc[t]]:
                    valid.append((state, deriv, disc))
        return valid

    def get_differential_subsystem_at_time(self, t, include_inequality=False):
        """
        Returns the valid derivative variables and differential equations
        at a point in time. First we identify the valid derivatives, then
        we identify constraints that contain these derivatives.

        """
        # Note that these are data objects
        variables, constraints = self.get_subsystem_at_time(
            t, include_inequality=include_inequality
        )

        # Note that these are time-indexed components, not data objects
        diff_deriv_disc_list = self._get_valid_diff_deriv_disc(
            t, variables, constraints
        )
        deriv_var_data, diff_cons, _ = self._get_differential_constraints(
            t, diff_deriv_disc_list, constraints
        )
        return deriv_var_data, diff_cons

    def _get_differential_constraints(
        self, t, diff_deriv_disc_list, constraints
    ):
        deriv_vars = [var for _, var, _ in diff_deriv_disc_list]
        deriv_var_data = [var[t] for var in deriv_vars]

        # Note that I can't get the time-indexed differential equations
        # without assuming that the equations have the same structure
        # at every point in time. The solution is to only call this
        # method at a few points in time - only those with different
        # incidence matrix structure.
        diff_eqns = get_constraints_containing_variables(
            deriv_var_data, constraints
        )

        # Filter out discretization equations:
        disc_eq_set = ComponentSet(eq[t] for _, _, eq in diff_deriv_disc_list)
        diff_eqns = [eq for eq in diff_eqns if eq not in disc_eq_set]
        return deriv_var_data, diff_eqns, disc_eq_set

    def get_naive_differential_subsystem_at_time(
        self, t, include_inequality=False
    ):
        # Var and Constraint data objects at this point in time.
        variables, constraints = self.get_subsystem_at_time(
            t, include_inequality=include_inequality
        )
        var_set = ComponentSet(variables)
        diff_deriv_disc_list = [
            (diff, deriv, disc)
            for diff, deriv, disc in self._diff_deriv_disc_list
            if deriv[t] in var_set
        ]
        # Get constraints that contain differential variables
        deriv_var_data, diff_cons, _ = self._get_differential_constraints(
            t, diff_deriv_disc_list, constraints
        )
        return deriv_var_data, diff_cons

    def get_naive_algebraic_subsystem_at_time(
        self, t, include_inequality=False
    ):
        # Var and Constraint data objects at this point in time.
        variables, constraints = self.get_subsystem_at_time(
            t, include_inequality=include_inequality
        )
        var_set = ComponentSet(variables)
        con_set = ComponentSet(constraints)
        _diff = self.get_naive_differential_subsystem_at_time(t)
        deriv_vars, diff_eqns = _diff
        diff_vars = [
            var[t] for var, _, _ in self._diff_deriv_disc_list
            if var[t] in var_set
        ]
        diff_deriv_var_set = ComponentSet(deriv_vars + diff_vars)
        disc_eqns = [
            eq[t] for _, _, eq in self._diff_deriv_disc_list
            if eq[t] in con_set
        ]
        diff_disc_con_set = ComponentSet(disc_eqns + diff_eqns)
        alg_vars = [var for var in variables if var not in diff_deriv_var_set]
        alg_cons = [con for con in constraints if con not in diff_disc_con_set]
        return alg_vars, alg_cons

    def get_algebraic_subsystem_at_time(self, t, include_inequality=False):
        """
        Get the algebraic subsystem at a point in time. The algebraic
        subsystem is defined as the variables and (equality) constraints
        that are not valid differential variables, differential equations,
        derivative variables, or discretization equations.

        """
        # Note that these are data objects:
        variables, constraints = self.get_subsystem_at_time(
            t, include_inequality=include_inequality
        )
        # Note that these are indexed components:
        diff_deriv_disc_list = self._get_valid_diff_deriv_disc(
            t, variables, constraints
        )
        deriv_vars, diff_eqns, disc_eq_set = self._get_differential_constraints(
            t, diff_deriv_disc_list, constraints
        )
        diff_vars = [var[t] for var, _, _ in diff_deriv_disc_list]
        diff_deriv_var_set = ComponentSet(diff_vars + deriv_vars)
        diff_disc_con_set = ComponentSet(diff_eqns) | disc_eq_set
        alg_vars = [var for var in variables if var not in diff_deriv_var_set]
        alg_cons = [con for con in constraints if con not in diff_disc_con_set]
        return alg_vars, alg_cons


def get_constraints_containing_variables(variables, constraints):
    """
    Returns any ConstraintData objects from the provided list that
    contain at least one of the VarData objects provided.

    """
    # NOTE: This could potentially be more efficient if we had a cached
    # incidence graph, maybe with sets of adjacent nodes already computed.
    # E.g.
    #   if igraph.adjacent(con).union(var_set)
    var_set = ComponentSet(variables)
    filtered_constraints = [
        con for con in constraints
        if any(
            var in var_set for var in 
            identify_variables(con.expr, include_fixed=False)
        )
    ]
    return filtered_constraints
