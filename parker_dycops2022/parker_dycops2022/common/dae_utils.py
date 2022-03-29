import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap

from pyomo.dae.flatten import slice_component_along_sets


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
