from pyomo.core.base.var import Var
from pyomo.core.base.set import (UnindexedComponent_set, Set)
from pyomo.core.base.componentuid import ComponentUID
from pyomo.common.collections import ComponentSet
from pyomo.dae.flatten import flatten_components_along_sets

import numpy as np

def apply_function_elementwise(fcn, *args):
    """ 
    Given a "meshgrid," applies a function elementwise.
    Useful if my function does not support numpy data types.
    """
    # TODO: Numpy seems to have something called a ufunc, whose
    # purpose seems to be exactly this. Can I use these somehow?
    if not args:
        return fcn()
    try:
        assert all(len(args[0]) == len(arg) for arg in args)
    except TypeError:
        # Some arg does not have __len__ defined. Assume we should
        # treat all args as scalars.
        return fcn(*args)
    return list(
            apply_function_elementwise(fcn, *arg)
            for arg in zip(*args)
            )


def expand_singletons(arr):
    try:
        if len(arr) == 1:
            return expand_singletons(arr[0])
        else:
            return list(
                    expand_singletons(a)
                    for a in arr)
    except TypeError:
        # arr is a scalar
        return arr


def _get_structured_variable_data(sets, comps, dereference=False):
    mesh = np.meshgrid(*sets, indexing="ij")
    # Without "ij", for two dimensions, meshgrid returns what
    # I would consider the transpose of the proper mesh.

    # TODO: references get tricky. If we have flattened
    # "base components" using references, we would like
    # to deference to get the underlying slices.
    # If we have flattened references, however, we may
    # or may not want to get the underlying slice.
    #
    # For now, we only dereference once, to undo the layer
    # of references added by the flattener. The ability
    # to completely dereference may be valuable in the future...
    set_cuids = [str(ComponentUID(s)) for s in sets]
    indices = [[val for val in s] for s in sets]
    cuid_buffer = {}
    if dereference:
        # TODO: Should we branch on whether each component is a reference
        # rather than use some "dereference" flag?
        variables = {
                str(ComponentUID(comp.referent, cuid_buffer=cuid_buffer)):
                    apply_function_elementwise(
                        lambda *ind: comp[ind].value,
                        # This function, which gives us the data to be stored,
                        # will vary with component type, and could possibly
                        # by user provided/configurable.
                        *mesh,
                        )
                    for comp in comps
                }
    else:
        variables = {
                str(ComponentUID(comp, cuid_buffer=cuid_buffer)):
                    apply_function_elementwise(
                        lambda *ind: comp[ind].value,
                        *mesh,
                        )
                    for comp in comps
                }

    return {
            "sets": set_cuids,
            "indices": indices,
            "variables": variables,
            }


def _get_unstructured_variable_data(comps):
    # TODO: If we decide to support "complete dereferencing,"
    # we will need a dereference flag here as well.
    cuid_buffer = {}
    return {
        "sets": None,
        "indices": None,
        "variables": {str(ComponentUID(comp, cuid_buffer)): comp.value
            for comp in comps},
        }


def _get_structured_variable_data_from_dict(sets, indices, comps_dict):
    """
    Arguments
    ---------
    sets: Tuple of Pyomo Sets
        Sets by which the components are indexed.
    comps_dict: Dict of dicts.
        Maps ComponentUID to a dict mapping indices to values.

    """
    mesh = np.meshgrid(*indices, indexing="ij")
    return {
            # TODO: Why am I storing sets and indices separately instead of one
            # "sets" field that contains a list of (name, indices) tuples?
            "sets": [str(ComponentUID(s)) for s in sets],
            "indices": [list(s) for s in indices],
            "variables": {
                name: apply_function_elementwise(
                    lambda *idx: data[idx],
                    *mesh,
                    )
                for name, data in comps_dict.items()
                },
            }


def get_structured_variables_from_model(
        model,
        sets,
        indices=None,
        flatten_vars=None,
        ):
    if indices is None:
        indices = tuple(next(iter(s)) for s in sets)
    ctype = Var

    if isinstance(sets, Set):
        # sets is a scalar (a single set)
        # TODO: What regularity assumptions must we impose on these sets?
        sets = (sets,)
    set_set = ComponentSet(sets)

    # In a general data structure, we should have a flag for whether
    # to flatten the model...
    if flatten_vars is None:
        # Flattening may be a nontrivial expense. If we are doing this in a
        # loop, we would like a way to avoid it in this function.
        sets_list, comps_list = flatten_components_along_sets(
                model,
                sets,
                ctype,
                indices,
                )
    else:
        sets_list, comps_list = flatten_vars

    data = dict()
    # Our struct may contain metadata, so we add a field for the model
    data["model"] = []
    model = data["model"]
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            model.append(_get_unstructured_variable_data(comps))
        else:
            # Dereference should switch depending on if we have flattened
            # our model.
            model.append(
                _get_structured_variable_data(sets, comps, dereference=True)
                )

    return data
