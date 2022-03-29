import numpy as np
from pyomo.core.base.set import Set
from parker_dycops2022.common.serialize.data_from_model import (
        apply_function_elementwise,
        _get_structured_variable_data_from_dict,
        )


def subtract_variable_data(data1, data2):
    """
    Subtract variable values in second data dict from those in first
    """
    sets1 = data1["sets"]
    indices1 = data1["indices"]
    variables1 = data1["variables"]

    sets2 = data2["sets"]
    indices2 = data2["indices"]
    variables2 = data2["variables"]

    assert sets1 == sets2
    assert indices1 == indices2
    assert set(variables1) == set(variables2)

    data = dict(data1)
    data["variables"] = {
            name: (
                np.array(variables1[name]) - np.array(variables2[name])
                ).tolist()
            for name in variables1
            }

    return data


def multiply_variable_data_by_scalars(data, scalars):
    """
    Arguments
    ---------
    data: List of dicts
        Contains variable names and values partitioned by indexing set
    scalars: Dict
        Maps variable name (CUID) to value

    """
    data = dict(data)
    variables = data["variables"]
    for name, values in variables.items():
        val_array = np.array(values)
        scalar = scalars[name]
        variables[name] = (scalar * val_array).tolist()
    return data


def multiply_variable_data(data1, data2):
    """
    Multiplies, elementwise, the values in two groups of structured variables.
    """
    sets1 = data1["sets"]
    indices1 = data1["indices"]
    variables1 = data1["variables"]

    sets2 = data2["sets"]
    indices2 = data2["indices"]
    variables2 = data2["variables"]

    assert sets1 == sets2
    assert indices1 == indices2
    assert set(variables1) == set(variables2)

    data = dict(data1)
    data["variables"] = {
            name: (
                np.array(variables1[name]) * np.array(variables2[name])
                ).tolist()
            for name in variables1
            }
    return data


def abs_variable_data(data):
    """
    """
    data = dict(data)
    variables = data["variables"]
    data["variables"] = {
            name: np.abs(np.array(variables[name])).tolist()
            for name in variables
            }
    return data


def sum_variable_data(data):
    """
    Note that this operation "squashes" the structure of the variables
    and therefore just returns a dict.
    """
    return {
            name: np.sum(np.array(values)).tolist()
            for name, values in data["variables"].items()
            }


def max_variable_data(data):
    """
    Note that this operation "squashes" the structure of the variables
    and therefore just returns a dict.
    """
    return {
            name: np.max(np.array(values)).tolist()
            for name, values in data["variables"].items()
            }


def concatenate_data_along_set(
        data1, data2, set_, offset=None, use_first=True,
        ):
    """
    We are concatenating two pieces of time series data.
    Each piece of data already has a time set, and we expect
    these sets to start at zero. In this case we need to apply
    some offset to the second so it comes "after" the first.
    We also expect the first entry of the second series to
    correspond to the last entry of the first series.
    In this case we need to choose which data series to use
    for this point. The default is to use the first data series.
    """
    if isinstance(set_, Set):
        set_name = set_.name
    else:
        set_name = set_

    sets1 = data1["sets"]
    indices1 = data1["indices"]
    variables1 = data1["variables"]

    sets2 = data2["sets"]
    indices2 = data2["indices"]
    variables2 = data2["variables"]

    assert sets1 == sets2
    concat_data = dict(data1)
    if sets1 is None:
        return concat_data

    non_time_indices1 = [
        vals for name, vals in zip(sets1, indices1) if name != set_name
    ]
    non_time_indices2 = [
        vals for name, vals in zip(sets2, indices2) if name != set_name
    ]
    assert non_time_indices1 == non_time_indices2

    # Values of the set we're interested in
    set_values1 = [
        (i, vals) for i, (name, vals) in enumerate(zip(sets1, indices1))
        if name == set_name
    ]
    set_values2 = [
        (i, vals) for i, (name, vals) in enumerate(zip(sets2, indices2))
        if name == set_name
    ]
    assert len(set_values1) == 1 and len(set_values2) == 1
    set_loc1, set_values1 = set_values1[0]
    set_loc2, set_values2 = set_values2[0]
    assert set_loc1 == set_loc2
    set_loc = set_loc1

    # Apply offset:
    # Here I am performing floating point arithmetic on the time set.
    # I should be aware of this if I use these values as keys in the
    # future.
    # TODO: Should I get the offset automatically from the last point
    # in the set_values1?
    if offset is None:
        offset = set_values1[-1]
    set_values2 = [v + offset for v in set_values2]

    slice_idx = [slice(None) for _ in sets1]

    # Choose the value of the time series
    if use_first:
        concat_indices = set_values1 + set_values2[1:]

        # Index to exclude the first entry of the second time set
        slice_idx[set_loc] = slice(1, None, None)
        slice_idx = tuple(slice_idx)

        concat_variables = {
            name: np.concatenate(
                (
                    np.array(variables1[name]),
                    np.array(variables2[name])[slice_idx],
                ),
                axis=set_loc,
            ).tolist()
            for name in variables1
        }
    else:
        concat_indices = set_values1[:-1] + set_values2

        # Index to exclude the last entry of the first time set
        slice_idx[set_loc] = slice(None, -1, None)
        slice_idx = tuple(slice_idx)

        concat_variables = {
            name: np.concatenate(
                (
                    np.array(variables1[name])[slice_idx],
                    np.array(variables2[name]),
                ),
                axis=set_loc,
            ).tolist()
            for name in variables1
        }

    concat_data["indices"][set_loc] = concat_indices
    concat_data["variables"] = concat_variables

    # Some "testing" code
    #if len(sets1) == 2:
    #    for name, concat_values in concat_data["variables"].items():
    #        values1 = variables1[name]
    #        values2 = variables2[name]
    #        print(name)
    #        print("data1")
    #        for vals in values1:
    #            print(vals)
    #        print("data2")
    #        for vals in values2:
    #            print(vals)
    #        print("concatenated")
    #        for vals in concat_values:
    #            print(vals)

    return concat_data
