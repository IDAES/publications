import bisect

from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.expr.numvalue import value
from pyomo.dae.flatten import get_slice_for_set
from pyomo.util.slices import slice_component_along_sets

"""
Our data structure for a time-varying input is:
{
    str(cuid): {(t0, t1): val, ...}
}
"""


def interval_data_from_time_series(data, use_left_value=False):
    """
    in:
    (
        [t0, ...],
        {
            str(cuid): [value0, ...],
        },
    )
    out:
    {
        str(cuid): {(t0, t1): value0 or value1, ...},
    }
    """
    # TODO: use_left_value seems like a bad name
    time, variable_data = data
    n_t = len(time)
    if n_t == 1:
        t0 = time[0]
        return {
            name: {(t0, t0): values[0]}
            for name, values in variable_data.items()
        }
    else:
        # This covers the case of n_t > 1 and n_t == 0
        interval_data = {}
        intervals = [(time[i-1], time[i]) for i in range(1, n_t)]
        for name, values in variable_data.items():
            interval_values = [
                values[i-1] if use_left_value else values[i]
                for i in range(1, n_t)
            ]
            interval_data[name] = dict(zip(intervals, interval_values))
        return interval_data


def initialize_time_series_data(variables, time, t0=None):
    """
    Initializes time series data from initial conditions
    """
    # TODO: What is the right API for this function?
    if t0 is None:
        # time could be a list, OrderedSet, or ContinuousSet
        t0 = next(iter(time))
    variable_data = {}
    for var in variables:
        if var.is_reference():
            try:
                variable_data[str(ComponentUID(var.referent))] = [value(var[t0])]
            except ValueError as err:
                if "No value for uninitialized NumericValue" in str(err):
                    variable_data[str(ComponentUID(var.referent))] = [None]
                else:
                    raise err

        else:
            try:
                variable_data[str(ComponentUID(var))] = [value(var[t0])]
            except ValueError as err:
                if "No value for uninitialized NumericValue" in str(err):
                    variable_data[str(ComponentUID(var))] = [None]
                else:
                    raise err
    return ([t0], variable_data)


def extend_time_series_data(
        data,
        variables,
        time,
        offset=None,
        include_first=False,
        include_last=True,
        ):
    """
    data:
    (
        [t0, t1, ...],
        {
            str(cuid): [value0, value1, ...],
        },
    )
    """
    existing_time, existing_name_map = data
    time = list(time)
    start = None if include_first else 1
    stop = None if include_last else -1
    slice_idx = slice(start, stop, None)
    time = time[slice_idx]
    variable_data = {}
    for var in variables:
        if var.is_reference():
            variable_data[str(ComponentUID(var.referent))] = [
                value(var[t]) for t in time
            ]
        else:
            variable_data[str(ComponentUID(var))] = [
                value(var[t]) for t in time
            ]
    if not set(variable_data.keys()) == set(existing_name_map.keys()):
        raise ValueError(
            "Trying to extend time series with different variables "
            "than were originally present."
        )
    new_name_map = {
        name: existing_name_map[name] + variable_data[name]
        for name in existing_name_map
    }
    if offset is not None:
        time = [t + offset for t in time]
    if existing_time[-1] >= time[0]:
        raise ValueError(
            "Cannot extend a data series with duplicated or "
            "unsorted time entries. Trying to place %s after %s."
            % (time[0], existing_time[-1])
        )
    new_time = existing_time + time
    return (new_time, new_name_map)


# Some prototype code for extracting inputs on an interval.
# I realized that this is more complicated than I thought and
# decided not to finish it, but this would be useful if I could
# finish it.
# What makes this complicated is (a) doing binary search on intervals
# (maybe with a tolerance?) and (b) the fact that our input dict
# may not be dense.
#
#def get_inputs_on_interval(input_dict, interval):
#    return {
#        name: get_values_on_interval(inputs, interval)
#        for name, inputs in input_dict.items()
#    }
#
#
#def get_values_on_interval(inputs, interval):
#    """
#    inputs: dict
#        Maps interval tuples to values
#    interval: tuple
#        A tuple containing low value, high value
#
#    Returns: dict
#        Maps intervals to values
#
#    """
#    input_intervals = list(sorted(inputs.keys()))
#
#    low, high = interval
#    low_index = binary_search_on_intervals(
#        input_intervals, low, use_right=True
#    )
#    high_index = binary_search_on_intervals(
#        input_intervals, high, use_right=False
#    )
#
#    # Find the input interval that contains the low and high
#    # values of our "target interval"
#    if low_index is None:
#        raise ValueError(
#            "Lower bound of target interval, %s, is not included "
#            "in the intervals we searched." % low_index
#        )
#    if high_index is None:
#        raise ValueError(
#            "Upper bound of target interval, %s, is not included "
#            "in the intervals we searched." % high_index
#        )
#
#    if low_index == high_index:
#        # The interval we want is contained in a single "input interval"
#        target_input_values = {interval: inputs[input_intervals[low_index]]}
#
#    else:
#        assert low_index < high_index
#        target_input_values = {}
#
#        # Assign the "upper part" of the interval containing the
#        # new lower bound
#        lb_low, ub_low = input_intervals[low_index]
#        target_input_values[low, ub_low] = inputs[lb_low, ub_low]
#
#        # Assign any intermediate intervals, which are unchanged
#        for i in range(low_index + 1, high_index):
#            interval = input_intervals[i]
#            target_input_values[interval] = inputs[interval]
#
#        # Assign the "lower part" of the interval containing the
#        # new upper bound
#        lb_high, ub_high = input_intervals[high_index]
#        target_input_values[lb_high, high] = inputs[lb_high, ub_high]
#
#    return target_input_values


def assert_disjoint_intervals(intervals):
    # FIXME: This does not properly recognize multiple of the same singleton
    # as not disjoint.
    for i, (lo, hi) in enumerate(sorted(intervals)):
        if not lo <= hi:
            raise ValueError(
                "Lower endpoint of interval is higher than upper endpoint"
            )
        if i != 0:
            _, prev_hi = intervals[i-1]
            if not prev_hi <= lo:
                raise ValueError(
                    "Intervals not disjoint"
                )


def load_inputs_into_model(model, time, input_dict, time_tol=0):
    for cuid, inputs in input_dict.items():
        var = model.find_component(cuid)
        assert var is not None

        intervals = list(sorted(inputs.keys()))
        assert_disjoint_intervals(intervals)
        for i, interval in enumerate(intervals):
            idx0 = time.find_nearest_index(interval[0], tolerance=time_tol)
            idx1 = time.find_nearest_index(interval[1], tolerance=time_tol)
            if idx0 is None or idx1 is None:
                continue
            # While loop to avoid second binary search?
            input_val = inputs[interval]
            idx_iter = range(idx0+1, idx1+1) if idx0 != idx1 else (idx0,)
            for idx in idx_iter:
                t = time.at(idx)
                var[t].set_value(input_val)


def get_inputs_at_time(input_dict, t):
    """
    Converts "series of intervals" data to scalar data
    """
    scalar_input_dict = {}
    for name, inputs in input_dict.items():
        intervals = list(inputs.keys())
        assert_disjoint_intervals(intervals)
        # This is a lazy way to find the input at a point in time.
        # Really we should use binary search...
        scalar_input_dict[name] = None
        for i, (lo, hi) in enumerate(intervals):
            if lo < t and t <= hi:
                scalar_input_dict[name] = inputs[lo, hi]
                break
    return scalar_input_dict


def set_values_at_time(model, t, data):
    """
    Loads scalar data into the model at specified points in time.
    """
    try:
        t_points = tuple(t)
    except TypeError:
        t_points = (t,)
    for name, val in data.items():
        var = model.find_component(name)
        for t in t_points:
            var[t].set_value(val)


def set_values(model, data):
    """
    data:
    {
        str(cuid): value
    }
    """
    for name, val in data.items():
        var = model.find_component(name)
        var.set_value(val)


def get_values_from_model_at_time(model, t, names):
    data = {
        name: model.find_component(name)[t].value
        for name in names
    }
    return data


# NOTE: Working with variables is nice, but they do not make
# reliable keys in a ComponentMap, so getting something like
# "scalar data" is difficult.
def copy_values_from_time(variables, time, t0, include_fixed=True):
    """
    Useful for initializing from initial conditions.
    Maybe the more general thing to do is
    (a) get scalar data from a model at a point in time
    (b) set values at specified (all) points in time from
        the scalar data.
    """
    for var in variables:
        for t in time:
            if include_fixed or not var[t].fixed:
                var[t].set_value(var[t0].value)


def get_tracking_cost_expression(
        variables,
        time,
        setpoint_data,
        weight_data=None,
        dereference=True,
        ):
    """
    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression.
    setpoint_data: dict
        Maps variable names to setpoint values
    weight_data: dict
        Maps variable names to tracking cost weights

    Returns
    -------
    Pyomo expression. Sum of weighted squared difference between
    variables and setpoint values.

    """
    # TODO: Should setpoints be mutable params? Indexed by variable names?
    # This is tempting as it makes changing the setpoint easy.
    variable_names = [
        # NOTE: passing strings through ComponentUID.
        # This may break things that rely on a particular form of the string.
        # I believe this approach is more consistent, however.
        # This is a temporary measure before I switch to using CUIDs as keys.
        str(ComponentUID(str(get_time_indexed_cuid(var)))) for var in variables
    ]
    #    str(ComponentUID(var))
    #    if not var.is_reference() or not dereference
    #    else str(ComponentUID(var.referent))
    #    for var in variables
    #]
    if weight_data is None:
        weight_data = {name: 1.0 for name in variable_names}

    def tracking_rule(m, t):
        return sum(
            weight_data[name] * (var[t] - setpoint_data[name])**2
            for name, var in zip(variable_names, variables)
        )
    # Note that I don't enforce that variables are actually indexed
    # by time. "time" could just be a list of sample points...
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr


def get_time_indexed_cuid(var, sets=None, dereference=None):
    """
    Arguments
    ---------
    var:
        Object to process
    time: Set
        Set to use if slicing a vardata object
    dereference: None or int
        Number of times we may access referent attribute to recover a
        "base component" from a reference.

    """
    # TODO: Does this function have a good name?
    # Should this function be generalized beyond a single indexing set?
    # Is allowing dereference to be an integer worth the confusion it might
    # add?
    if dereference is None:
        # Does this branch make sense? If given an unattached component,
        # we dereference, otherwise we don't dereference.
        remaining_dereferences = int(var.parent_block() is None)
    else:
        remaining_dereferences = int(dereference)
    if var.is_indexed():
        if var.is_reference() and remaining_dereferences:
            remaining_dereferences -= 1
            referent = var.referent
            if isinstance(referent, IndexedComponent_slice):
                return ComponentUID(referent)
            else:
                # If dereference is None, we propagate None, dereferencing
                # until we either reach a component attached to a block
                # or reach a non-reference component.
                dereference = dereference if dereference is None else\
                        remaining_dereferences
                # NOTE: Calling this function recursively
                return get_time_indexed_cuid(
                    referent, time, dereference=dereference
                )
        else:
            # Assume that var is indexed only by time
            # TODO: Get appropriate slice for indexing set.
            # TODO: Should we call slice_component_along_sets here as well?
            # To cover the case of b[t0].var, where var is indexed
            # by a set we care about, and we also care about time...
            # But then maybe we should slice only the sets we care about...
            # Don't want to do anything with these sets unless we're
            # presented with a vardata...
            #index = tuple(
            #    get_slice_for_set(s) for s in var.index_set().subsets()
            #)
            index = get_slice_for_set(var.index_set())
            return ComponentUID(var[:])
    else:
        if sets is None:
            raise ValueError(
                "A ComponentData %s was provided but no set. We need to know\n"
                "what set this component should be indexed by."
                % var.name
            )
        slice_ = slice_component_along_sets(var, sets)
        return ComponentUID(slice_)


def find_nearest_index(array, target, tolerance=None):
    # array needs to be sorted and we assume it is zero-indexed
    lo = 0
    hi = len(array)
    #arr = list(array)
    i = bisect.bisect_right(array, target, lo=lo, hi=hi)
    # i is the index at which target should be inserted if it is to be
    # right of any equal components. 

    if i == lo:
        # target is less than every entry of the set
        #nearest_index = i + 1
        #delta = self.at(nearest_index) - target
        nearest_index = i
        delta = array[nearest_index] - target
    elif i == hi:
        # target is greater than or equal to every entry of the set
        #nearest_index = i
        #delta = target - self.at(nearest_index)
        nearest_index = i - 1
        delta = target - array[nearest_index]
    else:
        # p_le <= target < p_g
        # delta_left = target - p_le
        # delta_right = p_g - target
        # delta = min(delta_left, delta_right)
        # Tie goes to the index on the left.
        delta, nearest_index = min(
            #(abs(target - self.at(j)), j) for j in [i, i+1]
            (abs(target - array[j]), j) for j in [i-1, i]
        )

    if tolerance is not None:
        if delta > tolerance:
            return None
    return nearest_index
