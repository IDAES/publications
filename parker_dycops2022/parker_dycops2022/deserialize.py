import json
from pyomo.core.base.componentuid import ComponentUID
from pyomo.util.slices import slice_component_along_sets

# NOTE: Here we are accessing files in the working directory.
# This is probably not what we want.
# Should probably use os.dirname(__file__) instead...
VARIANCE_FILE = "variance.json"
SCALING_FACTOR_FILE = "scaling_factor.json"


def expand_data_over_space(m, time, space, data):
    """
    data maps time/space cuids to values.
    We get a data object from the cuid, then generate slices
    over time only at every point in space.
    """
    sets = (time,)
    t0 = next(iter(time))
    new_data = {}
    for name, value in data.items():
        var = m.find_component(name)
        if var is None:
            # Steady model won't have accumulation variables
            assert "accumulation" in name
            continue
        factor_sets = list(var.index_set().subsets())
        if var.is_indexed() and len(factor_sets) == 2:
            # Assume our var is indexed by time, then space.
            for x in space:
                cuid = ComponentUID(
                    slice_component_along_sets(var[t0, x], sets)
                )
                new_data[str(cuid)] = value
        else:
            new_data[name] = value
    return new_data


def get_variance_of_time_slices(m, time, space):
    data = get_variance_data()
    return expand_data_over_space(m, time, space, data)


def get_scaling_of_time_slices(m, time, space):
    data = get_scaling_factor_data()
    return expand_data_over_space(m, time, space, data)


def get_variance_data():
    with open(VARIANCE_FILE, "r") as fp:
        data = json.load(fp)
    return data


def get_scaling_factor_data():
    with open(SCALING_FACTOR_FILE, "r") as fp:
        data = json.load(fp)
    return data
