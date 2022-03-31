import pyomo.environ as pyo

from pyomo.common.collections import ComponentMap
from pyomo.dae.flatten import flatten_components_along_sets
from pyomo.util.slices import slice_component_along_sets

def get_max_values_from_steady(m):
    time = m.fs.time
    assert len(time) == 1
    t0 = next(iter(time))
    gas_length = m.fs.MB.gas_phase.length_domain
    solid_length = m.fs.MB.solid_phase.length_domain
    sets = (time, gas_length, solid_length)
    sets_list, comps_list = flatten_components_along_sets(m, sets, pyo.Var)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is time and sets[1] is gas_length:
            gas_comps = comps
        elif len(sets) == 2 and sets[0] is time and sets[1] is solid_length:
            solid_comps = comps
    variables = gas_comps + solid_comps
    max_values = ComponentMap(
        (
            var,
            max(abs(data.value) for data in var.values()
                if data.value is not None),
        )
        for var in variables
    )

    # Maps time indexed vars to their max values over space
    max_values_time = {}
    for var in variables:
        for x in gas_length:
            sliced = slice_component_along_sets(var[t0, x], (time,))
            max_values_time[str(pyo.ComponentUID(sliced))] = max_values[var]

    return max_values_time
