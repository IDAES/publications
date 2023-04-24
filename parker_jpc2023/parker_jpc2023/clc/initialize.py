from parker_jpc2023.clc.model import (
    make_model,
)
from parker_jpc2023.clc.dae_utils import (
    generate_discretization_components_along_set,
    generate_diff_deriv_disc_components_along_set,
)

import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components
from pyomo.contrib.incidence_analysis import (
    IncidenceGraphInterface,
    solve_strongly_connected_components,
)
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
from pyomo.common.timing import HierarchicalTimer
from pyomo.util.subsystems import (
    TemporarySubsystemManager,
)
"""
"""


TIMER = HierarchicalTimer()


def set_default_design_vars(m):
    design_vars = {
            "fs.moving_bed.bed_diameter": 6.5,
            "fs.moving_bed.bed_height": 5.0,
            }
    design_vars = fix_design_variables(m, design_vars)
    return design_vars


def set_default_inlet_conditions(m):
    time = m.fs.time
    dynamic_inputs = {
        "fs.moving_bed.gas_inlet.flow_mol[*]": 128.20513,
        "fs.moving_bed.gas_inlet.pressure[*]": 2.00,
        "fs.moving_bed.gas_inlet.temperature[*]": 298.15,
        "fs.moving_bed.gas_inlet.mole_frac_comp[*,CO2]": 0.02499,
        "fs.moving_bed.gas_inlet.mole_frac_comp[*,H2O]": 0.00001,
        "fs.moving_bed.gas_inlet.mole_frac_comp[*,CH4]": 0.97500,

        "fs.moving_bed.solid_inlet.flow_mass[*]": 591.4,
        "fs.moving_bed.solid_inlet.temperature[*]": 1183.15,
        "fs.moving_bed.solid_inlet.particle_porosity[*]": 0.27,
        "fs.moving_bed.solid_inlet.mass_frac_comp[*,Fe2O3]": 0.45,
        "fs.moving_bed.solid_inlet.mass_frac_comp[*,Fe3O4]": 1e-9,
        "fs.moving_bed.solid_inlet.mass_frac_comp[*,Al2O3]": 0.55,
        }
    inputs = fix_dynamic_inputs(m, time, dynamic_inputs)
    return inputs


def set_optimal_design_vars(m):
    """
    These values of design variables were determined by
    Chinedu's 2020 paper in Powder Technology
    """
    design_vars = {
            "fs.moving_bed.bed_diameter": 4.58,
            "fs.moving_bed.bed_height": 4.60,
            }
    design_vars = fix_design_variables(m, design_vars)
    return design_vars


def set_optimal_inlet_conditions(m):
    """
    These inlet values were determined by Chinedu's 2020 paper
    in Powder Technology
    """
    time = m.fs.time
    dynamic_inputs = {
            "fs.moving_bed.gas_inlet.flow_mol[*]": 128.0,
            "fs.moving_bed.gas_inlet.pressure[*]": 2.50,
            "fs.moving_bed.gas_inlet.temperature[*]": 293.0,

            "fs.moving_bed.solid_inlet.flow_mass[*]": 549.0,
            "fs.moving_bed.solid_inlet.temperature[*]": 1183.0,
            }
    inputs = fix_dynamic_inputs(m, time, dynamic_inputs)
    return inputs


def set_values_to_inlets(m):
    # This is a special case of an "initialize_to_boundary_conditions"
    # method, that would need some knowledge of what axes to propagate
    # boundary conditions along
    gas_names = set_gas_values_to_inlets(m)
    solid_names = set_solid_values_to_inlets(m)
    return gas_names+solid_names


def set_gas_values_to_inlets(m):
    time = m.fs.time
    t0 = time.first()

    inlet_names = [
        "fs.moving_bed.gas_inlet.flow_mol[%s]" % t0,
        "fs.moving_bed.gas_inlet.pressure[%s]" % t0,
        "fs.moving_bed.gas_inlet.temperature[%s]" % t0,
        "fs.moving_bed.gas_inlet.mole_frac_comp[%s,CO2]" % t0,
        "fs.moving_bed.gas_inlet.mole_frac_comp[%s,H2O]" % t0,
        "fs.moving_bed.gas_inlet.mole_frac_comp[%s,CH4]" % t0,
        ]

    var_names = [
        "fs.moving_bed.gas_phase.properties[*,*].flow_mol",
        "fs.moving_bed.gas_phase.properties[*,*].pressure",
        "fs.moving_bed.gas_phase.properties[*,*].temperature",
        "fs.moving_bed.gas_phase.properties[*,*].mole_frac_comp[CO2]",
        "fs.moving_bed.gas_phase.properties[*,*].mole_frac_comp[H2O]",
        "fs.moving_bed.gas_phase.properties[*,*].mole_frac_comp[CH4]",
        ]

    for name, inlet_name in zip(var_names, inlet_names):
        var = m.find_component(name)
        inlet_var = m.find_component(inlet_name)
        for vardata in var.values():
            vardata.set_value(pyo.value(inlet_var))

    return var_names

def set_solid_values_to_inlets(m):
    time = m.fs.time
    t0 = time.first()

    inlet_names = [
        "fs.moving_bed.solid_inlet.flow_mass[%s]" % t0,
        "fs.moving_bed.solid_inlet.temperature[%s]" % t0,
        #"fs.moving_bed.solid_inlet.particle_porosity[%s]" % t0,
        "fs.moving_bed.solid_inlet.mass_frac_comp[%s,Fe2O3]" % t0,
        "fs.moving_bed.solid_inlet.mass_frac_comp[%s,Fe3O4]" % t0,
        "fs.moving_bed.solid_inlet.mass_frac_comp[%s,Al2O3]" % t0,
        ]

    # FIXME: Either both components or neither should be indexed
    # by time.
    var_names = [
        "fs.moving_bed.solid_phase.properties[*,*].flow_mass",
        "fs.moving_bed.solid_phase.properties[*,*].temperature",
        #"fs.moving_bed.solid_phase.properties[*,*].particle_porosity",
        "fs.moving_bed.solid_phase.properties[*,*].mass_frac_comp[Fe2O3]",
        "fs.moving_bed.solid_phase.properties[*,*].mass_frac_comp[Fe3O4]",
        "fs.moving_bed.solid_phase.properties[*,*].mass_frac_comp[Al2O3]",
        ]

    for name, inlet_name in zip(var_names, inlet_names):
        var = m.find_component(name)
        inlet_var = m.find_component(inlet_name)
        for vardata in var.values():
            vardata.set_value(pyo.value(inlet_var))

    return var_names


def set_gas_temperature_to_solid_inlet(m, include_fixed=False):
    time = m.fs.time
    gas = m.fs.moving_bed.gas_phase
    space = m.fs.moving_bed.gas_phase.length_domain
    for t, x in time*space:
        var = gas.properties[t, x].temperature
        if not var.fixed or include_fixed:
            var.set_value(m.fs.moving_bed.solid_inlet.temperature[t].value)


def add_inlet_objective(m):
    # Could get these components by calling flatten_dae_components
    # on inlet variables, I suppose...
    gas_inlet_names = [
            "fs.moving_bed.gas_inlet.flow_mol[*]",
            "fs.moving_bed.gas_inlet.pressure[*]",
            "fs.moving_bed.gas_inlet.temperature[*]",
            "fs.moving_bed.gas_inlet.mole_frac_comp[*,CO2]",
            "fs.moving_bed.gas_inlet.mole_frac_comp[*,H2O]",
            "fs.moving_bed.gas_inlet.mole_frac_comp[*,CH4]",
            ]
    solid_inlet_names = [
            "fs.moving_bed.solid_inlet.flow_mass[*]",
            "fs.moving_bed.solid_inlet.temperature[*]",
            "fs.moving_bed.solid_inlet.particle_porosity[*]",
            "fs.moving_bed.solid_inlet.mass_frac_comp[*,Fe2O3]",
            "fs.moving_bed.solid_inlet.mass_frac_comp[*,Fe3O4]",
            "fs.moving_bed.solid_inlet.mass_frac_comp[*,Al2O3]",
            ]
    inlet_names = gas_inlet_names + solid_inlet_names

    time = m.fs.time

    inlet_vars = [m.find_component(name) for name in inlet_names]

    # TODO:
    # - Scaling factors for these terms
    # - Allow "setpoint" to be something other than current values
    obj_expr = sum((var[t] - var[t].value)**2 for t in time for var in inlet_vars)

    m._inlet_objective = pyo.Objective(expr=obj_expr)

    return m._inlet_objective


def initialize_steady(m, calc_var_kwds=None):
    """
    This is my approach for initializing the steady state model.
    Set state variables to their inlet values, except gas temperature,
    which is initialized to the temperature of the solid inlet.
    Then deactivate discretization equations and strongly connected
    component decomposition.
    """
    if calc_var_kwds is None:
        calc_var_kwds = {}
    gas_inlet_names = set_gas_values_to_inlets(m)
    solid_inlet_names = set_solid_values_to_inlets(m)
    set_gas_temperature_to_solid_inlet(m)

    gas_phase = m.fs.moving_bed.gas_phase
    solid_phase = m.fs.moving_bed.solid_phase
    gas_length = m.fs.moving_bed.gas_phase.length_domain
    solid_length = m.fs.moving_bed.solid_phase.length_domain
    gas_disc_eqs = [con for _, con, _ in
            generate_discretization_components_along_set(m, gas_length)]
    solid_disc_eqs = [con for _, con, _ in
            generate_discretization_components_along_set(m, solid_length)]

    gas_sum_eqn_slice = gas_phase.properties[:,:].sum_component_eqn
    gas_sum_eqn_slice.attribute_errors_generate_exceptions = False
    solid_sum_eqn_slice = solid_phase.properties[:,:].sum_component_eqn
    solid_sum_eqn_slice.attribute_errors_generate_exceptions = False

    to_deactivate = []
    to_deactivate.extend(gas_sum_eqn_slice)
    to_deactivate.extend(solid_sum_eqn_slice)
    to_deactivate.extend(gas_disc_eqs)
    to_deactivate.extend(solid_disc_eqs)

    to_fix = []
    for name in gas_inlet_names+solid_inlet_names:
        ref = m.find_component(name)
        to_fix.extend(ref.values())

    with TemporarySubsystemManager(to_fix=to_fix, to_deactivate=to_deactivate):
        solve_strongly_connected_components(m, calc_var_kwds=calc_var_kwds)


def initialize_steady_without_solid_temperature(m):
    """
    """
    gas_inlet_names = set_gas_values_to_inlets(m)
    solid_inlet_names = set_solid_values_to_inlets(m)

    gas_phase = m.fs.moving_bed.gas_phase
    solid_phase = m.fs.moving_bed.solid_phase
    gas_length = m.fs.moving_bed.gas_phase.length_domain
    solid_length = m.fs.moving_bed.solid_phase.length_domain
    gas_disc_eqs = [con for _, con, _ in
            generate_discretization_components_along_set(m, gas_length)]
    solid_disc_eqs = [con for _, con, _ in
            generate_discretization_components_along_set(m, solid_length)]

    gas_sum_eqn_slice = gas_phase.properties[:,:].sum_component_eqn
    gas_sum_eqn_slice.attribute_errors_generate_exceptions = False
    solid_sum_eqn_slice = solid_phase.properties[:,:].sum_component_eqn
    solid_sum_eqn_slice.attribute_errors_generate_exceptions = False

    to_deactivate = []
    to_deactivate.extend(gas_sum_eqn_slice)
    to_deactivate.extend(solid_sum_eqn_slice)
    to_deactivate.extend(gas_disc_eqs)
    to_deactivate.extend(solid_disc_eqs)

    to_fix = []
    for name in gas_inlet_names+solid_inlet_names:
        ref = m.find_component(name)
        to_fix.extend(ref.values())

    with TemporarySubsystemManager(to_fix=to_fix, to_deactivate=to_deactivate):
        solve_strongly_connected_components(m)


def initialize_dynamic_from_steady(
        m_dyn, m_steady, flattened=None
        ):
    """
    """
    time = m_dyn.fs.time
    time_steady = m_steady.fs.time
    t_steady = time_steady.first()

    TIMER.start("get derivs")
    diff_deriv_disc_list = list(
        generate_diff_deriv_disc_components_along_set(m_dyn, time)
    )
    derivs = [var for _, var, _ in diff_deriv_disc_list]
    TIMER.stop("get derivs")

    TIMER.start("flatten dynamic")
    if flattened is None:
        scalar_vars, dae_vars = flatten_dae_components(m_dyn, time, pyo.Var)
    else:
        scalar_vars, dae_vars = flattened
    TIMER.stop("flatten dynamic")
    TIMER.start("flatten steady")
    steady_scalar_vars, steady_dae_vars = flatten_dae_components(
        m_steady, time_steady, pyo.Var
    )
    TIMER.stop("flatten steady")

    deriv_cuid_set = set(str(pyo.ComponentUID(var.referent)) for var in derivs)
    scalar_value_map = dict(
        (str(pyo.ComponentUID(var)), var.value) for var in steady_scalar_vars
    )
    dae_value_map = dict(
        (str(pyo.ComponentUID(var.referent)), var[t_steady].value)
        for var in steady_dae_vars
    )

    TIMER.start("set values")
    for var in scalar_vars:
        cuid = str(pyo.ComponentUID(var))
        var.set_value(scalar_value_map[cuid])
    for var in dae_vars:
        cuid = str(pyo.ComponentUID(var.referent))
        if cuid in dae_value_map:
            for t in time:
                var[t].set_value(dae_value_map[cuid])
        else:
            assert cuid in deriv_cuid_set
            # TODO: Better way of initializing derivatives
            for t in time:
                var[t].set_value(0.0)
    TIMER.stop("set values")

    return scalar_vars, dae_vars


if __name__ == "__main__":
    pass
