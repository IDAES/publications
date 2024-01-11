##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
This module contains the code necessary to set up a minimal example
of the moving bed reduction CLC model
"""
import pyomo.environ as pyo
from pyomo.util.slices import slice_component_along_sets
from pyomo.core.expr.visitor import replace_expressions

from idaes.core import FlowsheetBlock, EnergyBalanceType

from idaes.models_extra.gas_solid_contactors.unit_models.moving_bed import (
    MBR as MovingBed,
)
from idaes.models_extra.gas_solid_contactors.properties.methane_iron_OC_reduction import (
    GasPhaseParameterBlock,
    SolidPhaseParameterBlock,
    HeteroReactionParameterBlock,
)
from idaes.core.util.model_statistics import degrees_of_freedom


def make_model(
    dynamic=False,
    ntfe=10,
    ntcp=1,
    tfe_width=15.0,
    nxfe=10,
):
    """Make a square reduction reactor model.

    Does not do any initialization, scaling, or set up anything necessary
    for the optimization problem.

    """
    if dynamic:
        # This is the *model* horizon
        horizon = ntfe * tfe_width

    # Create spatial grid
    # Why is this necessary? Shouldn't it be the default?
    xfe_list = [1.0*i/nxfe for i in range(nxfe+1)]

    # Create time grid
    if dynamic:
        time_list = [0.0, horizon]
    else:
        time_list = [0.0]

    # Create top-level and flowsheet models
    if dynamic:
        name = "Dynamic MB reduction CLC flowsheet model"
    else:
        name = "Steady-state MB reduction CLC flowsheet model"
    model = pyo.ConcreteModel(name=name)
    model.fs = FlowsheetBlock(
        dynamic=dynamic,
        time_set=time_list,
        time_units=pyo.units.s,
    )

    # Set up thermo props and reaction props
    model.fs.gas_properties = GasPhaseParameterBlock()
    model.fs.solid_properties = SolidPhaseParameterBlock()
    model.fs.hetero_reactions = HeteroReactionParameterBlock(
        solid_property_package=model.fs.solid_properties,
        gas_property_package=model.fs.gas_properties,
    )

    mb_config = {
        "finite_elements": nxfe,
        "has_holdup": True,
        "length_domain_set": xfe_list,
        "transformation_method": "dae.finite_difference",
        "gas_transformation_scheme": "BACKWARD",
        "solid_transformation_scheme": "FORWARD",
        "pressure_drop_type": "ergun_correlation",
        "gas_phase_config": {
            "property_package": model.fs.gas_properties,
        },
        "solid_phase_config": {
            "property_package": model.fs.solid_properties,
            "reaction_package": model.fs.hetero_reactions,
        },
    }

    # Create MovingBed unit model
    model.fs.moving_bed = MovingBed(**mb_config)

    if dynamic:
        # Apply time-discretization
        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(
            model, wrt=model.fs.time, nfe=ntfe, ncp=ntcp, scheme='LAGRANGE-RADAU'
        )

    # Note that we need to fix inputs/disturbances *before* fixing initial,
    # conditions as we rely on having square subsystems to identify diff eqns.
    fix_dof(model)

    if dynamic:
        # Fix initial conditions
        t0 = model.fs.time.first()
        length = model.fs.moving_bed.length_domain
        for x in length:
            if x != length.first():
                # Fix gas phase initial conditions
                model.fs.moving_bed.gas_phase.material_holdup[t0, x, ...].fix()
                model.fs.moving_bed.gas_phase.energy_holdup[t0, x, ...].fix()

            if x != length.last():
                # Fix solid phase initial conditions
                model.fs.moving_bed.solid_phase.material_holdup[t0, x, ...].fix()
                model.fs.moving_bed.solid_phase.energy_holdup[t0, x, ...].fix()

        # Initialize derivatives to zero
        model.fs.moving_bed.gas_phase.material_accumulation[...].set_value(0.0)
        model.fs.moving_bed.gas_phase.energy_accumulation[...].set_value(0.0)
        model.fs.moving_bed.solid_phase.material_accumulation[...].set_value(0.0)
        model.fs.moving_bed.solid_phase.energy_accumulation[...].set_value(0.0)

    return model


def fix_dof(model):
    mb = model.fs.moving_bed
    # Fix geometry variables
    mb.bed_diameter.fix(6.5*pyo.units.m)
    mb.bed_height.fix(5*pyo.units.m)

    # Fix inlet variables for all t
    time = model.fs.time

    # The following code should be valid for steady state model as well
    # Fix inlets to nominal values
    for t in time:
        # Inlets fixed to ss values, no disturbances
        mb.gas_inlet.flow_mol[t].fix(128.20513*pyo.units.mol/pyo.units.s)
        #mb.gas_inlet.pressure[t].fix(2.00)
        mb.gas_inlet.pressure[t].fix(2.00*pyo.units.bar)

        mb.solid_inlet.flow_mass[t].fix(591.4*pyo.units.kg/pyo.units.s)
        mb.gas_inlet.temperature[t].fix(298.15*pyo.units.K)
        mb.gas_inlet.mole_frac_comp[t, "CO2"].fix(0.02499)
        mb.gas_inlet.mole_frac_comp[t, "H2O"].fix(0.00001)
        mb.gas_inlet.mole_frac_comp[t, "CH4"].fix(0.97500)

        mb.solid_inlet.temperature[t].fix(1183.15*pyo.units.K)
        mb.solid_inlet.particle_porosity[t].fix(0.27)
        mb.solid_inlet.mass_frac_comp[t, "Fe2O3"].fix(0.45)
        mb.solid_inlet.mass_frac_comp[t, "Fe3O4"].fix(1e-9)
        mb.solid_inlet.mass_frac_comp[t, "Al2O3"].fix(0.55)


def get_state_variable_names(space):
    """
    These are a somewhat arbitrary set of time-indexed variables that are
    (more than) sufficient to solve for the rest of the variables.
    They happen to correspond do the property packages' state variables.

    """
    #space = m.fs.moving_bed.gas_phase.length_domain
    setpoint_states = []
    setpoint_states.extend(
        "fs.moving_bed.gas_phase.properties[*,%s].flow_mol" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.moving_bed.gas_phase.properties[*,%s].temperature" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.moving_bed.gas_phase.properties[*,%s].pressure" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.moving_bed.gas_phase.properties[*,%s].mole_frac_comp[CH4]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.moving_bed.gas_phase.properties[*,%s].mole_frac_comp[H2O]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.moving_bed.gas_phase.properties[*,%s].mole_frac_comp[CO2]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.moving_bed.solid_phase.properties[*,%s].flow_mass" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.moving_bed.solid_phase.properties[*,%s].temperature" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.moving_bed.solid_phase.properties[*,%s].mass_frac_comp[Fe2O3]" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.moving_bed.solid_phase.properties[*,%s].mass_frac_comp[Fe3O4]" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.moving_bed.solid_phase.properties[*,%s].mass_frac_comp[Al2O3]" % x
        for x in space if x != space.last()
    )
    return setpoint_states


def preprocess_dynamic_model(model):
    """Apply scaling transformation and perform variable elimination
    """
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    mbr = model.fs.moving_bed
    for var in mbr.solid_phase.energy_holdup[...]:
        model.scaling_factor[var] = 1e-3
    for var in mbr.solid_phase.energy_accumulation[...]:
        model.scaling_factor[var] = 1e-3
    for con in mbr.solid_phase.energy_accumulation_disc_eq[...]:
        model.scaling_factor[con] = 1e-3
    for con in mbr.solid_phase.energy_holdup_calculation[...]:
        model.scaling_factor[con] = 1e-3
    for var in mbr.solid_phase._enthalpy_flow[...]:
        model.scaling_factor[var] = 1e-3
    for var in mbr.solid_phase.properties[...].enth_mass:
        model.scaling_factor[var] = 1e-3
    for con in mbr.solid_phase.enthalpy_flow_linking_constraint[...]:
        model.scaling_factor[con] = 1e-3
    for var in mbr.gas_phase.deltaP[:, :]:
        model.scaling_factor[var] = 1e-5
    for con in mbr.gas_phase_config_pressure_drop[:, :]:
        model.scaling_factor[con] = 1e-5
    for var in mbr.gas_phase.pressure_dx[...]:
        model.scaling_factor[var] = 1e-5
    for con in mbr.gas_phase.pressure_dx_disc_eq[...]:
        model.scaling_factor[con] = 1e-5
    for var in mbr.gas_phase._enthalpy_flow[...]:
        model.scaling_factor[var] = 1e-3
    for var in mbr.gas_phase.properties[...].enth_mol:
        model.scaling_factor[var] = 1e-3
    for con in mbr.gas_phase.enthalpy_flow_linking_constraint[...]:
        model.scaling_factor[con] = 1e-3
    for var in mbr.gas_phase.energy_holdup[...]:
        model.scaling_factor[var] = 1e-3
    for var in mbr.gas_phase.energy_accumulation[...]:
        model.scaling_factor[var] = 1e-3
    for con in mbr.gas_phase.energy_accumulation_disc_eq[...]:
        model.scaling_factor[con] = 1e-3
    for con in mbr.gas_phase.energy_holdup_calculation[...]:
        model.scaling_factor[con] = 1e-3

    for con in mbr.solid_phase.reactions[:, :].gen_rate_expression[:]:
        model.scaling_factor[con] = 1e4
    for con in mbr.solid_phase.reactions[:, :].rate_constant_eqn[:]:
        model.scaling_factor[con] = 1e6
    for con in mbr.solid_phase.reactions[:, :].OC_conv_eqn:
        model.scaling_factor[con] = 1e6
    for con in mbr.solid_phase.reactions[:, :].OC_conv_temp_eqn:
        model.scaling_factor[con] = 1e3

    print("Applying scaling transformation...")
    scaling = pyo.TransformationFactory("core.scale_model")
    scaling.apply_to(model, rename=False)

    #
    # Replace expressions
    #
    mbr.solid_super_vel[:, 1.0].deactivate()
    sub_map = {
        id(mbr.velocity_superficial_solid[t]): (
            mbr.solid_phase.properties[t, 1.0].flow_mass
            / (
                mbr.bed_area
                * mbr.solid_phase.properties[t, 1.0].dens_mass_particle
            )
        )
        for t in model.fs.time
    }
    for con in model.component_data_objects(
        (pyo.Constraint, pyo.Objective), active=True
    ):
        con.set_value(replace_expressions(con.expr, sub_map))
        #new_expr = replace_expressions(con.expr, sub_map)
        #if con.expr is not new_expr:
        #    con.set_value(new_expr)


if __name__ == "__main__":
    m = make_model(dynamic=True)
    dof = degrees_of_freedom(m)
    print(f"dof = {dof}")
