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
"""
from pyomo.environ import ConcreteModel, SolverFactory, value
import pyomo.environ as pyo

from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    large_residuals_set,
)

# Import IDAES logger
import idaes.logger as idaeslog


from nmpc_examples.nmpc.model_helper import DynamicModelHelper
from nmpc_examples.nmpc.cost_expressions import (
    get_tracking_cost_from_constant_setpoint,
)
from nmpc_examples.nmpc.input_constraints import (
    get_piecewise_constant_constraints,
)

import enum


class ModelVersion(enum.Enum):
    IDAES_1_7 = 1
    IDAES_1_12 = 2
    IDAES_1_7_patch1 = 3


def set_default_design_variables(m):
    m.fs.MB.bed_diameter.fix(6.5) # m
    m.fs.MB.bed_height.fix(5) # m


def set_default_inlet_conditions(m, version=ModelVersion.IDAES_1_7):
    m.fs.MB.gas_inlet.flow_mol[:].fix(128.20513)  # mol/s
    m.fs.MB.gas_inlet.temperature[:].fix(298.15)  # K
    m.fs.MB.gas_inlet.pressure[:].fix(2.00)  # bar
    m.fs.MB.gas_inlet.mole_frac_comp[:, "CO2"].fix(0.02499)
    m.fs.MB.gas_inlet.mole_frac_comp[:, "H2O"].fix(0.00001)
    m.fs.MB.gas_inlet.mole_frac_comp[:, "CH4"].fix(0.975)

    m.fs.MB.solid_inlet.flow_mass[:].fix(591.4)  # kg/s
    m.fs.MB.solid_inlet.temperature[:].fix(1183.15)  # K
    m.fs.MB.solid_inlet.mass_frac_comp[:, "Fe2O3"].fix(0.45)
    m.fs.MB.solid_inlet.mass_frac_comp[:, "Fe3O4"].fix(1e-9)
    m.fs.MB.solid_inlet.mass_frac_comp[:, "Al2O3"].fix(0.55)
    if (
        version == ModelVersion.IDAES_1_12
        or version == ModelVersion.IDAES_1_7_patch1
    ):
        m.fs.MB.solid_inlet.particle_porosity[:].fix(0.27)


def fix_initial_conditions(m):
    t0 = m.fs.time.first()
    x0 = m.fs.MB.gas_phase.length_domain.first()
    xf = m.fs.MB.gas_phase.length_domain.last()
    m.fs.MB.gas_phase.material_holdup[t0, ...].fix()
    m.fs.MB.gas_phase.energy_holdup[t0, ...].fix()
    m.fs.MB.gas_phase.material_holdup[t0, x0, ...].unfix()
    m.fs.MB.gas_phase.energy_holdup[t0, x0, ...].unfix()

    m.fs.MB.solid_phase.material_holdup[t0, ...].fix()
    m.fs.MB.solid_phase.energy_holdup[t0, ...].fix()
    m.fs.MB.solid_phase.material_holdup[t0, xf, ...].unfix()
    m.fs.MB.solid_phase.energy_holdup[t0, xf, ...].unfix()


def initialize_derivative_variables(m):
    t0 = m.fs.time.first()
    m.fs.MB.gas_phase.material_accumulation[...].set_value(0.0)
    m.fs.MB.gas_phase.energy_accumulation[...].set_value(0.0)
    m.fs.MB.solid_phase.material_accumulation[...].set_value(0.0)
    m.fs.MB.solid_phase.energy_accumulation[...].set_value(0.0)


def add_piecewise_constant_constraints(
    m,
    sample_points=None,
    n_samples=5,
    sample_time=60.0,
):
    if sample_points is None:
        sample_points = [i*sample_time for i in range(n_samples + 1)]
    inputs = [
        m.fs.MB.gas_inlet.flow_mol,
        m.fs.MB.solid_inlet.flow_mass,
    ]
    input_set, pwc_con = get_piecewise_constant_constraints(
        inputs,
        m.fs.time,
        sample_points,
    )
    m.input_set = input_set
    m.piecewise_constant_constraints = pwc_con
    return inputs


def get_state_variable_names(space):
    setpoint_states = []
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].flow_mol" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].temperature" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].pressure" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[CH4]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[H2O]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.gas_phase.properties[*,%s].mole_frac_comp[CO2]" % x
        for x in space if x != space.first()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].flow_mass" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].temperature" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Fe2O3]" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Fe3O4]" % x
        for x in space if x != space.last()
    )
    setpoint_states.extend(
        "fs.MB.solid_phase.properties[*,%s].mass_frac_comp[Al2O3]" % x
        for x in space if x != space.last()
    )
    return setpoint_states


def add_objective(
    m,
    inputs=None,
    version=ModelVersion.IDAES_1_7,
):
    x0 = m.fs.MB.gas_phase.length_domain.first()
    xf = m.fs.MB.gas_phase.length_domain.last()
    if inputs is None:
        slices = [
            m.fs.MB.gas_phase.properties[:, x0].flow_mol,
            m.fs.MB.solid_phase.properties[:, xf].flow_mass,
        ]
        cuids = [pyo.ComponentUID(slice_) for slice_ in slices]
        values = [128.20513, 591.4]
        inputs = dict(zip(cuids, values))

    nxfe = m.fs.MB.gas_phase.length_domain.get_discretization_info()["nfe"]
    m_setpoint = make_model(
        steady=True,
        nxfe=nxfe,
        version=version,
    )
    setpoint_helper = DynamicModelHelper(m_setpoint, m_setpoint.fs.time)
    setpoint_helper.load_data_at_time(inputs)

    ipopt = pyo.SolverFactory("ipopt")
    res = ipopt.solve(m_setpoint, tee=True)
    pyo.assert_optimal_termination(res)

    setpoint_data = setpoint_helper.get_data_at_time()

    #differential_vars = []
    #differential_vars.extend(
    #    pyo.Reference(m.fs.MB.gas_phase.material_holdup[:, x, "Vap", j])
    #    for x in m.fs.MB.gas_phase.length_domain if x != x0
    #    for j in m.fs.gas_properties.component_list
    #)
    #differential_vars.extend(
    #    pyo.Reference(m.fs.MB.gas_phase.energy_holdup[:, x, "Vap"])
    #    for x in m.fs.MB.gas_phase.length_domain if x != x0
    #)
    #differential_vars.extend(
    #    pyo.Reference(m.fs.MB.solid_phase.material_holdup[:, x, "Sol", j])
    #    for x in m.fs.MB.solid_phase.length_domain if x != x0
    #    for j in m.fs.solid_properties.component_list
    #)
    #differential_vars.extend(
    #    pyo.Reference(m.fs.MB.solid_phase.energy_holdup[:, x, "Sol"])
    #    for x in m.fs.MB.solid_phase.length_domain if x != x0
    #)
    state_var_names = get_state_variable_names(m.fs.MB.gas_phase.length_domain)
    differential_vars = [m.find_component(name) for name in state_var_names]

    m.tracking_cost = get_tracking_cost_from_constant_setpoint(
        differential_vars,
        m.fs.time,
        setpoint_data,
    )
    m.tracking_obj = pyo.Objective(
        expr=sum(
            m.tracking_cost[t] for t in m.fs.time if t != m.fs.time.first()
        )
    )


def make_model(
    steady=True,
    n_samples=5,
    sample_time=60.0,
    ntfe_per_sample=2,
    nxfe=10,
    t0=0.0,
    version=ModelVersion.IDAES_1_7,
    initialize=True,
):
    if version == ModelVersion.IDAES_1_7:
        # FIXME: Local imports for now to facilitate switching between
        # IDAES versions. (imports fail due to no get_solver when we try
        # to import 1.12 versions with IDAES 1.7, and we can't construct
        # the 1.7 model with IDAES 1.12 as the model isn't built with
        # any units)
        from parker_focapo2023.idaes_1_7_gas_solid_contactors.unit_models.moving_bed import (
            MBR,
            BiMBR,
        )

        # Import property packages
        from parker_focapo2023.idaes_1_7_gas_solid_contactors.properties.methane_iron_OC_reduction.gas_phase_thermo import (
            GasPhaseThermoParameterBlock,
        )
        from parker_focapo2023.idaes_1_7_gas_solid_contactors.properties.methane_iron_OC_reduction.solid_phase_thermo import (
            SolidPhaseThermoParameterBlock,
        )
        from parker_focapo2023.idaes_1_7_gas_solid_contactors.properties.methane_iron_OC_reduction.hetero_reactions import (
            HeteroReactionParameterBlock,
        )
    elif version == ModelVersion.IDAES_1_12:
        # Import MBR unit model
        from parker_focapo2023.idaes_1_12_gas_solid_contactors.unit_models.moving_bed import (
            MBR,
            BiMBR,
        )
        
        # Import property packages
        from parker_focapo2023.idaes_1_12_gas_solid_contactors.properties.methane_iron_OC_reduction.gas_phase_thermo import (
            GasPhaseParameterBlock as GasPhaseThermoParameterBlock,
        )
        from parker_focapo2023.idaes_1_12_gas_solid_contactors.properties.methane_iron_OC_reduction.solid_phase_thermo import (
            SolidPhaseParameterBlock as SolidPhaseThermoParameterBlock,
        )
        from parker_focapo2023.idaes_1_12_gas_solid_contactors.properties.methane_iron_OC_reduction.hetero_reactions import (
            HeteroReactionParameterBlock,
        )
    elif version == ModelVersion.IDAES_1_7_patch1:
        from parker_focapo2023.idaes_1_7_patch1_gas_solid_contactors.unit_models.moving_bed import (
            MBR,
            BiMBR,
        )

        # Import property packages
        from parker_focapo2023.idaes_1_7_patch1_gas_solid_contactors.properties.methane_iron_OC_reduction.gas_phase_thermo import (
            GasPhaseThermoParameterBlock,
        )
        from parker_focapo2023.idaes_1_7_patch1_gas_solid_contactors.properties.methane_iron_OC_reduction.solid_phase_thermo import (
            SolidPhaseThermoParameterBlock,
        )
        from parker_focapo2023.idaes_1_7_patch1_gas_solid_contactors.properties.methane_iron_OC_reduction.hetero_reactions import (
            HeteroReactionParameterBlock,
        )

    dynamic = not steady
    m = ConcreteModel()
    fs_config = {"dynamic": dynamic}
    if dynamic:
        horizon = n_samples * sample_time
        fs_config["time_set"] = [t0, horizon]
    else:
        fs_config["time_set"] = [t0]
    fs_config["time_units"] = pyo.units.s
    m.fs = FlowsheetBlock(default=fs_config)

    # Set up thermo props and reaction props
    m.fs.gas_properties = GasPhaseThermoParameterBlock()
    m.fs.solid_properties = SolidPhaseThermoParameterBlock()

    m.fs.hetero_reactions = HeteroReactionParameterBlock(
        default={
            "solid_property_package": m.fs.solid_properties,
            "gas_property_package": m.fs.gas_properties,
        }
    )

    m.fs.MB = BiMBR(
        default={
            "transformation_method": "dae.finite_difference",
            "finite_elements": nxfe,
            "has_holdup": True,
            "gas_phase_config": {"property_package": m.fs.gas_properties},
            "solid_phase_config": {
                "property_package": m.fs.solid_properties,
                "reaction_package": m.fs.hetero_reactions,
            },
        }
    )

    set_default_design_variables(m)
    set_default_inlet_conditions(m, version=version)

    solver = SolverFactory("ipopt")
    if steady:
        # If steady, initialize and solve to default steady state

        # State arguments for initializing property state blocks
        # Gas phase temperature is initialized at solid
        # temperature because thermal mass of solid >> thermal mass of gas
        # Particularly useful for initialization if reaction takes place
        blk = m.fs.MB
        gas_phase_state_args = {
            "flow_mol": blk.gas_inlet.flow_mol[t0].value,
            "temperature": blk.solid_inlet.temperature[t0].value,
            "pressure": blk.gas_inlet.pressure[t0].value,
            "mole_frac": {
                "CH4": blk.gas_inlet.mole_frac_comp[t0, "CH4"].value,
                "CO2": blk.gas_inlet.mole_frac_comp[t0, "CO2"].value,
                "H2O": blk.gas_inlet.mole_frac_comp[t0, "H2O"].value,
            },
        }
        solid_phase_state_args = {
            "flow_mass": blk.solid_inlet.flow_mass[t0].value,
            "temperature": blk.solid_inlet.temperature[t0].value,
            "mass_frac": {
                "Fe2O3": blk.solid_inlet.mass_frac_comp[t0, "Fe2O3"].value,
                "Fe3O4": blk.solid_inlet.mass_frac_comp[t0, "Fe3O4"].value,
                "Al2O3": blk.solid_inlet.mass_frac_comp[t0, "Al2O3"].value,
            },
        }

        if initialize:
            m.fs.MB.initialize(
                outlvl=idaeslog.INFO,
                gas_phase_state_args=gas_phase_state_args,
                solid_phase_state_args=solid_phase_state_args,
            )
            # Create a solver
            solver.solve(m.fs.MB, tee=True)

    if dynamic:
        ntfe = ntfe_per_sample * n_samples
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.fs.time, scheme="BACKWARD", nfe=ntfe)

        fix_initial_conditions(m)
        # Set inlets again now that time has been discretized.
        set_default_inlet_conditions(m, version=version)
        initialize_derivative_variables(m)

        if initialize:
            m_steady = make_model(steady=True, nxfe=nxfe, version=version)
            m_steady_helper = DynamicModelHelper(m_steady, m_steady.fs.time)
            t0 = m_steady.fs.time.first()
            steady_data = m_steady_helper.get_data_at_time()
            steady_scalar_data = m_steady_helper.get_scalar_variable_data()

            m_helper = DynamicModelHelper(m, m.fs.time)
            m_helper.load_data_at_time(steady_data)
            m_helper.load_scalar_data(steady_scalar_data)
            solver.solve(m, tee=True)

    return m


def main():
    version = ModelVersion.IDAES_1_12
    m = make_model(
        steady=False,
        version=version,
    )
    add_objective(m, version=version)
    ipopt = pyo.SolverFactory("ipopt")
    ipopt.solve(m, tee=True)


if __name__ == "__main__":
    main()
