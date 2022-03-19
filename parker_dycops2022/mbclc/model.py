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
import enum
import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.dae.flatten import flatten_dae_components
from pyomo.common.timing import HierarchicalTimer

from idaes.core import FlowsheetBlock, EnergyBalanceType
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog

# Import unit model and property package from workspace as some of my
# code broke when units were changed.
from common.unit_models.moving_bed import MBR as MovingBed
# Import Bidirectional Moving Bed
from common.bidirectional_moving_bed import MBR as BiMovingBed

from common.properties.methane_iron_oc import (
    GasPhaseParameterBlock,
    SolidPhaseParameterBlock,
    HeteroReactionParameterBlock,
)

from idaes.core.util.initialization import initialize_by_time_element
from idaes.core.util.dyn_utils import (
    copy_values_at_time,
    copy_non_time_indexed_values,
)

from idaes.apps.caprese.categorize import (
    categorize_dae_variables_and_constraints,
    VariableCategory,
    ConstraintCategory,
)


TIMER = HierarchicalTimer()


solver_available = pyo.SolverFactory('ipopt').available()
if solver_available:
    solver = pyo.SolverFactory('ipopt')
    solver.options = {
            #'tol': 1e-6,
            #'bound_push': 1e-8,
            'linear_solver': 'ma57',
            'halt_on_ampl_error': 'yes',
            }
else:
    solver = None


class SpatialDiscretization(enum.Enum):
    UNIFORM = 0
    CUSTOM = 1


def get_spatial_fe(nxfe):
    """
    This function defines the custom discretization scheme we
    use for the spatial domain of the CLC model. I believe
    Anca came up with this scheme.
    """
    fe_set = [0, 0.004]
    fe_a = 1/4.0
    fe_b = 0.2
    for i in range(1, nxfe+1):
        if i < nxfe*fe_a:
            fe_set.append(i*fe_b/(nxfe*fe_a))
        elif i == nxfe:
            fe_set.append(1)
        else:
            fe_set.append(fe_b + (i-nxfe*fe_a)*(1-fe_b)/(nxfe*(1-fe_a)))
    return fe_set


def make_model(**kwargs):

    horizon = kwargs.pop('horizon', 300.)
    tfe_width = kwargs.pop('tfe_width', 15.)
    ntcp = kwargs.pop('ntcp', 1)

    # IDAES logger:
    #outlvl = idaeslog.INFO
    #log = idaeslog.getIdaesLogger(__name__)
    #solver_log = idaeslog.getSolveLogger(__name__)

    # Create spatial grid
    nxfe = kwargs.pop('nxfe', 10)
    nxcp = kwargs.pop('nxcp', 3)
    spatial_disc = kwargs.pop('spatial_disc', SpatialDiscretization.UNIFORM)
    if spatial_disc == SpatialDiscretization.UNIFORM:
        xfe_set = [1.0*i/nxfe for i in range(nxfe+1)]
    elif spatial_disc == SpatialDiscretization.CUSTOM:
        xfe_set = get_spatial_fe(nxfe)
    else:
        raise ValueError()

    # Should we create a steady-state model?
    steady = kwargs.pop('steady', False)

    # Should we use different discretizations for solid and gas phases?
    # NOTE: default has been changed to True.
    bidirectional = kwargs.pop('bidirectional', True)

    # Create time grid
    if steady:
        time_set = [0]
    else:
        time_set = [0, horizon]
    ntfe = round(horizon/tfe_width)
    ntfe = kwargs.pop("ntfe", ntfe)

    # Create top-level and flowsheet models
    if steady:
        name = "Steady-state MB reduction CLC flowsheet model"
    else:
        name = "Dynamic MB reduction CLC flowsheet model"
    model = pyo.ConcreteModel(name=name)
    model.fs = FlowsheetBlock(default={
                    "dynamic": not steady,
                    "time_set": time_set,
                    "time_units": pyo.units.s,
                    })

    # Set up thermo props and reaction props
    model.fs.gas_properties = GasPhaseParameterBlock()
    model.fs.solid_properties = SolidPhaseParameterBlock()
    model.fs.hetero_reactions = HeteroReactionParameterBlock(default={
        "solid_property_package": model.fs.solid_properties,
        "gas_property_package": model.fs.gas_properties,
        })

    mb_config = {
        "finite_elements": nxfe,
        "has_holdup": True,
        "length_domain_set": xfe_set,
        "transformation_method": "dae.collocation",
        "collocation_points": nxcp,
        "transformation_scheme": "LAGRANGE-RADAU",
        "pressure_drop_type": "ergun_correlation",
        "gas_phase_config": {
            "property_package": model.fs.gas_properties,
            },
        "solid_phase_config": {
            "property_package": model.fs.solid_properties,
            "reaction_package": model.fs.hetero_reactions,
            }
        }

    # Create MovingBed unit model
    if bidirectional:
        mb_config.update({
            "transformation_method": "dae.finite_difference",
            "transformation_scheme": "BACKWARD",
            })
        model.fs.MB = BiMovingBed(default=mb_config)
    else:
        model.fs.MB = MovingBed(default=mb_config)

    # Time-discretization
    if not steady:
        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(model, wrt=model.fs.time, nfe=ntfe, ncp=ntcp,
                scheme='LAGRANGE-RADAU')

    return model


def make_square_model(**kwargs):
    # TODO: Why is this function in this file?
    # This argument processing exists just so I can change
    # the defaults?...
    steady = kwargs.pop("steady", False)

    horizon = kwargs.pop('horizon', 300.)
    tfe_width = kwargs.pop('tfe_width', 15.)
    ntcp = kwargs.pop('ntcp', 1)

    nxfe = kwargs.pop('nxfe', 10)
    nxcp = kwargs.pop('nxcp', 3)

    m = make_model(
            steady=steady,
            horizon=horizon,
            tfe_width=tfe_width,
            ntcp=ntcp,
            nxfe=nxfe,
            nxcp=nxcp,
            **kwargs,
            )
    time = m.fs.time

    design_vars = ["fs.MB.bed_diameter", "fs.MB.bed_height"]
    design_var_values = {var: None for var in design_vars}
    dynamic_inputs = [
        "fs.MB.gas_inlet.flow_mol[*]",
        "fs.MB.gas_inlet.pressure[*]",
        "fs.MB.gas_inlet.temperature[*]",
        "fs.MB.gas_inlet.mole_frac_comp[*,CO2]",
        "fs.MB.gas_inlet.mole_frac_comp[*,H2O]",
        "fs.MB.gas_inlet.mole_frac_comp[*,CH4]",

        "fs.MB.solid_inlet.flow_mass[*]",
        "fs.MB.solid_inlet.temperature[*]",
        "fs.MB.solid_inlet.particle_porosity[*]",
        "fs.MB.solid_inlet.mass_frac_comp[*,Fe2O3]",
        "fs.MB.solid_inlet.mass_frac_comp[*,Fe3O4]",
        "fs.MB.solid_inlet.mass_frac_comp[*,Al2O3]",
        ]
    input_values = {var: None for var in dynamic_inputs}
    fix_design_variables(m, design_var_values)
    fix_dynamic_inputs(m, time, input_values)

    return m


def make_square_dynamic_model(**kwargs):
    steady = kwargs.pop("steady", False)

    horizon = kwargs.pop('horizon', 300.)
    tfe_width = kwargs.pop('tfe_width', 15.)
    ntcp = kwargs.pop('ntcp', 1)

    nxfe = kwargs.pop('nxfe', 10)
    nxcp = kwargs.pop('nxcp', 3)

    TIMER.start("make model")
    m = make_model(
            steady=steady,
            horizon=horizon,
            tfe_width=tfe_width,
            ntcp=ntcp,
            nxfe=nxfe,
            nxcp=nxcp,
            **kwargs,
            )
    TIMER.stop("make model")
    time = m.fs.time
    t0 = time.first()

    design_vars = ["fs.MB.bed_diameter", "fs.MB.bed_height"]
    design_var_values = {var: None for var in design_vars}
    dynamic_inputs = [
        "fs.MB.gas_inlet.flow_mol[*]",
        "fs.MB.gas_inlet.pressure[*]",
        "fs.MB.gas_inlet.temperature[*]",
        "fs.MB.gas_inlet.mole_frac_comp[*,CO2]",
        "fs.MB.gas_inlet.mole_frac_comp[*,H2O]",
        "fs.MB.gas_inlet.mole_frac_comp[*,CH4]",

        "fs.MB.solid_inlet.flow_mass[*]",
        "fs.MB.solid_inlet.temperature[*]",
        "fs.MB.solid_inlet.particle_porosity[*]",
        "fs.MB.solid_inlet.mass_frac_comp[*,Fe2O3]",
        "fs.MB.solid_inlet.mass_frac_comp[*,Fe3O4]",
        "fs.MB.solid_inlet.mass_frac_comp[*,Al2O3]",
        ]
    input_values = {var: None for var in dynamic_inputs}
    fix_design_variables(m, design_var_values)
    fix_dynamic_inputs(m, time, input_values)

    flatten_out = kwargs.pop("flatten_out", None)

    # Identify proper differential variables to fix and fix them
    # at t0
    input_vars = [m.find_component(name) for name in dynamic_inputs]
    TIMER.start("flatten vars")
    scalar_vars, dae_vars = flatten_dae_components(m, time, pyo.Var)
    if flatten_out is not None:
        flatten_out[0] = scalar_vars
        flatten_out[1] = dae_vars
    TIMER.stop("flatten vars")
    TIMER.start("flatten cons")
    scalar_cons, dae_cons = flatten_dae_components(m, time, pyo.Constraint)
    TIMER.stop("flatten cons")
    TIMER.start("categorize")
    var_cat, con_cat = categorize_dae_variables_and_constraints(
        m, dae_vars, dae_cons, time, input_vars=input_vars
    )
    TIMER.stop("categorize")
    VC = VariableCategory
    CC = ConstraintCategory
    diff_vars = var_cat[VC.DIFFERENTIAL]
    for var in diff_vars:
        var[t0].fix()
    return m, var_cat, con_cat


def fix_dof(model):
    # Fix geometry variables
    model.fs.MB.bed_diameter.fix(6.5)
    model.fs.MB.bed_height.fix(5)

    # Fix inlet variables for all t
    time = model.fs.time

    # The following code should be valid for steady state model as well
    # Fix inlets to nominal values
    for t in time:
        # Inlets fixed to ss values, no disturbances
        model.fs.MB.gas_inlet.flow_mol[t].fix(128.20513)
        model.fs.MB.gas_inlet.pressure[t].fix(2.00)

        model.fs.MB.solid_inlet.flow_mass[t].fix(591.4)
        model.fs.MB.gas_inlet.temperature[t].fix(298.15)
        model.fs.MB.gas_inlet.mole_frac_comp[t, "CO2"].fix(0.02499)
        model.fs.MB.gas_inlet.mole_frac_comp[t, "H2O"].fix(0.00001)
        model.fs.MB.gas_inlet.mole_frac_comp[t, "CH4"].fix(0.97500)

        model.fs.MB.solid_inlet.temperature[t].fix(1183.15)
        model.fs.MB.solid_inlet.particle_porosity[t].fix(0.27)
        model.fs.MB.solid_inlet.mass_frac_comp[t, "Fe2O3"].fix(0.45)
        model.fs.MB.solid_inlet.mass_frac_comp[t, "Fe3O4"].fix(1e-9)
        model.fs.MB.solid_inlet.mass_frac_comp[t, "Al2O3"].fix(0.55)


def fix_design_variables(model, design_var_values):
    design_vars = []
    for cuid, value in design_var_values.items():
        var = model.find_component(cuid)
        design_vars.append(var)
        if value is None:
            var.fix()
        else:
            var.fix(value)
    return design_vars


def fix_dynamic_inputs(model, time, input_var_values):
    # These could be control inputs or disturbances
    input_vars = []
    for cuid, values in input_var_values.items():
        # Reference to slice along time
        ref = model.find_component(cuid)
        input_vars.append(ref)
        if values is None:
            ref.fix()
        else:
            try:
                assert len(values) == len(time)
            except TypeError:
                # values is a scalar
                values = [values]*len(time)
            for t, val in zip(time, values):
                ref[t].fix(val)
    return input_vars


def add_constraints_for_missing_variables(m):
    """
    Adds constraints that have been skipped at a boundary and without
    which, a variable does not participate in the model.

    This is necessary so that we have a valid value for every variable
    at every point in space, which is a useful assumption to be able
    to make when serializing the model wrt space.
    """
    time = m.fs.time
    t0 = time.first()
    space = m.fs.MB.gas_phase.length_domain
    x0 = space.first()
    xf = space.last()

    # Rate reaction stoichiometry constraints define the values of
    # rate_reaction_generation variables and do not exist at the solid
    # inlets
    p = "Sol"
    r = "R1"
    sp = m.fs.MB.solid_phase
    stoich_dict = m.fs.hetero_reactions.rate_reaction_stoichiometry
    for j in m.fs.solid_properties.component_list:
        stoich = stoich_dict[r, p, j]
        sp.rate_reaction_stoichiometry_constraint.add(
                (t0, xf, p, j),
                (
                    sp.rate_reaction_generation[t0, xf, p, j] -
                    stoich*sp.rate_reaction_extent[t0, xf, r]
                    == 0
                    ),
                )

    # Balance equations define the values of df/dx variables and are
    # skipped at inlets.
    p = "Vap"
    gp = m.fs.MB.gas_phase
    for j in m.fs.gas_properties.component_list:
        gp.material_balances.add(
                (t0, x0, j),
                (
                    - gp.material_flow_dx[t0, x0, p, j] +
                    gp.length*gp.mass_transfer_term[t0, x0, p, j]
                    == 0
                    ),
                )
    gp.enthalpy_balances.add(
            (t0, x0),
            (
                -gp.enthalpy_flow_dx[t0, x0, p] +
                gp.length*gp.heat[t0, x0]
                == 0
                ),
            )
    gp.pressure_balance.add(
            (t0, x0),
            (
                - gp.pressure_dx[t0, x0] +
                gp.length*gp.deltaP[t0, x0]
                == 0
                )
            )

    p = "Sol"
    sp = m.fs.MB.solid_phase
    r = "R1"
    dh_rxn_dict = m.fs.hetero_reactions.dh_rxn
    mw_dict = m.fs.solid_properties.mw_comp
    for j in m.fs.solid_properties.component_list:
        sp.material_balances.add(
                (t0, xf, j),
                (
                    sp.material_flow_dx[t0, xf, p, j] +
                    sp.length *
                    sp.rate_reaction_generation[t0, xf, p, j] *
                    mw_dict[j]
                    == 0
                    )
                )
    sp.enthalpy_balances.add(
            (t0, xf),
            (
                sp.enthalpy_flow_dx[t0, xf, p] +
                sp.length*sp.heat[t0, xf] +
                sp.length*(
                    - dh_rxn_dict[r]*sp.rate_reaction_extent[t0, xf, r]
                    )
                == 0
                )
            )


if __name__ == "__main__":
    m = make_model()
