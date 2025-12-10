#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

#  This code was developed at Carnegie Mellon University by Ilayda Akkor and
#  Chrysanthos E. Gounaris, as part of a research project funded by the Dow 
#  Chemical Company. In particular, we acknowledge the close collaboration of
#  Dow's Shachit S. Iyer, John Dowdle and Le Wang, who provided key contributions
#  pertaining to project conceptualization, research design and methodology,
#  model design, and results discussions.

# If you find this code useful for your research, please consider citing
# "Mathematical Modeling and Economic Optimization of a Piperazine-Based
# Post-Combustion Carbon Capture Process" by Ilayda Akkor, Shachit S. Iyer,
# John Dowdle, Le Wang and Chrysanthos E. Gounaris

"""
Liquid phase properties for solvent flow in the absorber column, mixture of CO2-H2O-PZ
"""

# Import components from Pyomo
from pyomo.environ import (
    Constraint,
    Expression,
    Reference,
    Param,
    units as pyunits,
    Var,
    Set,
    exp,
    log,
    value,
)

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    PhysicalParameterBlock,
    StateBlockData,
    StateBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    Component,
    LiquidPhase,
)
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables,
)
from idaes.core.util.constants import Constants as const
import idaes.logger as idaeslog

__author__ = "Ilayda Akkor"


@declare_process_block_class("PZParameterBlock")
class PZParameterData(PhysicalParameterBlock):
    """
    Class for defining components & parameters of vapor phase property model.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()

    def build(self):
        """
        Callable method for Block construction.
        """
        super(PZParameterData, self).build()

        self._state_block_class = PZStateBlock

        # Define Component objects for all species
        self.CO2 = Component()
        self.H2O = Component()
        self.PZ = Component()
        self.N2 = Component()

        # Create a subset of components list
        volatile_components = ["CO2", "H2O", "PZ"]
        self.volatile_components = Set(
            initialize=volatile_components, doc="Components that can have mass transfer"
        )

        # Define Phase objects for all phases
        self.Liq = LiquidPhase()

        # Define basic parameters
        self.mw_comp = Param(
            self.component_list,
            mutable=True,
            initialize={"CO2": 0.04401, "H2O": 0.01802, "PZ": 0.086136, "N2": 0.02801},
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )

        self.cp_vol = Param(
            mutable=True,
            initialize=3.23e6,
            units=pyunits.J / pyunits.m**3 / pyunits.K,
            doc="Liquid heat capacity",
        )

        self.deltaH_vap = Param(
            mutable=True,
            initialize=48000,
            units=pyunits.J / pyunits.mol,
            doc="Heat of vaporization of water",
        )

        reactions = [1, 2, 3, 4, 5, 6, 7]
        A_init = {1: 231.4, 2: 216, 3: 132.9, 4: -11.91, 5: -29.31, 6: -8.21, 7: -30.78}

        def lnK_param_A_init(b, j):
            return A_init[j]

        self.lnK_param_A = Param(
            reactions,
            mutable=True,
            initialize=lnK_param_A_init,
            units=pyunits.dimensionless,
            doc="Parameter in regressed expression for log of equilibrium constant",
        )

        B_init = {1: -12092, 2: -12432, 3: -13446, 4: -4351, 5: 5615, 6: -5286, 7: 5615}

        def lnK_param_B_init(b, j):
            return B_init[j]

        self.lnK_param_B = Param(
            reactions,
            mutable=True,
            initialize=lnK_param_B_init,
            units=pyunits.K,
            doc="Parameter in regressed expression for log of equilibrium constant",
        )

        C_init = {1: -36.78, 2: -35.48, 3: -22.48, 4: 0, 5: 0, 6: 0, 7: 0}

        def lnK_param_C_init(b, j):
            return C_init[j]

        self.lnK_param_C = Param(
            reactions,
            mutable=True,
            initialize=lnK_param_C_init,
            units=pyunits.dimensionless,
            doc="Parameter in regressed expression for log of equilibrium constant",
        )

        self.diffusivity_param_1 = Param(
            initialize=0.024 * 10 ** (-4),
            mutable=True,
            units=pyunits.m ** 2 / pyunits.s
        )

        self.diffusivity_param_2 = Param(
            initialize=-2122,
            mutable=True,
            units=pyunits.K)

        self.henry_param_1 = Param(
            initialize=1.7107 * 10 ** 7,
            mutable=True,
            units=pyunits.kPa * pyunits.m ** 3 / pyunits.mol,
        )

        self.henry_param_2 = Param(initialize=-1886.1, mutable=True, units=pyunits.K)

        self.reference_rate = Param(
            initialize=53.7,
            mutable=True,
            units=pyunits.m ** 3 / pyunits.mol / pyunits.s,
        )
        self.activation_energy = Param(
            initialize=-3.36 * 10 ** 4, mutable=True, units=pyunits.J / pyunits.mol
        )

        self.dens_mass_param_1 = Param(initialize=0.0407, mutable=True, units=pyunits.kg / pyunits.mole)
        self.dens_mass_param_2 = Param(initialize=0.008, mutable=True, units=pyunits.kg / pyunits.mole)
        self.dens_mass_param_3 = Param(initialize=0.991, mutable=True, units=pyunits.dimensionless)

        self.equilibrium_pressure_CO2_variation = Param(initialize=1, mutable=True)
        self.equilibrium_pressure_H2O_variation = Param(initialize=1, mutable=True)
        self.equilibrium_pressure_PZ_variation = Param(initialize=1, mutable=True)

        self.viscosity_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_1_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_2_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_3_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_4_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_5_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_6_variation = Param(initialize=1, mutable=True)
        self.log_k_eq_7_variation = Param(initialize=1, mutable=True)
        
        self.heat_of_absorption_variation = Param(initialize=1, mutable=True)
        
        self.dens_mass_variation = Param(initialize=1, mutable=True)

        self.dens_mass_H2O_param_1 = Param(
            initialize=1002.3, mutable=True, units=pyunits.kg / pyunits.m ** 3
        )
        self.dens_mass_H2O_param_2 = Param(
            initialize=0.1321, mutable=True, units=pyunits.kg / pyunits.m ** 3 / pyunits.K
        )
        self.dens_mass_H2O_param_3 = Param(
            initialize=0.00308,
            mutable=True,
            units=pyunits.kg / pyunits.m ** 3 / pyunits.K ** 2,
        )

        self.T_ref_h2o = Param(initialize=273.15, mutable=True, units=pyunits.K)

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "temperature": {"method": None},
                "mole_frac_comp": {"method": None},
                "flow_mol": {"method": None},
                "flow_mol_comp": {"method": "_flow_mol_comp"},
                "mw_comp": {"method": "_mw_comp"},
                "dens_mass": {"method": "_dens_mass"},
                "visc_d": {"method": "_visc_d"},
                "reaction_rate": {"method": "_reaction_rate"},
                "henry": {"method": "_henry"},
                "conc_mol_comp": {"method": "_conc_mol_comp"},
                "conc_mol": {"method": "_conc_mol"},
                "log_k_eq": {"method": "_log_k_eq"},
            }
        )

        obj.define_custom_properties(
            {
                "alpha": {"method": "_loading"},
                "equilibrium_pressure": {"method": "_equilibrium_pressure"},
                "weight_percent_PZ": {"method": "_weight_percent_PZ"},
                "diffusivity_cst": {"method": "_diffusivity_cst"},
                "viscosity_water": {"method": "_viscosity_water"},
                "deltaH_abs": {"method": "_heat_absorption"},
                "concentration_spec": {"method": "_concentration_spec"},
                "P_sat": {"method": "_saturation_pressure"},
            }
        )

        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _PZStateBlock(StateBlock):
    def initialize(
        blk,
        state_args=None,
        state_vars_fixed=False,
        hold_state=False,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="properties")

        # Create solver
        solver_obj = get_solver(solver, optarg)

        flags = _prepare_state(blk, state_args, state_vars_fixed)
        _initialize_state(blk, solver_obj, init_log, solve_log)
        _restore_state(blk, flags, hold_state)

        init_log.info("Initialization Complete")

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        _unfix_state(blk, flags, outlvl)


@declare_process_block_class("PZStateBlock", block_class=_PZStateBlock)
class PZStateBlockData(StateBlockData):
    """
    Class for defining state variables, adding property variables and constraints
    """

    def build(self):
        """Callable method for Block construction."""
        super(PZStateBlockData, self).build()

        # Create state variables

        self.mole_frac_comp = Var(
            self.component_list,
            initialize=0.33,
            bounds=(0, None),
            units=pyunits.mole / pyunits.mole,
            doc="Component mole fractions",
        )
        self.temperature = Var(
            initialize=322.15,
            bounds=(298.15, 655),
            units=pyunits.K,
            doc="State temperature",
        )
        self.flow_mol = Var(
            initialize=100,
            bounds=(0, None),
            units=pyunits.mol / pyunits.s,
            doc="State temperature",
        )

        _add_loading(self)
        _add_heat_of_absorption(self)

    def _equilibrium_pressure(self):
        _add_equilibrium_pressure(self)

    def _visc_d(self):
        _add_viscosity(self)

    def _flow_mol_comp(self):
        _add_flow_mol_comp(self)

    def _dens_mass(self):
        _add_density(self)

    def _mw_comp(self):
        _add_molecular_weight(self)

    def _conc_mol_comp(self):
        _add_concentration(self)

    def _conc_mol(self):
        _add_concentration_tot(self)

    def _reaction_rate(self):
        _add_rxn_rate_cst(self)

    def _henry(self):
        _add_henrys_cst(self)

    def _log_k_eq(self):
        _add_equilibrium_cst(self)

    def _weight_percent_PZ(self):
        _add_weight_percent_PZ(self)

    def _diffusivity_cst(self):
        _add_diffusivity_cst(self)

    def _viscosity_water(self):
        _add_viscosity_water(self)

    def _concentration_spec(self):
        _add_concentration_spec(self)

    def _saturation_pressure(self):
        _add_saturation_pressure(self)

    def _heat_absorption(self):
        _add_heat_of_absorption(self)

    def _loading(self):
        _add_loading(self)

    def define_state_vars(self):
        return _return_state_var_dict(self)

    def get_material_flow_terms(self, p, j):
        return self.flow_mol * self.mole_frac_comp[j]

    def get_enthalpy_flow_terms(self, p):
        """Create enthalpy flow terms."""
        return self.flow_mol * self.enth_mol

    def default_material_balance_type(self):
        return MaterialBalanceType.componentPhase

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar


def _return_state_var_dict(self):
    return {
        "mole_frac_comp": self.mole_frac_comp,
        "temperature": self.temperature,
        "flow_mol": self.flow_mol,
    }


def _add_flow_mol_comp(self):
    x_in = {"CO2": 0.041871, "H2O": 0.870899, "PZ": 0.08723, "N2": 0}

    def liq_component_flowrate_init(m, j):
        return x_in[j] * 100

    self.flow_mol_comp = Var(
        self.component_list,
        initialize=liq_component_flowrate_init,
        units=pyunits.mole / pyunits.s,
        bounds=(0, None),
        doc="Molar flowrate of gas per component(mol/s)",
    )

    def liq_flowrate_component_rule(blk, j):
        return blk.flow_mol_comp[j] == blk.mole_frac_comp[j] * blk.flow_mol

    self.liq_flowrate_eq = Constraint(
        self.component_list,
        rule=liq_flowrate_component_rule,
        doc="Liquid flowrate per component",
    )

    if self.config.defined_state is False:

        def sum_f_l_j_rule(m):
            return sum(m.flow_mol_comp[j] for j in m.component_list) == self.flow_mol

        self.sum_f_l_j = Constraint(rule=sum_f_l_j_rule)


def _add_loading(self):
    self.alpha = Var(
        initialize=0.35,
        units=pyunits.dimensionless,
        doc="Solvent loading",
    )

    if self.config.defined_state is False:

        def loading_rule(b):
            return (
                self.alpha * 2 * self.mole_frac_comp["PZ"] == self.mole_frac_comp["CO2"]
            )

        self.loading_eq = Constraint(rule=loading_rule)


def _add_equilibrium_pressure(self):
    self.equilibrium_pressure = Var(
        self.params.volatile_components,
        initialize=1000,
        bounds=(0, None),
        units=pyunits.Pa,
        doc="Equilibrium pressures. CO2 experimental correlation, "
        "H2O and PZ Antione equation",
    )

    p1_init = {"H2O": 72.55, "PZ": 172.78}
    p2_init = {"H2O": 7206.7, "PZ": 13492}
    p3_init = {"H2O": 7.1385, "PZ": 21.91}
    p4_init = {"H2O": 4.04 * 10 ** (-6), "PZ": 1.378 * 10 ** (-5)}

    def equilibrium_pressure_param_1_init(b, j):
        return p1_init[j]

    def equilibrium_pressure_param_2_init(b, j):
        return p2_init[j]

    def equilibrium_pressure_param_3_init(b, j):
        return p3_init[j]

    def equilibrium_pressure_param_4_init(b, j):
        return p4_init[j]

    self.equilibrium_pressure_param_1 = Param(
        ["H2O", "PZ"],
        mutable=True,
        initialize=equilibrium_pressure_param_1_init,
        units=pyunits.dimensionless,
    )

    self.equilibrium_pressure_param_2 = Param(
        ["H2O", "PZ"],
        mutable=True,
        initialize=equilibrium_pressure_param_2_init,
        units=pyunits.K,
    )

    self.equilibrium_pressure_param_3 = Param(
        ["H2O", "PZ"],
        mutable=True,
        initialize=equilibrium_pressure_param_3_init,
        units=pyunits.dimensionless,
    )

    self.equilibrium_pressure_param_4 = Param(
        ["H2O", "PZ"],
        mutable=True,
        initialize=equilibrium_pressure_param_4_init,
        units=1 / pyunits.K**2,
    )

    self.equilibrium_pressure_CO2_param_1 = Param(
        initialize=35.3, units=pyunits.dimensionless
    )
    self.equilibrium_pressure_CO2_param_2 = Param(initialize=11054, units=pyunits.K)
    self.equilibrium_pressure_CO2_param_3 = Param(
        initialize=18.9, units=pyunits.dimensionless
    )
    self.equilibrium_pressure_CO2_param_4 = Param(initialize=4958, units=pyunits.K)
    self.equilibrium_pressure_CO2_param_5 = Param(initialize=10163, units=pyunits.K)

    if self.config.defined_state is False:

        def equilibrium_pressure_rule(b, j):
            if j == "CO2":
                return self.equilibrium_pressure[j] / pyunits.Pa == \
                       self.params.equilibrium_pressure_CO2_variation* exp(
                    self.equilibrium_pressure_CO2_param_1
                    - self.equilibrium_pressure_CO2_param_2 / self.temperature
                    - self.equilibrium_pressure_CO2_param_3 * (self.alpha**2)
                    + self.equilibrium_pressure_CO2_param_4
                    * self.alpha
                    / self.temperature
                    + self.equilibrium_pressure_CO2_param_5
                    * (self.alpha**2)
                    / self.temperature
                )
            elif j == "H2O":
                return self.equilibrium_pressure[j] / pyunits.Pa == \
                       self.params.equilibrium_pressure_H2O_variation * self.mole_frac_comp[
                    j
                ] * exp(
                    self.equilibrium_pressure_param_1[j]
                    - self.equilibrium_pressure_param_2[j] / self.temperature
                    - self.equilibrium_pressure_param_3[j]
                    * log(self.temperature / pyunits.K)
                    + self.equilibrium_pressure_param_4[j] * self.temperature**2
                )
            elif j == "PZ":
                return self.equilibrium_pressure[j] / pyunits.Pa == \
                       self.params.equilibrium_pressure_PZ_variation * self.mole_frac_comp[
                    j
                ] * exp(
                    self.equilibrium_pressure_param_1[j]
                    - self.equilibrium_pressure_param_2[j] / self.temperature
                    - self.equilibrium_pressure_param_3[j]
                    * log(self.temperature / pyunits.K)
                    + self.equilibrium_pressure_param_4[j] * self.temperature**2
                )

        self.equilibrium_pressure_eq = Constraint(
            self.params.volatile_components, rule=equilibrium_pressure_rule
        )


def _add_molecular_weight(self):
    self.mw_comp = Reference(self.params.mw_comp)


def _add_density(self):
    self.dens_mass_H2O = Var(
        initialize=985, units=pyunits.kg / pyunits.m**3, doc="Density of water"
    )

    if self.config.defined_state is False:

        def dens_H2O_rule(b):
            return (
                self.dens_mass_H2O
                == self.params.dens_mass_H2O_param_1
                - self.params.dens_mass_H2O_param_2 * (self.temperature - self.params.T_ref_h2o)
                - self.params.dens_mass_H2O_param_3 * (self.temperature - self.params.T_ref_h2o) ** 2
            )

        self.dens_H2O_eq = Constraint(rule=dens_H2O_rule)

    self.dens_mass = Var(
        initialize=1030, units=pyunits.kg / pyunits.m**3, doc="Mixture density"
    )

    if self.config.defined_state is False:

        def dens_rule(b):
            return self.dens_mass == self.params.dens_mass_variation * self.dens_mass_H2O * (
                self.params.dens_mass_param_1
                * self.mole_frac_comp["CO2"]
                / self.mw_comp["CO2"]
                + self.params.dens_mass_param_2
                * self.mole_frac_comp["PZ"]
                / self.mw_comp["PZ"]
                + self.params.dens_mass_param_3
            )

        self.dens_eq = Constraint(rule=dens_rule)


def _add_weight_percent_PZ(self):
    self.weight_percent_PZ = Var(initialize=0.3, units=pyunits.dimensionless)

    if self.config.defined_state is False:

        def weight_percent_rule(b):
            return (
                self.weight_percent_PZ
                * (
                    self.mole_frac_comp["PZ"] * self.mw_comp["PZ"] * 1000
                    + self.mole_frac_comp["H2O"] * self.mw_comp["H2O"] * 1000
                )
                == self.mole_frac_comp["PZ"] * self.mw_comp["PZ"] * 1000
            )

        self.weight_percent_PZ_eq = Constraint(rule=weight_percent_rule)


def _add_viscosity_water(self):
    self.viscosity_water = Var(
        initialize=0.005, bounds=(0, None), units=pyunits.kg / pyunits.m / pyunits.s
    )

    self.viscosity_water_param_1 = Param(
        initialize=(10 ** (-3)) * 1.002, units=pyunits.kg / pyunits.m / pyunits.s
    )
    self.viscosity_water_param_2 = Param(initialize=1.3272, units=pyunits.dimensionless)
    self.viscosity_water_param_3 = Param(initialize=0.001053, units=1 / pyunits.K)
    self.viscosity_water_T_1 = Param(initialize=293, units=pyunits.K)
    self.viscosity_water_T_2 = Param(initialize=168, units=pyunits.K)

    if self.config.defined_state is False:

        def viscosity_water_rule(b):
            return self.viscosity_water == self.viscosity_water_param_1 * exp(
                2.303
                * (
                    self.viscosity_water_param_2
                    * (self.viscosity_water_T_1 - self.temperature)
                    - self.viscosity_water_param_3
                    * (self.temperature - self.viscosity_water_T_1) ** 2
                )
                / (self.temperature - self.viscosity_water_T_2)
            )

        self.viscosity_water_eq = Constraint(rule=viscosity_water_rule)


def _add_viscosity(self):
    self.visc_d = Var(
        initialize=0.005, bounds=(1e-5, None), units=pyunits.kg / pyunits.m / pyunits.s
    )

    self.visc_d_param_1 = Param(initialize=487.52, units=pyunits.dimensionless)
    self.visc_d_param_2 = Param(initialize=1389.31, units=pyunits.dimensionless)
    self.visc_d_param_3 = Param(initialize=1.58, units=pyunits.K)
    self.visc_d_param_4 = Param(initialize=4.5, units=pyunits.K)
    self.visc_d_param_5 = Param(initialize=8.73, units=pyunits.K)
    self.visc_d_param_6 = Param(initialize=0.0038, units=pyunits.dimensionless)
    self.visc_d_param_7 = Param(initialize=0.3, units=pyunits.K)
    self.visc_d_param_8 = Param(initialize=1, units=pyunits.K)

    if self.config.defined_state is False:

        def viscosity_rule(b):
            return self.visc_d == self.params.viscosity_variation * self.viscosity_water * exp(
                (
                    (self.visc_d_param_1 * self.weight_percent_PZ + self.visc_d_param_2)
                    * self.temperature
                    + (
                        self.visc_d_param_3 * self.weight_percent_PZ
                        + self.visc_d_param_4
                    )
                )
                * (
                    self.alpha
                    * (
                        self.visc_d_param_5 * self.weight_percent_PZ
                        - self.visc_d_param_6 * self.temperature
                        - self.visc_d_param_7
                    )
                    + self.visc_d_param_8
                )
                * self.weight_percent_PZ
                / (self.temperature**2)
            )

        self.viscosity_eq = Constraint(rule=viscosity_rule)


def _add_rxn_rate_cst(self):
    self.reaction_rate = Var(
        initialize=160,
        bounds=(1e-3, None),
        units=pyunits.m**3 / pyunits.mol / pyunits.s,
        doc="Reaction rate constant",
    )

    self.reference_temperature = Param(
        initialize=298.15, mutable=True, units=pyunits.K
    )
    if self.config.defined_state is False:

        def rxn_rate_cst_rule(b):
            return self.reaction_rate == self.params.reference_rate * exp(
                self.params.activation_energy
                * (1 / self.temperature - 1 / self.reference_temperature)
                / (
                    round(value(const.gas_constant), 3)
                    * pyunits.J
                    / pyunits.mole
                    / pyunits.K
                )
            )

        self.rxn_rate_cst_eq = Constraint(rule=rxn_rate_cst_rule)


def _add_henrys_cst(self):
    self.henry = Var(
        initialize=5,
        bounds=(1e-3, None),
        units=pyunits.kPa * pyunits.m**3 / pyunits.mol,
        doc="Henry's constant for carbon dioxide",
    )

    if self.config.defined_state is False:

        def henry_cst_rule(b):
            return (
                self.henry
                == self.params.henry_param_1
                * exp(self.params.henry_param_2 / self.temperature)
                * 0.0001
            )

        self.henrys_cst_eq = Constraint(rule=henry_cst_rule)


def _add_diffusivity_cst(self):
    self.diffusivity_cst = Var(
        initialize=1e-10,
        bounds=(1e-15, None),
        units=pyunits.m**2 / pyunits.s,
        doc="Diffusivity of carbon dioxide in liquid",
    )


    if self.config.defined_state is False:

        def diffusivity_cst_rule(b):
            return self.diffusivity_cst == self.params.diffusivity_param_1 * exp(
                self.params.diffusivity_param_2 / self.temperature
            ) * ((self.viscosity_water / self.visc_d) ** 0.8)

        self.diffusivity_cst_eq = Constraint(rule=diffusivity_cst_rule)


def _add_heat_of_absorption(self):
    self.deltaH_abs = Var(
        initialize=6e4,
        bounds=(None, None),
        units=pyunits.J / pyunits.mol / pyunits.K,
        doc="Heat of absorption of CO2",
    )

    if self.config.defined_state is False:

        def heat_absorption_rule(b):
            return self.deltaH_abs == - self.params.heat_of_absorption_variation * (
                round(value(const.gas_constant), 3)
                * pyunits.J
                / pyunits.mole
                / pyunits.K
            ) * (-11054 + 4958 * self.alpha + 10163 * (self.alpha**2))

        self.heat_absorption_eq = Constraint(rule=heat_absorption_rule)


def _add_concentration_tot(self):
    self.conc_mol = Var(
        initialize=40000,
        units=pyunits.mol / pyunits.m**3,
        doc="Total concentration of the vapor phase",
    )


def _add_concentration(self):
    self.conc_mol_comp = Var(
        self.component_list,
        initialize=40000 / 3,
        units=pyunits.mol / pyunits.m**3,
        doc="Concentration of components",
    )

    def concentration_rule(b, j):
        return self.conc_mol_comp[j] == self.mole_frac_comp[j] * self.conc_mol

    self.concentration_eq = Constraint(self.component_list, rule=concentration_rule)


def _add_equilibrium_cst(self):
    self.param_A = Reference(self.params.lnK_param_A)
    self.param_B = Reference(self.params.lnK_param_B)
    self.param_C = Reference(self.params.lnK_param_C)

    reactions = [1, 2, 3, 4, 5, 6, 7]

    self.log_k_eq = Var(
        reactions,
        initialize=-20,
        bounds=(None, 0),
        units=pyunits.dimensionless,
        doc="Log of equilibrium constants for speciation reactions",
    )

    if self.config.defined_state is False:

        def equilibrium_cst_rule(b, j):
            if j == 1:
                return self.log_k_eq[j] == log(self.params.log_k_eq_1_variation) + self.param_A[j] + self.param_B[
                    j
                ] / self.temperature + self.param_C[j] * log(
                    self.temperature / pyunits.K
                )
            elif j == 2:
                return self.log_k_eq[j] == log(self.params.log_k_eq_2_variation) + self.param_A[j] + self.param_B[
                    j
                ] / self.temperature + self.param_C[j] * log(
                    self.temperature / pyunits.K
                )
            elif j == 3:
                return self.log_k_eq[j] == log(self.params.log_k_eq_3_variation) + self.param_A[j] + self.param_B[
                    j
                ] / self.temperature + self.param_C[j] * log(
                    self.temperature / pyunits.K
                )
            elif j == 4:
                return self.log_k_eq[j] == log(self.params.log_k_eq_4_variation) + self.param_A[j] + self.param_B[
                    j
                ] / self.temperature + self.param_C[j] * log(
                    self.temperature / pyunits.K
                )
            elif j == 5:
                return self.log_k_eq[j] == log(self.params.log_k_eq_5_variation) + self.param_A[j] + self.param_B[
                    j
                ] / self.temperature + self.param_C[j] * log(
                    self.temperature / pyunits.K
                )
            elif j == 5:
                return self.log_k_eq[j] == log(self.params.log_k_eq_6_variation) + self.param_A[j] + self.param_B[
                    j
                ] / self.temperature + self.param_C[j] * log(
                    self.temperature / pyunits.K
                )
            else:
                return (
                    self.log_k_eq[j]
                    == log(self.params.log_k_eq_7_variation) + self.param_A[j] + self.param_B[j] / self.temperature
                )

        self.equilibrium_cst_eq = Constraint(reactions, rule=equilibrium_cst_rule)


def _add_concentration_spec(self):
    speciation_components = [
        "HCO3-",
        "H3O+",
        "CO2",
        "H2O",
        "CO3-2",
        "OH-",
        "PZ",
        "PZH+",
        "PZCOO-",
        "H+PZCOO-",
        "PZ(COO-)2",
    ]

    c_spec_init = {
        "HCO3-": 15.05,
        "H3O+": 2.09e-7,
        "CO2": 3.13e-7,
        "H2O": 35209.2,
        "CO3-2": 2.36,
        "OH-": 0.0337,
        "PZ": 878,
        "PZH+": 1161.9,
        "PZCOO-": 803.75,
        "H+PZCOO-": 558.4,
        "PZ(COO-)2": 169.2,
    }

    def spec_concentration_init(b, j):
        return c_spec_init[j]

    self.concentration_spec = Var(
        speciation_components,
        initialize=spec_concentration_init,
        bounds=(1e-14, None),
        units=pyunits.mol / pyunits.m**3,
        doc="Concentrations of all species that is "
        "involved in reactions in the liquid film",
    )

    self.ln_concentration_spec = Var(
        speciation_components,
        initialize=log(40000 / 11),
        bounds=(log(1e-14), None),
        units=pyunits.dimensionless,
        doc="Log of speciation concentrations",
    )

    if self.config.defined_state is False:

        def ln_concentration_spec_rule(b, j):
            return self.ln_concentration_spec[j] == log(
                self.concentration_spec[j] * pyunits.m**3 / pyunits.mole
            )

        self.ln_concentration_spec_eq = Constraint(
            speciation_components, rule=ln_concentration_spec_rule
        )
    if self.config.defined_state is False:

        def reaction_1_rule(b):
            return self.log_k_eq[1] + self.ln_concentration_spec[
                "CO2"
            ] + 2 * self.ln_concentration_spec["H2O"] == self.ln_concentration_spec[
                "HCO3-"
            ] + self.ln_concentration_spec[
                "H3O+"
            ] + log(
                self.conc_mol * pyunits.m**3 / pyunits.mole
            )

        self.reaction_1_eq = Constraint(rule=reaction_1_rule)

        def reaction_2_rule(b):
            return (
                self.log_k_eq[2]
                + self.ln_concentration_spec["HCO3-"]
                + self.ln_concentration_spec["H2O"]
                == self.ln_concentration_spec["H3O+"]
                + self.ln_concentration_spec["CO3-2"]
            )

        self.reaction_2_eq = Constraint(rule=reaction_2_rule)

        def reaction_3_rule(b):
            return (
                self.log_k_eq[3] + 2 * self.ln_concentration_spec["H2O"]
                == self.ln_concentration_spec["H3O+"]
                + self.ln_concentration_spec["OH-"]
            )

        self.reaction_3_eq = Constraint(rule=reaction_3_rule)

        def reaction_4_rule(b):
            return (
                self.log_k_eq[4]
                + self.ln_concentration_spec["H2O"]
                + self.ln_concentration_spec["PZH+"]
                == self.ln_concentration_spec["PZ"] + self.ln_concentration_spec["H3O+"]
            )

        self.reaction_4_eq = Constraint(rule=reaction_4_rule)

        def reaction_5_rule(b):
            return self.log_k_eq[5] + self.ln_concentration_spec[
                "PZ"
            ] + self.ln_concentration_spec["CO2"] + self.ln_concentration_spec[
                "H2O"
            ] == self.ln_concentration_spec[
                "H3O+"
            ] + self.ln_concentration_spec[
                "PZCOO-"
            ] + log(
                self.conc_mol * pyunits.m**3 / pyunits.mole
            )

        self.reaction_5_eq = Constraint(rule=reaction_5_rule)

        def reaction_6_rule(b):
            return (
                self.log_k_eq[6]
                + self.ln_concentration_spec["H+PZCOO-"]
                + self.ln_concentration_spec["H2O"]
                == self.ln_concentration_spec["PZCOO-"]
                + self.ln_concentration_spec["H3O+"]
            )

        self.reaction_6_eq = Constraint(rule=reaction_6_rule)

        def reaction_7_rule(b):
            return self.log_k_eq[7] + self.ln_concentration_spec[
                "PZCOO-"
            ] + self.ln_concentration_spec["CO2"] + self.ln_concentration_spec[
                "H2O"
            ] == self.ln_concentration_spec[
                "H3O+"
            ] + self.ln_concentration_spec[
                "PZ(COO-)2"
            ] + log(
                self.conc_mol * pyunits.m**3 / pyunits.mole
            )

        self.reaction_7_eq = Constraint(rule=reaction_7_rule)

        def co2_balance(b):
            return (
                self.conc_mol_comp["CO2"]
                == self.concentration_spec["CO2"]
                + self.concentration_spec["HCO3-"]
                + self.concentration_spec["CO3-2"]
                + self.concentration_spec["PZCOO-"]
                + self.concentration_spec["H+PZCOO-"]
                + 2 * self.concentration_spec["PZ(COO-)2"]
            )

        self.co2_balance_eq = Constraint(rule=co2_balance)

        def pz_balance(b):
            return (
                self.conc_mol_comp["PZ"]
                == self.concentration_spec["PZ"]
                + self.concentration_spec["PZH+"]
                + self.concentration_spec["PZCOO-"]
                + self.concentration_spec["H+PZCOO-"]
                + self.concentration_spec["PZ(COO-)2"]
            )

        self.pz_balance_eq = Constraint(rule=pz_balance)

        def h2o_balance(b):
            return (
                self.conc_mol_comp["H2O"]
                == self.concentration_spec["H2O"]
                + self.concentration_spec["OH-"]
                + self.concentration_spec["H3O+"]
            )

        self.h2o_balance_eq = Constraint(rule=h2o_balance)

        def charge_balance(b):
            return (
                self.concentration_spec["H3O+"] + self.concentration_spec["PZH+"]
                == 2 * self.concentration_spec["CO3-2"]
                + 2 * self.concentration_spec["PZ(COO-)2"]
                + self.concentration_spec["HCO3-"]
                + self.concentration_spec["OH-"]
                + self.concentration_spec["PZCOO-"]
            )

        self.charge_balance_eq = Constraint(rule=charge_balance)


def _add_saturation_pressure(self):
    self.P_sat = Var(
        self.params.volatile_components,
        initialize=500000,
        bounds=(0, None),
        units=pyunits.Pa,
    )

    def saturation_pressure_rule(b, j):
        if j == "CO2":
            return (
                self.P_sat[j] / pyunits.Pa
                == exp(
                    self.equilibrium_pressure_CO2_param_1
                    - self.equilibrium_pressure_CO2_param_2 / self.temperature
                    - self.equilibrium_pressure_CO2_param_3 * (self.alpha**2)
                    + self.equilibrium_pressure_CO2_param_4
                    * self.alpha
                    / self.temperature
                    + self.equilibrium_pressure_CO2_param_5
                    * (self.alpha**2)
                    / self.temperature
                )
                / self.mole_frac_comp[j]
            )
        elif j == "H2O" or j == "PZ":
            return self.P_sat[j] / pyunits.Pa == exp(
                self.equilibrium_pressure_param_1[j]
                - self.equilibrium_pressure_param_2[j] / self.temperature
                - self.equilibrium_pressure_param_3[j]
                * log(self.temperature / pyunits.K)
                + self.equilibrium_pressure_param_4[j] * self.temperature**2
            )

    self.saturation_pressure_eq = Constraint(
        self.params.volatile_components, rule=saturation_pressure_rule
    )


def _prepare_state(blk, state_args, state_vars_fixed):
    # Fix state variables if not already fixed
    if state_vars_fixed is False:
        flags = fix_state_vars(blk, state_args)
    else:
        flags = None

    # for k in blk.keys():
    #     if blk[k].config.defined_state is False:
    #           blk[k].sum_f_l_j.deactivate()
    #     blk[k].conc_mol.fix()

    # Check that degrees of freedom are zero after fixing state vars
    for k in blk.keys():
        if degrees_of_freedom(blk[k]) != 0:
            raise Exception(
                "State vars fixed but degrees of freedom "
                "for state block is not zero during "
                "initialization."
            )

    return flags


def _initialize_state(blk, solver, init_log, solve_log):
    # Check that there is something to solve for
    free_vars = 0
    for k in blk.keys():
        free_vars += number_unfixed_variables(blk[k])
    if free_vars > 0:
        # If there are free variables, call the solver to initialize
        try:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solve_indexed_blocks(solver, [blk], tee=True)  # slc.tee)
        except:
            res = None
    else:
        res = None

    init_log.info("Properties Initialized {}.".format(idaeslog.condition(res)))


def _restore_state(blk, flags, hold_state):
    # Return state to initial conditions
    if hold_state is True:
        return flags
    else:
        blk.release_state(flags)


def _unfix_state(blk, flags, outlvl):
    init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")

    # for k in blk.keys():
    #     if blk[k].config.defined_state is False:
    #         blk[k].sum_f_l_j.activate()
    #     blk[k].conc_mol.unfix()

    if flags is not None:
        # Unfix state variables
        revert_state_vars(blk, flags)

    init_log.info_high("State Released.")
