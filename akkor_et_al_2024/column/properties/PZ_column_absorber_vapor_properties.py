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
#  Chrysanthos E. Gounaris, as part of a research project funded by The Dow Chemical
#  Company. In particular, we acknowledge the close collaboration with the following
#  researchers from Dow: Shachit S. Iyer, John Dowdle and Le Wang,
#  who provided key contributions pertaining to the project conceptualization,
#  the research design and the methodology, model design choices and rationales,
#  and the discussions of the results.

# If you find this code useful for your research, please consider citing
# "Mathematical Modeling and Economic Optimization of a Piperazine-Based
# Post-Combustion Carbon Capture Process" by Ilayda Akkor, Shachit S. Iyer,
# John Dowdle, Le Wang and Chrysanthos E. Gounaris


"""
Vapor phase properties for feed gas containing CO2-H2O-N2, and PZ that goes into the vapor phase in the absorber
"""

# Import components from Pyomo
from pyomo.environ import (
    Constraint,
    Expression,
    Reference,
    Param,
    units as pyunits,
    Var,
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
    VaporPhase,
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

        # Define Phase objects for all phases
        self.Vap = VaporPhase()

        # Define basic parameters
        self.mw_comp = Param(
            self.component_list,
            mutable=True,
            initialize={
                "CO2": 0.04401,
                "H2O": 0.01802,
                "PZ": 0.086136,
                "N2": 0.02801,
            },
            units=pyunits.kg / pyunits.mol,
            doc="Molecular weight",
        )

        self.diff_vol = Param(
            ["CO2", "H2O", "PZ"],
            mutable=True,
            initialize={"CO2": 26.9, "H2O": 12.7, "PZ": 97.2},
            units=pyunits.dimensionless,
            doc="Diffusion volumes of molecules",
        )

        self.heat_capacity_param_A = Param(
            ["CO2", "H2O", "N2"],
            mutable=True,
            initialize={"CO2": 18.86, "H2O": 33.80, "N2": 30.81},
            doc="Heat capacity equation parameters",
        )

        self.heat_capacity_param_B = Param(
            ["CO2", "H2O", "N2"],
            mutable=True,
            initialize={"CO2": 0.07937, "H2O": -0.00795, "N2": -0.01187},
            doc="Heat capacity equation parameters",
        )

        self.heat_capacity_param_C = Param(
            ["CO2", "H2O", "N2"],
            mutable=False,
            initialize={"CO2": -6.7834e-5, "H2O": 2.8228e-5, "N2": 2.3968e-5},
            doc="Heat capacity equation parameters",
        )

        self.heat_capacity_param_D = Param(
            ["CO2", "H2O", "N2"],
            mutable=False,
            initialize={"CO2": 2.4426e-8, "H2O": -1.3115e-8, "N2": -1.0176e-8},
            doc="Heat capacity equation parameters",
        )

        self.viscosity_param_A = Param(
            ["CO2", "H2O", "N2"],
            mutable=False,
            initialize={"CO2": 578.08, "H2O": 658.25, "N2": 90.30},
            doc="Viscosity parameters",
        )

        self.viscosity_param_B = Param(
            ["CO2", "H2O", "N2"],
            mutable=False,
            initialize={"CO2": 185.24, "H2O": 283.16, "N2": 46.14},
            doc="Viscosity parameters",
        )

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "temperature": {"method": None},
                "pressure": {"method": None},
                "flow_mol": {"method": None},
                "mole_frac_comp": {"method": None},
                "flow_mol_comp": {"method": "_flow_mol_comp"},
                "mw_comp": {"method": "_mw_comp"},
                "dens_mass": {"method": "_dens_mass"},
                "visc_d": {"method": "_visc_d"},
                "diffus_comp": {"method": "_diffus_comp"},
                "conc_mol_comp": {"method": "_conc_mol_comp"},
                "conc_mol": {"method": "_conc_mol"},
                "cp_mol_comp": {"method": "_cp_mol_comp"},
                "pressure_comp": {"method": "_pressure_comp"},
            }
        )

        obj.define_custom_properties(
            {
                "cp_vol": {"method": "_cp_vol"},
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
            initialize=0.2,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Component mole fractions",
        )
        self.pressure = Var(
            initialize=104500,
            bounds=(101300, None),
            units=pyunits.Pa,
            doc="State pressure",
        )
        self.temperature = Var(
            initialize=313.15,
            bounds=(298.15, 655),
            units=pyunits.K,
            doc="State temperature",
        )
        self.flow_mol = Var(
            initialize=20,
            bounds=(0, None),
            units=pyunits.mol / pyunits.s,
            doc="Molar flowrate",
        )

    def _visc_d(self):
        _add_viscosity(self)

    def _flow_mol_comp(self):
        _add_flow_mol_comp(self)

    def _cp_mol_comp(self):
        _add_heat_capacity_comp(self)

    def _dens_mass(self):
        _add_density(self)

    def _mw_comp(self):
        _add_molecular_weight(self)

    def _conc_mol_comp(self):
        _add_concentration(self)

    def _conc_mol(self):
        _add_concentration_tot(self)

    def _pressure_comp(self):
        _add_pressure_comp(self)

    def _cp_vol(self):
        _add_cp_vol(self)

    def _diffus_comp(self):
        _add_diffusivity(self)

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
        "pressure": self.pressure,
        "flow_mol": self.flow_mol,
    }


def _add_flow_mol_comp(self):
    y_in = {"CO2": 0.041, "H2O": 0.0875, "PZ": 0, "N2": 0.8715}

    def gas_component_flowrate_init(m, j):
        return y_in[j] * 20

    self.flow_mol_comp = Var(
        self.component_list,
        initialize=gas_component_flowrate_init,
        bounds=(0, None),
        doc="Molar flowrate of gas per component(mol/s)",
    )

    def gas_flowrate_component_rule(blk, j):
        return blk.flow_mol_comp[j] == blk.mole_frac_comp[j] * blk.flow_mol

    self.gas_flowrate_component_eq = Constraint(
        self.component_list,
        rule=gas_flowrate_component_rule,
        doc="Gas flowrate per component",
    )

    if self.config.defined_state is False:

        def sum_f_g_j(blk):
            return sum(blk.flow_mol_comp[j] for j in blk.component_list) == blk.flow_mol

        self.sum_f_g_j = Constraint(rule=sum_f_g_j)


def _add_molecular_weight(self):
    self.mw_comp = Reference(self.params.mw_comp)


def _add_density(self):
    self.dens_mass = Var(
        initialize=1, units=pyunits.kg / pyunits.m**3, doc="Mixture density"
    )

    if self.config.defined_state is False:

        def dens_rule(blk):
            return blk.dens_mass * 8.314 * blk.temperature == sum(
                blk.pressure * blk.mole_frac_comp[j] * blk.mw_comp[j]
                for j in self.component_list
            )

        self.ideal_gas_eq = Constraint(rule=dens_rule)


def _add_diffusivity(self):
    self.diff_vol = Reference(self.params.diff_vol)

    self.diffus_comp = Var(
        ["CO2", "H2O", "PZ"],
        initialize=1.8e-5,
        units=pyunits.m**2 / pyunits.s,
        bounds=(1e-7, None),
        doc="Binary diffusivity constant (Fuller's equation)",
    )

    if self.config.defined_state is False:

        def diff_rule(b, j):
            return (
                b.diffus_comp[j]
                * (
                    b.pressure
                    / 1000
                    * 0.00987
                    * (b.diff_vol[j] ** (1.0 / 3.0) + 20.1 ** (1.0 / 3.0)) ** 2
                )
                == 10 ** (-7)
                * (b.temperature**1.75)
                * (1 / (b.mw_comp[j] * 1000) + 1 / 29) ** 0.5
            )

        self.diffusivity_eq = Constraint(["CO2", "H2O", "PZ"], rule=diff_rule)


def _add_concentration_tot(self):
    self.conc_mol = Var(
        initialize=40,
        units=pyunits.mol / pyunits.m**3,
        bounds=(0, None),
        doc="Total concentration of the vapor phase",
    )

    def concentration_tot_rule(b):
        return b.pressure == b.conc_mol * 8.314 * b.temperature

    self.concentration_tot_eq = Constraint(rule=concentration_tot_rule)


def _add_concentration(self):
    self.conc_mol_comp = Var(
        self.component_list,
        initialize=10,
        units=pyunits.mol / pyunits.m**3,
        bounds=(0, None),
        doc="Concentration of components",
    )

    def concentration_rule(b, j):
        return b.conc_mol_comp[j] == b.mole_frac_comp[j] * b.conc_mol

    self.concentration_eq = Constraint(self.component_list, rule=concentration_rule)


def _add_pressure_comp(self):
    self.pressure_comp = Var(
        self.component_list, initialize=10000, units=pyunits.Pa, bounds=(0, None)
    )

    if self.config.defined_state is False:

        def pressure_comp_rule(b, j):
            return b.pressure_comp[j] == b.pressure * b.mole_frac_comp[j]

        self.pressure_comp_eq = Constraint(self.component_list, rule=pressure_comp_rule)


def _add_heat_capacity_comp(self):
    self.cp_mol_comp = Var(
        ["CO2", "H2O", "N2"],
        initialize=30,
        bounds=(0.1, None),
        units=pyunits.J / pyunits.mol / pyunits.K,
        doc="Componentwise heat capacities, "
        "(PZ excluded since its concentration in vapor"
        "phase is very small",
    )

    self.A = Reference(self.params.heat_capacity_param_A)
    self.B = Reference(self.params.heat_capacity_param_B)
    self.C = Reference(self.params.heat_capacity_param_C)
    self.D = Reference(self.params.heat_capacity_param_D)

    def heat_capacity_comp_rule(b, j):
        return (
            b.cp_mol_comp[j]
            == b.A[j]
            + b.B[j] * b.temperature
            + b.C[j] * b.temperature**2
            + b.D[j] * b.temperature**3
        )

    self.heat_capacity_comp_eq = Constraint(
        ["CO2", "H2O", "N2"], rule=heat_capacity_comp_rule
    )


def _add_cp_vol(self):
    self.cp_vol = Var(
        initialize=1100,
        bounds=(0, None),
        units=pyunits.J / pyunits.m**3 / pyunits.K,
        doc="Phase heat capacity",
    )

    def heat_capacity_rule(b):
        return b.cp_vol == sum(
            b.cp_mol_comp[j] * b.conc_mol_comp[j] for j in ["CO2", "H2O", "N2"]
        )

    self.heat_capacity_eq = Constraint(rule=heat_capacity_rule)


def _add_viscosity(self):
    self.viscosity_comp = Var(
        ["H2O", "CO2", "N2"],
        initialize=1e-3,
        bounds=(1e-10, None),
        units=pyunits.g / pyunits.cm / pyunits.s,
        doc="Component viscosities",
    )

    self.A_vis = Reference(self.params.viscosity_param_A)
    self.B_vis = Reference(self.params.viscosity_param_B)

    if self.config.defined_state is False:

        def viscosity_comp_rule(b, j):
            return (
                b.viscosity_comp[j]
                == (10 ** (b.A_vis[j] * (1 / b.temperature - 1 / b.B_vis[j]))) * 0.01
            )

        self.viscosity_comp_eq = Constraint(
            ["H2O", "CO2", "N2"],
            rule=viscosity_comp_rule,
            doc="Expression for viscosities in cp, "
            "converted to g/cm/s to be plugged"
            "in the Wilke equation",
        )

    self.mu_intermediate = Var([1, 2, 3])

    if self.config.defined_state is False:

        def mu_intermediate_rule(b, j):
            if j == 1:
                cst_co2_h2o = (
                    (2**0.5)
                    * (
                        1
                        + (self.viscosity_comp["CO2"] / self.viscosity_comp["H2O"])
                        ** 0.5
                        * (self.mw_comp["H2O"] / self.mw_comp["CO2"]) ** 0.25
                    )
                    ** 2
                    / 4
                    / ((1 + self.mw_comp["CO2"] / self.mw_comp["H2O"]) ** 0.5)
                )

                cst_co2_n2 = (
                    (2**0.5)
                    * (
                        1
                        + (self.viscosity_comp["CO2"] / self.viscosity_comp["N2"])
                        ** 0.5
                        * (self.mw_comp["N2"] / self.mw_comp["CO2"]) ** 0.25
                    )
                    ** 2
                    / 4
                    / ((1 + self.mw_comp["CO2"] / self.mw_comp["N2"]) ** 0.5)
                )

                return (
                    self.mu_intermediate[1]
                    * (
                        self.mole_frac_comp["CO2"]
                        + cst_co2_h2o * self.mole_frac_comp["H2O"]
                        + cst_co2_n2 * self.mole_frac_comp["N2"]
                    )
                    == self.viscosity_comp["CO2"] * self.mole_frac_comp["CO2"]
                )

            elif j == 2:
                cst_h2o_co2 = (
                    (2**0.5)
                    * (
                        1
                        + (self.viscosity_comp["H2O"] / self.viscosity_comp["CO2"])
                        ** 0.5
                        * (self.mw_comp["CO2"] / self.mw_comp["H2O"]) ** 0.25
                    )
                    ** 2
                    / 4
                    / ((1 + self.mw_comp["H2O"] / self.mw_comp["CO2"]) ** 0.5)
                )

                cst_h2o_n2 = (
                    (2**0.5)
                    * (
                        1
                        + (self.viscosity_comp["H2O"] / self.viscosity_comp["N2"])
                        ** 0.5
                        * (self.mw_comp["N2"] / self.mw_comp["H2O"]) ** 0.25
                    )
                    ** 2
                    / 4
                    / ((1 + self.mw_comp["H2O"] / self.mw_comp["N2"]) ** 0.5)
                )

                return (
                    self.mu_intermediate[2]
                    * (
                        self.mole_frac_comp["H2O"]
                        + cst_h2o_co2 * self.mole_frac_comp["CO2"]
                        + cst_h2o_n2 * self.mole_frac_comp["N2"]
                    )
                    == self.viscosity_comp["H2O"] * self.mole_frac_comp["H2O"]
                )

            else:
                cst_n2_co2 = (
                    (2**0.5)
                    * (
                        1
                        + (self.viscosity_comp["N2"] / self.viscosity_comp["CO2"])
                        ** 0.5
                        * (self.mw_comp["CO2"] / self.mw_comp["N2"]) ** 0.25
                    )
                    ** 2
                    / 4
                    / ((1 + self.mw_comp["N2"] / self.mw_comp["CO2"]) ** 0.5)
                )

                cst_n2_h2o = (
                    (2**0.5)
                    * (
                        1
                        + (self.viscosity_comp["N2"] / self.viscosity_comp["H2O"])
                        ** 0.5
                        * (self.mw_comp["H2O"] / self.mw_comp["N2"]) ** 0.25
                    )
                    ** 2
                    / 4
                    / ((1 + self.mw_comp["N2"] / self.mw_comp["H2O"]) ** 0.5)
                )

                return (
                    self.mu_intermediate[3]
                    * (
                        self.mole_frac_comp["N2"]
                        + cst_n2_co2 * self.mole_frac_comp["CO2"]
                        + cst_n2_h2o * self.mole_frac_comp["H2O"]
                    )
                    == self.viscosity_comp["N2"] * self.mole_frac_comp["N2"]
                )

        self.mu_intermediate_eq = Constraint(
            [1, 2, 3], rule=mu_intermediate_rule, doc="Wilke (1950)"
        )

    self.visc_d = Var(
        initialize=0.02,
        units=pyunits.cp,
        bounds=(1e-5, None),
        doc="Phase dynamic viscosity",
    )

    if self.config.defined_state is False:

        def viscosity_rule(b):
            return (
                self.visc_d
                == (b.mu_intermediate[1] + b.mu_intermediate[2] + b.mu_intermediate[3])
                * 100
            )

        self.viscosity_eq = Constraint(rule=viscosity_rule)


def _prepare_state(blk, state_args, state_vars_fixed):
    # Fix state variables if not already fixed
    if state_vars_fixed is False:
        flags = fix_state_vars(blk, state_args)
    else:
        flags = None

    # Deactivate sum of mole fractions constraint
    for k in blk.keys():
        if blk[k].config.defined_state is False:
            blk[k].mole_fraction_constraint.deactivate()

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

    # Reactivate sum of mole fractions constraint
    for k in blk.keys():
        if blk[k].config.defined_state is False:
            blk[k].mole_fraction_constraint.activate()

    if flags is not None:
        # Unfix state variables
        revert_state_vars(blk, flags)

    init_log.info_high("State Released.")
