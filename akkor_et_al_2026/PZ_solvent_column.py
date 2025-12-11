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
Packed solvent column model for PZ system
"""

import copy

# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    Param,
    Expression,
    TransformationFactory,
    Constraint,
    value,
    exp,
    sin,
    ceil,
    units as pyunits,
)
from pyomo.dae import ContinuousSet
from pyomo.common.config import ConfigBlock, ConfigValue, Bool, In

# Import IDAES Libraries
from idaes.core.base.unit_model import UnitModelBlockData
from idaes.core.util.constants import Constants as const
from idaes.core import declare_process_block_class, useDefault
import idaes.logger as idaeslog
from idaes.core.util.config import DefaultBool
from idaes.core.util.config import is_physical_parameter_block


@declare_process_block_class("PZPackedColumn")
class PZPackedColumnData(UnitModelBlockData):
    """
    PZ Packed Column Model Class.
    """

    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. Solvent Columns do not yet support dynamic behavior.""",
        ),
    )

    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    Must be True if dynamic = True,
    **default** - False.
    **Valid values:** {
    **useDefault** - get flag from parent (default = False),
    **True** - construct holdup terms,
    **False** - do not construct holdup terms}""",
        ),
    )

    CONFIG.declare(
        "finite_elements",
        ConfigValue(
            default=40,
            domain=int,
            description="Number of finite elements in length domain",
            doc="""Number of finite elements to use when discretizing length
    domain (default=40)""",
        ),
    )

    CONFIG.declare(
        "intercooler",
        ConfigValue(
            domain=Bool,
            default=False,
            description="Intercooler flag",
            doc="""Indicates whether the column model will have an intercooler on not, default = False""",
        ),
    )

    CONFIG.declare(
        "intercooled_temperature",
        ConfigValue(
            domain=float,
            default=306,
            description="Intercooler outlet temperature",
            doc="""Liquid temperature after the intercooler, default=306 K""",
        ),
    )

    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Pressure change term construction flag",
            doc="""Indicates whether the column model will account for pressure drop, **default** - False.
    If true: linear pressure drop with a fixed outlet""",
        ),
    )

    _PhaseCONFIG = ConfigBlock()

    _PhaseCONFIG.declare(
        "property_package",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations
    (default = 'use_parent_value')
    - 'use_parent_value' - get package from parent (default = None)
    - a ParameterBlock object""",
        ),
    )

    _PhaseCONFIG.declare(
        "property_package_args",
        ConfigValue(
            default={},
            description="Arguments for constructing vapor property package",
            doc="""A dict of arguments to be passed to the PropertyBlockData
    and used when constructing these
    (default = 'use_parent_value')
    - 'use_parent_value' - get package from parent (default = None)
    - a dict (see property package for documentation)
                """,
        ),
    )

    CONFIG.declare("vapor_phase", _PhaseCONFIG(doc="vapor side config arguments"))
    CONFIG.declare("liquid_phase", _PhaseCONFIG(doc="liquid side config arguments"))

    def build(self):
        super().build()

        self.length_domain = ContinuousSet(
            bounds=(0, self.config.finite_elements),
            doc="Dimensionless, will get discretized into given number of finite elements",
        )

        # custom discretization for phases
        d0 = dict(**self.config.vapor_phase.property_package_args)
        d0.update(defined_state=True)
        d1 = copy.deepcopy(d0)
        d1["defined_state"] = False

        def idx_map_vap(i):  # i = (t, x)
            if i[1] == self.length_domain.first():
                return 0
            else:
                return 1

        self.vapor_properties = (
            self.config.vapor_phase.property_package.build_state_block(
                self.flowsheet().time,
                self.length_domain,
                doc="Material properties",
                initialize={0: d0, 1: d1},
                idx_map=idx_map_vap,
            )
        )

        d0 = dict(**self.config.vapor_phase.property_package_args)
        d0.update(defined_state=True)
        d1 = copy.copy(d0)
        d1["defined_state"] = False

        def idx_map_liq(i):
            if i[1] == self.length_domain.last():
                return 0
            else:
                return 1

        self.liquid_properties = (
            self.config.liquid_phase.property_package.build_state_block(
                self.flowsheet().time,
                self.length_domain,
                doc="Material properties",
                initialize={0: d0, 1: d1},
                idx_map=idx_map_liq,
            )
        )

        # column design variables
        self.length = Var(
            initialize=12.2,
            bounds=(0, 100),
            units=pyunits.m,
            doc="Column packed length",
        )

        self.diameter = Var(
            initialize=0.66,
            bounds=(0, 100),
            units=pyunits.m,
            doc="Column inner diameter",
        )

        self.cross_sectional_area = Var(
            initialize=0.342,
            bounds=(0, None),
            units=pyunits.m**2,
            doc="Cross sectional area of column",
        )

        self.flux_mol = Var(
            self.liquid_properties.params.volatile_components,
            self.length_domain,
            bounds=(None, None),
            units=pyunits.mol / pyunits.m**2 / pyunits.s,
            doc="Flux of species across phases",
        )

        self.material_transfer_coefficient_tot = Var(
            self.liquid_properties.params.volatile_components,
            self.length_domain,
            bounds=(0, None),
            units=pyunits.mol / pyunits.m**2 / pyunits.kPa / pyunits.s,
            doc="Overall material transfer coefficient",
        )

        self.vapor_velocity = Var(
            self.length_domain,
            initialize=1.456,
            bounds=(0, None),
            units=pyunits.m / pyunits.s,
            doc="Superficial velocity of vapor phase",
        )

        self.liquid_velocity = Var(
            self.length_domain,
            initialize=0.007,
            bounds=(0, None),
            units=pyunits.m / pyunits.s,
            doc="Superficial velocity of liquid phase",
        )

        self.P_drop_z = Param(
            initialize=0.2,
            mutable=True,
            units=pyunits.kPa/pyunits.m
        )

        # Packing parameters
        self.a = Param(
            initialize=250,
            mutable=True,
            units=pyunits.m**2 / pyunits.m**3,
            doc="Specific area",
        )

        self.void = Param(
            initialize=0.96,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Void fraction",
        )

        self.theta = Param(
            initialize=60,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Corrugation angle",
        )

        self.d_e = Expression(expr=4 * self.void / self.a, doc="Equivalent diameter")

        # discretization of the length domain
        zdiscretize = TransformationFactory("dae.finite_difference")

        zdiscretize.apply_to(
            self,
            nfe=self.config.finite_elements,
            wrt=self.length_domain,
            scheme="BACKWARD",
        )

        @self.Constraint(doc="Column cross-sectional area")
        def column_cross_section_area_eq(blk):
            return blk.cross_sectional_area == (const.pi * 0.25 * blk.diameter**2)

        @self.Constraint(self.length_domain, doc="Isothermal gas phase")
        def isothermal_gas_eq(blk, x):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                return (
                    blk.vapor_properties[self.flowsheet().time.first(), x].temperature
                    - blk.vapor_properties[
                        self.flowsheet().time.first(), self.length_domain.prev(x)
                    ].temperature
                    == 0
                )

        @self.Constraint(self.length_domain, doc="Isothermal liquid phase")
        def isothermal_liq_eq(blk, x):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                return (
                    blk.liquid_properties[
                        self.flowsheet().time.first(), self.length_domain.prev(x)
                    ].temperature
                    - blk.liquid_properties[
                        self.flowsheet().time.first(), x
                    ].temperature
                    == 0
                )

        self.driving_force = Var(
            self.liquid_properties.params.volatile_components,
            self.length_domain,
            bounds=(None, None),
        )

        @self.Constraint(
            self.length_domain,
            self.liquid_properties.params.volatile_components,
            doc="Driving force for material flux",
        )
        def driving_force_eq(blk, x, j):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                return (
                    blk.driving_force[j, x]
                    == blk.vapor_properties[
                        self.flowsheet().time.first(), x
                    ].pressure_comp[j]
                    / 1000
                    - blk.liquid_properties[
                        0, self.length_domain.prev(x)
                    ].equilibrium_pressure[j]
                    / 1000
                )

        @self.Constraint(
            self.length_domain,
            self.liquid_properties.params.volatile_components,
            doc="Interfacial flux",
        )
        def flux_eq(blk, x, j):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                if j in self.liquid_properties.params.volatile_components:
                    return (
                        blk.flux_mol[j, x]
                        == blk.material_transfer_coefficient_tot[j, x]
                        * blk.driving_force[j, x]
                    )
                else:
                    return blk.flux_mol[j, x] == 0

        @self.Constraint(
            self.length_domain,
            self.vapor_properties.component_list,
            doc="Gas phase mole balance",
        )
        def gas_mole_balance_eq(blk, x, j):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                expr = (
                    blk.vapor_properties[
                        self.flowsheet().time.first(), x
                    ].flow_mol_comp[j]
                    - blk.vapor_properties[
                        self.flowsheet().time.first(), self.length_domain.prev(x)
                    ].flow_mol_comp[j]
                )
                if j in self.liquid_properties.params.volatile_components:
                    expr += (
                        blk.a
                        * blk.cross_sectional_area
                        * blk.flux_mol[j, x]
                        * (blk.length / self.config.finite_elements)
                    )
                return 0 == expr

        @self.Constraint(
            self.length_domain,
            self.liquid_properties.component_list,
            doc="Liquid phase mole balance",
        )
        def liq_mole_balance_eq(blk, x, j):
            if x == self.length_domain.last():
                return Constraint.Skip
            else:
                expr = (
                    blk.liquid_properties[
                        self.flowsheet().time.first(), x
                    ].flow_mol_comp[j]
                    - blk.liquid_properties[
                        self.flowsheet().time.first(), self.length_domain.next(x)
                    ].flow_mol_comp[j]
                )
                if j in self.liquid_properties.params.volatile_components:
                    expr -= (
                        blk.a
                        * blk.cross_sectional_area
                        * blk.flux_mol[j, self.length_domain.next(x)]
                        * (blk.length / self.config.finite_elements)
                    )

                return 0 == expr

        @self.Constraint(self.length_domain, doc="Vapor phase flow and velocity")
        def vapor_flow_eq(blk, x):
            return (
                blk.vapor_properties[self.flowsheet().time.first(), x].flow_mol
                == blk.vapor_velocity[x]
                * blk.cross_sectional_area
                * blk.vapor_properties[self.flowsheet().time.first(), x].conc_mol
            )

        @self.Constraint(self.length_domain, doc="Liquid phase flow and velocity")
        def liquid_flow_eq(blk, x):
            return (
                blk.liquid_properties[self.flowsheet().time.first(), x].flow_mol
                == blk.liquid_velocity[x]
                * blk.cross_sectional_area
                * blk.liquid_properties[self.flowsheet().time.first(), x].conc_mol
            )

        # material transfer coefficients calculations
        self.effective_vapor_velocity = Var(
            self.length_domain,
            initialize=1.75,
            bounds=(0, None),
            units=pyunits.m / pyunits.s,
            doc="Effective vapor velocity",
        )
        self.effective_liquid_velocity = Var(
            self.length_domain,
            bounds=(0, None),
            units=pyunits.m / pyunits.s,
            doc="Effective liquid velocity",
        )

        self.material_transfer_coefficient_gas = Var(
            self.liquid_properties.params.volatile_components,
            self.length_domain,
            bounds=(0, None),
            units=pyunits.m / pyunits.s,
            doc="Gas-side material transfer coefficient",
        )

        @self.Constraint(self.length_domain, doc="Liquid velocity calculation")
        def liq_velocity_eq(blk, x):
            i = (
                self.length_domain.last()
                - self.length_domain.last() / self.config.finite_elements
            )
            return blk.liquid_velocity[x] * blk.liquid_properties[
                self.flowsheet().time.first(), i
            ].dens_mass * blk.cross_sectional_area == blk.liquid_properties[
                self.flowsheet().time.first(), i
            ].flow_mol * (
                sum(
                    blk.liquid_properties[
                        self.flowsheet().time.first(), i
                    ].mole_frac_comp[j]
                    * blk.liquid_properties[
                        self.flowsheet().time.first(), i
                    ].params.mw_comp[j]
                    * 1000
                    for j in blk.liquid_properties.params.volatile_components
                )
                / 1000
            )

        @self.Constraint(self.length_domain, doc="Effective gas velocity (m/s)")
        def effective_vapor_velocity_eq(blk, x):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                return blk.effective_vapor_velocity[x] == blk.vapor_velocity[
                    x
                ] / blk.void / sin(blk.theta * 2 * const.pi / 360)

        @self.Constraint(self.length_domain, doc="Effective liquid velocity (m/s)")
        def effective_liquid_velocity_eq(blk, x):
            if x == self.length_domain.last():
                return Constraint.Skip
            else:
                return blk.effective_liquid_velocity[x] * 2 * blk.liquid_properties[
                    self.flowsheet().time.first(), x
                ].dens_mass * (
                    (3 * blk.liquid_properties[self.flowsheet().time.first(), x].visc_d)
                    ** (1 / 3)
                ) * (
                    (const.pi * blk.diameter) ** (1 - 1 / 3)
                ) == 3 * (
                    (
                        blk.liquid_properties[
                            self.flowsheet().time.first(), x
                        ].dens_mass
                        * blk.liquid_velocity[x]
                        * blk.cross_sectional_area
                    )
                    ** (1 - 1 / 3)
                ) * (
                    blk.liquid_properties[self.flowsheet().time.first(), x].dens_mass
                    ** 2
                    * 9.81
                ) ** (
                    1 / 3
                )

        self.mass_transfer_coef_gas_variation_CO2 = Param(initialize=1, mutable=True)

        @self.Constraint(
            self.length_domain,
            self.liquid_properties.params.volatile_components,
            doc="Gas-side material transfer coefficient calculation",
        )
        def material_transfer_coefficient_gas_eq(blk, x, j):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                if j == 'CO2':
                    return blk.material_transfer_coefficient_gas[
                               j, x
                           ] == blk.mass_transfer_coef_gas_variation_CO2 * 0.0338 * blk.vapor_properties[
                               self.flowsheet().time.first(), x
                           ].diffus_comp[
                               j
                           ] * (
                                   (
                                           blk.vapor_properties[self.flowsheet().time.first(), x].dens_mass
                                           * blk.d_e
                                           * (
                                                   blk.effective_liquid_velocity[self.length_domain.prev(x)]
                                                   + blk.effective_vapor_velocity[x]
                                           )
                                   )
                                   ** 0.8
                           ) / (
                                   blk.d_e
                                   * (
                                           (
                                                   blk.vapor_properties[
                                                       self.flowsheet().time.first(), x
                                                   ].visc_d
                                                   * (10 ** (-3))
                                           )
                                           ** (0.8 - 0.33)
                                   )
                                   * (
                                           (
                                                   blk.vapor_properties[
                                                       self.flowsheet().time.first(), x
                                                   ].dens_mass
                                                   * blk.vapor_properties[
                                                       self.flowsheet().time.first(), x
                                                   ].diffus_comp[j]
                                           )
                                           ** 0.33
                                   )
                           )
                else:
                    return blk.material_transfer_coefficient_gas[
                        j, x
                    ] == 0.0338 * blk.vapor_properties[
                        self.flowsheet().time.first(), x
                    ].diffus_comp[
                        j
                    ] * (
                        (
                            blk.vapor_properties[self.flowsheet().time.first(), x].dens_mass
                            * blk.d_e
                            * (
                                blk.effective_liquid_velocity[self.length_domain.prev(x)]
                                + blk.effective_vapor_velocity[x]
                            )
                        )
                        ** 0.8
                    ) / (
                        blk.d_e
                        * (
                            (
                                blk.vapor_properties[
                                    self.flowsheet().time.first(), x
                                ].visc_d
                                * (10 ** (-3))
                            )
                            ** (0.8 - 0.33)
                        )
                        * (
                            (
                                blk.vapor_properties[
                                    self.flowsheet().time.first(), x
                                ].dens_mass
                                * blk.vapor_properties[
                                    self.flowsheet().time.first(), x
                                ].diffus_comp[j]
                            )
                            ** 0.33
                        )
                    )

        self.mass_transfer_coef_variation_CO2 = Param(initialize=1, mutable=True)
        self.mass_transfer_coef_variation_H2O = Param(initialize=1, mutable=True)
        self.mass_transfer_coef_variation_PZ = Param(initialize=1, mutable=True)

        @self.Constraint(
            self.length_domain,
            self.liquid_properties.params.volatile_components,
            doc="Overall material transfer coefficient",
        )

        def material_transfer_coefficient_tot_eq(blk, x, j):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                if j == "CO2":
                    return blk.material_transfer_coefficient_tot[j, x] * (
                        8.314
                        * blk.vapor_properties[
                            self.flowsheet().time.first(), x
                        ].temperature
                        * (
                            (
                                blk.liquid_properties[
                                    self.flowsheet().time.first(),
                                    self.length_domain.prev(x),
                                ].reaction_rate
                                * blk.liquid_properties[
                                    self.flowsheet().time.first(),
                                    self.length_domain.prev(x),
                                ].diffusivity_cst
                                * blk.liquid_properties[
                                    self.flowsheet().time.first(),
                                    self.length_domain.prev(x),
                                ].concentration_spec["PZ"]
                            )
                            ** 0.5
                        )
                        + blk.liquid_properties[
                            self.flowsheet().time.first(), self.length_domain.prev(x)
                        ].henry
                        * blk.material_transfer_coefficient_gas[j, x]
                        * 1000
                    ) == blk.mass_transfer_coef_variation_CO2 * 1000 * blk.material_transfer_coefficient_gas[j, x] * (
                        (
                            blk.liquid_properties[
                                self.flowsheet().time.first(),
                                self.length_domain.prev(x),
                            ].reaction_rate
                            * blk.liquid_properties[
                                self.flowsheet().time.first(),
                                self.length_domain.prev(x),
                            ].diffusivity_cst
                            * blk.liquid_properties[
                                self.flowsheet().time.first(),
                                self.length_domain.prev(x),
                            ].concentration_spec["PZ"]
                        )
                        ** 0.5
                    )
                elif j == 'H2O':
                    return (
                        blk.material_transfer_coefficient_tot[j, x]
                        * 8.314
                        * blk.vapor_properties[
                            self.flowsheet().time.first(), x
                        ].temperature
                        == blk.mass_transfer_coef_variation_H2O *
                        blk.material_transfer_coefficient_gas[j, x] * 1000
                    )
                else:
                    return (
                        blk.material_transfer_coefficient_tot[j, x]
                        * 8.314
                        * blk.vapor_properties[
                            self.flowsheet().time.first(), x
                        ].temperature
                        == blk.mass_transfer_coef_variation_PZ *
                        blk.material_transfer_coefficient_gas[j, x] * 1000
                    )

        # material balance expressions
        def x_sum_calc(blk, x):
            return sum(
                blk.liquid_properties[self.flowsheet().time.first(), x].mole_frac_comp[
                    j
                ]
                for j in blk.liquid_properties.component_list
            )

        self.x_sum = Expression(self.length_domain, rule=x_sum_calc)

        def y_sum_calc(blk, x):
            return sum(
                blk.vapor_properties[self.flowsheet().time.first(), x].mole_frac_comp[j]
                for j in blk.vapor_properties.component_list
            )

        self.y_sum = Expression(self.length_domain, rule=y_sum_calc)

        def C_l_tot_check(blk, x):
            return blk.liquid_properties[
                self.flowsheet().time.first(), x
            ].conc_mol - sum(
                blk.liquid_properties[self.flowsheet().time.first(), x].conc_mol_comp[j]
                for j in blk.liquid_properties.component_list
            )

        self.C_l_sum = Expression(self.length_domain, rule=C_l_tot_check)

        def C_g_tot_check(blk, x):
            return blk.vapor_properties[
                self.flowsheet().time.first(), x
            ].conc_mol - sum(
                blk.vapor_properties[self.flowsheet().time.first(), x].conc_mol_comp[j]
                for j in blk.vapor_properties.component_list
            )

        self.C_g_sum = Expression(self.length_domain, rule=C_g_tot_check)

        def mb_check(blk):
            return (
                blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].flow_mol
                + blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].flow_mol
                - blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].flow_mol
                - blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].flow_mol
            ) / (
                blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].flow_mol
                + blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].flow_mol
            )

        self.mb_check = Expression(rule=mb_check)

        def species_mb_check(blk, j):
            return (
                blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].flow_mol
                * blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].mole_frac_comp[j]
                + blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].flow_mol
                * blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].mole_frac_comp[j]
                - blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].flow_mol
                * blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].mole_frac_comp[j]
                - blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].flow_mol
                * blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].mole_frac_comp[j]
            ) / (
                blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].flow_mol
                * blk.vapor_properties[
                    self.flowsheet().time.first(), self.length_domain.first()
                ].mole_frac_comp[j]
                + blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].flow_mol
                * blk.liquid_properties[
                    self.flowsheet().time.first(), self.length_domain.last()
                ].mole_frac_comp[j]
            )

        self.species_mb_check = Expression(
            self.vapor_properties.component_list, rule=species_mb_check
        )

        
        # column is isobaric in the first step of initialization
        @self.Constraint(self.length_domain, doc="Isobaric gas phase")
        def isobaric_gas_eq(blk, x):
              if x == self.length_domain.first():
                  return Constraint.Skip
              else:
                  return (
                      blk.vapor_properties[self.flowsheet().time.first(), x].pressure
                      - blk.vapor_properties[
                          self.flowsheet().time.first(), self.length_domain.prev(x)
                      ].pressure
                      == 0
                  )
        
        # pressure drop model is added later if chosen
        if self.config.has_pressure_change:

            @self.Constraint(
                self.length_domain, doc="Linear pressure drop with fixed outlet"
            )
            def pressure_drop_eq(blk, x):
                if x == self.length_domain.first():
                    return Constraint.Skip
                else:
                    return blk.vapor_properties[
                        self.flowsheet().time.first(), x
                    ].pressure / 1000 - blk.vapor_properties[
                        self.flowsheet().time.first(), self.length_domain.prev(x)
                    ].pressure / 1000 == - self.P_drop_z * blk.length/self.config.finite_elements

        # heat transfer model
        self.heat_transfer_coefficient = Var(
            self.length_domain,
            initialize=65.48,
            bounds=(0, None),
            units=pyunits.W / pyunits.m**2 / pyunits.K,
            doc="Heat transfer coefficient (W/m2/K)",
        )

        self.heat_transfer_coefficient_variation = Param(initialize=1, mutable=True)

        @self.Constraint(
            self.length_domain, doc="Heat transfer coefficient calculation"
        )
        def heat_transfer_coefficient_eq(blk, x):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                return blk.heat_transfer_coefficient[x] * (
                    blk.diameter**0.5
                ) == self.heat_transfer_coefficient_variation * 4.05 * (
                    (
                        blk.vapor_properties[self.flowsheet().time.first(), x].dens_mass
                        * blk.vapor_velocity[x]
                    )
                    ** 0.5
                ) * (
                    blk.vapor_properties[self.flowsheet().time.first(), x].cp_vol
                    ** 0.33
                )

        @self.Constraint(self.length_domain, doc="Gas side energy balance")
        def gas_energy_balance_eq(blk, x):
            if x == self.length_domain.first():
                return Constraint.Skip
            else:
                return blk.vapor_velocity[x] * (
                    blk.vapor_properties[self.flowsheet().time.first(), x].temperature
                    - blk.vapor_properties[
                        self.flowsheet().time.first(), self.length_domain.prev(x)
                    ].temperature
                ) * blk.vapor_properties[self.flowsheet().time.first(), x].cp_vol == (
                    blk.length / self.config.finite_elements
                ) * blk.a * (
                    blk.heat_transfer_coefficient[x]
                    * (
                        blk.liquid_properties[
                            self.flowsheet().time.first(), self.length_domain.prev(x)
                        ].temperature
                        - blk.vapor_properties[
                            self.flowsheet().time.first(), x
                        ].temperature
                    )
                )

        if self.config.intercooler:
            self.T_intercooler = Var(
                units=pyunits.K, doc="Intercooler outlet temperature"
            )
            self.Q_intercooler = Var(
                bounds=(0, None),
                units=pyunits.J / pyunits.s,
                doc="Heat removed by intercooler",
            )

            @self.Constraint(doc="Intercooler duty calculation")
            def Q_intercooler_eq(blk):
                x = ceil(self.config.finite_elements / 2) + 1
                return blk.Q_intercooler == blk.liquid_properties[
                    self.flowsheet().time.first(), x
                ].flow_mol / blk.liquid_properties[
                    self.flowsheet().time.first(), x
                ].conc_mol * blk.liquid_properties[
                    self.flowsheet().time.first(), x
                ].params.cp_vol * (
                    blk.liquid_properties[self.flowsheet().time.first(), x].temperature
                    - blk.T_intercooler
                )

        @self.Constraint(self.length_domain, doc="Liquid side energy balance")
        def liq_energy_balance_eq(blk, x):
            if x == self.length_domain.last():
                return Constraint.Skip
            else:
                if self.config.intercooler:
                    if x == ceil(self.config.finite_elements / 2):
                        return (
                            blk.liquid_properties[
                                self.flowsheet().time.first(), x
                            ].temperature
                            == blk.T_intercooler
                        )
                    else:
                        return (
                            blk.liquid_properties[
                                self.flowsheet().time.first(), x
                            ].temperature
                            - blk.liquid_properties[
                                self.flowsheet().time.first(),
                                self.length_domain.next(x),
                            ].temperature
                        ) * blk.liquid_velocity[x] * blk.liquid_properties[
                            self.flowsheet().time.first(), x
                        ].params.cp_vol == -(
                            blk.length / self.config.finite_elements
                        ) * blk.a * (
                            blk.heat_transfer_coefficient[self.length_domain.next(x)]
                            * (
                                blk.liquid_properties[
                                    self.flowsheet().time.first(), x
                                ].temperature
                                - blk.vapor_properties[
                                    self.flowsheet().time.first(),
                                    self.length_domain.next(x),
                                ].temperature
                            )
                            - blk.liquid_properties[
                                self.flowsheet().time.first(), x
                            ].deltaH_abs
                            * blk.flux_mol["CO2", self.length_domain.next(x)]
                            - blk.liquid_properties[
                                self.flowsheet().time.first(), x
                            ].params.deltaH_vap
                            * blk.flux_mol["H2O", self.length_domain.next(x)]
                        )
                else:
                    return (
                        blk.liquid_properties[
                            self.flowsheet().time.first(), x
                        ].temperature
                        - blk.liquid_properties[
                            self.flowsheet().time.first(), self.length_domain.next(x)
                        ].temperature
                    ) * blk.liquid_velocity[x] * blk.liquid_properties[
                        self.flowsheet().time.first(), x
                    ].params.cp_vol == -(
                        blk.length / self.config.finite_elements
                    ) * blk.a * (
                        blk.heat_transfer_coefficient[self.length_domain.next(x)]
                        * (
                            blk.liquid_properties[
                                self.flowsheet().time.first(), x
                            ].temperature
                            - blk.vapor_properties[
                                self.flowsheet().time.first(),
                                self.length_domain.next(x),
                            ].temperature
                        )
                        - blk.liquid_properties[
                            self.flowsheet().time.first(), x
                        ].deltaH_abs
                        * blk.flux_mol["CO2", self.length_domain.next(x)]
                        - blk.liquid_properties[
                            self.flowsheet().time.first(), x
                        ].params.deltaH_vap
                        * blk.flux_mol["H2O", self.length_domain.next(x)]
                    )

        # flooding calculations
        self.flooding_velocity = Var(self.length_domain)

        @self.Constraint(self.length_domain, doc="Flooding velocity calculation")
        def flooding_velocity_eq(blk, x):
            if x == self.length_domain.first() or x == self.length_domain.last():
                return Constraint.Skip
            else:
                return (
                    blk.flooding_velocity[x]
                    == (
                        (9.8 * (blk.void**3) / blk.a)
                        * (
                            blk.liquid_properties[
                                self.flowsheet().time.first(), x
                            ].dens_mass
                            / blk.vapor_properties[
                                self.flowsheet().time.first(), x
                            ].dens_mass
                        )
                        * (
                            (
                                blk.liquid_properties[
                                    self.flowsheet().time.first(), x
                                ].visc_d
                                / blk.liquid_properties[
                                    self.flowsheet().time.first(), x
                                ].viscosity_water
                            )
                            ** (-0.2)
                        )
                        * exp(
                            -4
                            * (
                                (
                                    (
                                        (
                                            blk.liquid_properties[
                                                self.flowsheet().time.first(), x
                                            ].flow_mol
                                            * (
                                                sum(
                                                    blk.liquid_properties[
                                                        self.flowsheet().time.first(), x
                                                    ].mole_frac_comp[j]
                                                    * blk.liquid_properties[
                                                        self.flowsheet().time.first(),
                                                        self.length_domain.first(),
                                                    ].params.mw_comp[j]
                                                    * 1000
                                                    for j in blk.liquid_properties.component_list
                                                )
                                                / 1000
                                            )
                                        )
                                        / (
                                            blk.vapor_properties[
                                                self.flowsheet().time.first(), x
                                            ].flow_mol
                                            * (
                                                sum(
                                                    blk.vapor_properties[
                                                        self.flowsheet().time.first(), x
                                                    ].mole_frac_comp[j]
                                                    * blk.vapor_properties[
                                                        self.flowsheet().time.first(),
                                                        self.length_domain.first(),
                                                    ].params.mw_comp[j]
                                                    * 1000
                                                    for j in blk.vapor_properties.component_list
                                                )
                                                / 1000
                                            )
                                        )
                                    )
                                    * (
                                        (
                                            blk.vapor_properties[
                                                self.flowsheet().time.first(), x
                                            ].dens_mass
                                            / blk.liquid_properties[
                                                self.flowsheet().time.first(), x
                                            ].dens_mass
                                        )
                                        ** 0.5
                                    )
                                )
                                ** 0.25
                            )
                        )
                    )
                    ** 0.5
                )

    def initialize(blk, solver, outlvl):
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        
        # list of variables to be fixed for the first step of initialization
        base_vars = [
            "material_transfer_coefficient_tot",
            "effective_vapor_velocity",
            "effective_liquid_velocity",
            "material_transfer_coefficient_gas",
            "heat_transfer_coefficient",
            "flux_mol"]
        intercooler_and_pressure_drop_vars = [
            "T_intercooler",
            "Q_intercooler",
            "flooding_velocity"
        ]

        # list of equations to be deactivated for the first step of initialization
        base_eqs = [
            "flux_eq",
            "liq_velocity_eq",
            "effective_vapor_velocity_eq",
            "effective_liquid_velocity_eq",
            "material_transfer_coefficient_gas_eq",
            "material_transfer_coefficient_tot_eq",
            "heat_transfer_coefficient_eq"]
        intercooler_and_pressure_drop_eqs = ["pressure_drop_eq", "Q_intercooler_eq", "flooding_velocity_eq"]
        energy_balance_eqs = ["gas_energy_balance_eq", "liq_energy_balance_eq"]

        for v in blk.component_objects(Var):
            if v.local_name in base_vars:
                v.fix()
            if v.local_name in intercooler_and_pressure_drop_vars:
                v.fix()
        blk.flux_mol.fix(0)

        for c in blk.component_objects(Constraint):
            if c.local_name in base_eqs:
                c.deactivate()
            if c.local_name in intercooler_and_pressure_drop_eqs:
                c.deactivate()
            if c.local_name in energy_balance_eqs:
                c.deactivate()

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            results = solver.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Step 1 - Base column: {}.".format(idaeslog.condition(results))
        )

        for v in blk.component_objects(Var):
            if v.local_name in base_vars:
                v.unfix()

        for c in blk.component_objects(Constraint):
            if c.local_name in base_eqs:
                c.activate()

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            results = solver.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Step 2 - Mass transfer: {}.".format(idaeslog.condition(results))
        )

        # deactivate isothermal equations to add energy balance
        blk.isothermal_gas_eq.deactivate()
        blk.isothermal_liq_eq.deactivate()

        for v in blk.component_objects(Var):
            if v.local_name in intercooler_and_pressure_drop_vars:
                v.unfix()
                if v.local_name == "T_intercooler":
                    # if intercooler is selected fix the temperature
                    v.fix(blk.config.intercooled_temperature)
        for c in blk.component_objects(Constraint):
            if c.local_name in energy_balance_eqs:
                c.activate()
            if c.local_name in intercooler_and_pressure_drop_eqs:
                c.activate()
                if c.local_name == "pressure_drop_eq":
                    # if pressure drop is selected then deactivate the isobaric equation
                    blk.isobaric_gas_eq.deactivate()

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            results = solver.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Step 3 - Pressure Drop and Heat transfer: {}.".format(
                idaeslog.condition(results)
            )
        )

    def check_model(blk):
        if value(blk.mb_check.expr) >= 1e-8 or value(blk.mb_check.expr) <= -1e-8:
            print(
                "Warning: mass balance does not close off in the column, value: "
                + str(value(blk.mb_check.expr))
                + " mol/s"
            )
        for j in blk.liquid_properties.component_list:
            if (
                value(blk.species_mb_check[j]) >= 1e-8
                or value(blk.species_mb_check[j]) <= -1e-8
            ):
                print(
                    "Warning: species mass balance does not close off in the column for "
                    + j
                    + ", value: "
                    + str(value(blk.mb_check.expr))
                    + " mol/s"
                )
        for i in blk.length_domain:
            if value(blk.x_sum[i].expr) < 0.9999 or value(blk.x_sum[i].expr) > 1.0001:
                print(
                    "Warning: sum of liquid mole fractions in the column doesn"
                    "t add up to 1"
                )
            if value(blk.y_sum[i].expr) < 0.9999 or value(blk.y_sum[i].expr) > 1.0001:
                print(
                    "Warning: sum of gas mole fractions in the column doesn"
                    "t add up to 1"
                )
            if (
                value(blk.C_l_sum[i].expr) >= 1e-5
                or value(blk.C_l_sum[i].expr) <= -1e-5
            ):
                print(
                    "Warning: sum of concentrations miscalculated in the column for liquid"
                    + str(value(blk.C_l_sum[i].expr))
                    + " mol/m3"
                )
            if (
                value(blk.C_g_sum[i].expr) >= 1e-5
                or value(blk.C_g_sum[i].expr) <= -1e-5
            ):
                print(
                    "Warning: sum of concentrations in the column miscalculated for gas"
                    + str(value(blk.C_g_sum[i].expr))
                    + " mol/m3"
                )
