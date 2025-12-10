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


from pyomo.environ import *
from pyomo.common.config import ConfigBlock, ConfigValue, Bool, In

# Import IDAES Libraries
from idaes.core.base.unit_model import UnitModelBlockData
from idaes.core import declare_process_block_class, useDefault
from idaes.core.util.config import DefaultBool
from idaes.core.util.config import is_physical_parameter_block


@declare_process_block_class("FlashTank")
class FlashTankData(UnitModelBlockData):
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
            default=useDefault,
            domain=DefaultBool,
            description="Holdup construction flag",
            doc="""Indicates whether holdup terms should be constructed or not.
        Must be True if dynamic = True,
        **default** - False.
        **Valid values:** {
        **useDefault** - get flag from parent (default = False),
        **True** - construct holdup terms,
        **False** - do not construct holdup terms}""",
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

        self.vapor_properties = (
            self.config.vapor_phase.property_package.build_state_block(
                self.flowsheet().time,
                doc="Material properties",
            )
        )

        self.liquid_properties = (
            self.config.liquid_phase.property_package.build_state_block(
                self.flowsheet().time,
                doc="Material properties",
            )
        )

        self.feed_properties = (
            self.config.liquid_phase.property_package.build_state_block(
                self.flowsheet().time,
                doc="Material properties",
            )
        )

        # Estimate saturated pressure values for initialization
        P_sat_init = {"CO2": 5881, "H2O": 472.9, "PZ": 51.6}

        def flash_K_init(m, j):
            return round(P_sat_init[j] / 670, 1)

        self.K_flash = Var(
            self.liquid_properties.component_list,
            initialize=flash_K_init,
            bounds=(1e-5, None),
            doc="Equilibrium constant",
        )

        @self.Constraint(
            self.liquid_properties.component_list, doc="Flash component balance"
        )
        def mb_balance_comp_eq(f, j):
            return (
                f.feed_properties[0].flow_mol * f.feed_properties[0].mole_frac_comp[j]
                == f.vapor_properties[0].flow_mol
                * f.vapor_properties[0].mole_frac_comp[j]
                + f.liquid_properties[0].flow_mol
                * f.liquid_properties[0].mole_frac_comp[j]
            )

        @self.Constraint(
            self.liquid_properties.component_list, doc="Equilibrium between phases"
        )
        def equilibrium_eq(f, j):
            return (
                f.vapor_properties[0].mole_frac_comp[j]
                == f.K_flash[j] * f.liquid_properties[0].mole_frac_comp[j]
            )

        @self.Constraint(
            self.liquid_properties.component_list, doc="Equilibrium between phases"
        )
        def equilibrium_constant_eq(f, j):
            return (
                f.K_flash[j] * f.vapor_properties[0].pressure
                == f.liquid_properties[0].P_sat[j]
            )

        @self.Constraint(doc="Rachford-Rice equation")
        def rr_eq(f):
            return (
                sum(
                    (f.feed_properties[0].mole_frac_comp[j] * (f.K_flash[j] - 1))
                    / (
                        1
                        + (f.K_flash[j] - 1)
                        * f.vapor_properties[0].flow_mol
                        / f.feed_properties[0].flow_mol
                    )
                    for j in f.liquid_properties.component_list
                )
                == 0
            )

        @self.Constraint(doc="Flash material balance")
        def mb_eq(f):
            return (
                f.feed_properties[0].flow_mol
                == f.liquid_properties[0].flow_mol + f.vapor_properties[0].flow_mol
            )

        # Enthalpy balance
        self.h_flash_gas = Var(doc="Enthalpy at outlet")

        self.h_flash_gas_variation = Param(initialize=1, mutable=True)

        @self.Constraint(doc="Enthalpy calculation for outlet")
        def h_gas_eq(f):
            return f.h_flash_gas == f.h_flash_gas_variation  * \
                   (f.vapor_properties[0].mole_frac_comp["H2O"] * (
                f.liquid_properties[0].params.Hf_flash["H2O"]
                + f.liquid_properties[0].Cp_int["H2O"]
                + f.liquid_properties[0].params.deltaH_vap
            ) + f.vapor_properties[0].mole_frac_comp["CO2"] * (
                f.liquid_properties[0].params.Hf_flash["CO2"]
                + f.liquid_properties[0].Cp_int["CO2"]
                - f.liquid_properties[0].deltaH_abs
            ) + f.vapor_properties[
                0
            ].mole_frac_comp[
                "PZ"
            ] * (
                f.liquid_properties[0].params.Hf_flash["PZ"]
                + f.liquid_properties[0].Cp_int["PZ"]
                + f.liquid_properties[0].params.deltaH_PZ
            ))

        @self.Constraint(doc="Enthalpy balance")
        def enthalpy_bal_eq(f):
            return (
                f.feed_properties[0].flow_mol * f.feed_properties[0].h_flash
                == f.h_flash_gas * f.vapor_properties[0].flow_mol
                + f.liquid_properties[0].h_flash * f.liquid_properties[0].flow_mol
            )

        # Expressions
        def x_sum_calc(f):
            return sum(
                f.liquid_properties[0].mole_frac_comp[j]
                for j in f.liquid_properties.component_list
            )

        self.x_sum_flash = Expression(rule=x_sum_calc)

        def y_sum_calc(f):
            return sum(
                f.vapor_properties[0].mole_frac_comp[j]
                for j in f.vapor_properties.component_list
            )

        self.y_sum_flash = Expression(rule=y_sum_calc)
