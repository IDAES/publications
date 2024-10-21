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


from idaes.core.base.unit_model import UnitModelBlockData
from idaes.core import declare_process_block_class, useDefault
import idaes.logger as idaeslog
from pyomo.common.config import ConfigBlock, ConfigValue, Bool, In
from idaes.core.util.config import DefaultBool
from idaes.core.util.config import is_physical_parameter_block
from pyomo.environ import *
from column.PZ_solvent_column import PZPackedColumn
from flash_tank import FlashTank


@declare_process_block_class("PZ_AFS")
class PZ_AFSData(UnitModelBlockData):
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

        self.stripper = PZPackedColumn(
            intercooler=False,
            has_pressure_change=True,
            vapor_phase=self.config.vapor_phase,
            liquid_phase=self.config.liquid_phase,
        )

        self.flash_tank = FlashTank(
            vapor_phase=self.config.vapor_phase, liquid_phase=self.config.liquid_phase
        )

        # Inlets to flash tank and stripper column
        self.F_hot = Var(
            doc="Flowrate of the rich solvent stream that enters the flash tank"
        )
        self.F_warm = Var(
            doc="Flowrate of the rich solvent stream that enters the stripper column from the top"
        )
        self.T_hot = Var(
            doc="Temperature of the rich solvent stream that enters the flash tank, after steam heater"
        )
        self.T_warm = Var(
            doc="Temperature of the rich solvent stream that enters the stripper column from the top"
        )
        self.abs_x = Var(
            self.stripper.liquid_properties.component_list,
            doc="Composition of rich solvent coming from the absorber",
        )

        # Temperature at outlet of flash is same for liquid and vapor phases
        self.T_con = Constraint(
            expr=self.flash_tank.liquid_properties[0].temperature
            == self.flash_tank.vapor_properties[0].temperature
        )

        # connectors between flash tank and stripper column
        # feed to flash tank is a mixture of F_hot and solvent coming down the stripper column
        @self.Constraint(
            self.flash_tank.liquid_properties.component_list,
            doc="Composition calculation for the feed to flash tank",
        )
        def flash_feed_composition_eq(blk, j):
            return blk.flash_tank.feed_properties[0].mole_frac_comp[
                j
            ] * blk.flash_tank.feed_properties[0].flow_mol == (
                blk.stripper.liquid_properties[0, 0].mole_frac_comp[j]
                * blk.stripper.liquid_properties[0, 0].flow_mol
                + blk.F_hot * blk.abs_x[j]
            )

        @self.Constraint(doc="Feed flowrate calculation")
        def flash_feed_eq(blk):
            return (
                blk.flash_tank.feed_properties[0].flow_mol
                == blk.F_hot + blk.stripper.liquid_properties[0, 0].flow_mol
            )

        @self.Constraint(doc="Feed temperature calculation")
        def flash_feed_T_eq(blk):
            return blk.flash_tank.feed_properties[
                0
            ].temperature * blk.flash_tank.feed_properties[0].flow_mol == (
                blk.T_hot * blk.F_hot
                + blk.stripper.liquid_properties[0, 0].flow_mol
                * blk.stripper.liquid_properties[0, 0].temperature
            )

        @self.Constraint(doc="Vapor outlet of flash tank enters the stripper column")
        def stripper_gas_T_eq(blk):
            return (
                blk.stripper.vapor_properties[0, 0].temperature
                == blk.flash_tank.vapor_properties[0].temperature
            )

        @self.Constraint(doc="Vapor outlet of flash tank enters the stripper column")
        def stripper_gas_flow_eq(blk):
            return (
                blk.stripper.vapor_properties[0, 0].flow_mol
                == blk.flash_tank.vapor_properties[0].flow_mol
            )

        @self.Constraint(
            self.flash_tank.vapor_properties.component_list,
            doc="Vapor outlet of flash tank enters the stripper column",
        )
        def stripper_gas_composition_eq(blk, j):
            return (
                blk.stripper.vapor_properties[0, 0].mole_frac_comp[j]
                == blk.flash_tank.vapor_properties[0].mole_frac_comp[j]
            )

        N = self.config.finite_elements

        @self.Constraint(doc="Liquid inlet to the stripper column")
        def stripper_liq_flow_eq(blk):
            return blk.stripper.liquid_properties[0, N].flow_mol == blk.F_warm

        @self.Constraint(
            self.stripper.liquid_properties.component_list,
            doc="Liquid inlet to the stripper column",
        )
        def stripper_liq_composition_eq(blk, j):
            return (
                blk.stripper.liquid_properties[0, N].mole_frac_comp[j] == blk.abs_x[j]
            )

        @self.Constraint(doc="Liquid inlet to the stripper column")
        def stripper_liq_T_eq(blk):
            return blk.stripper.liquid_properties[0, N].temperature == blk.T_warm

    def initialize(blk, solver, outlvl):
        # fix inlets
        blk.F_hot.fix(53.67)
        blk.F_warm.fix(32.6)
        blk.T_hot.fix(156 + 273)
        blk.T_warm.fix(124 + 273)
        blk.abs_x["CO2"].fix(0.0647)
        blk.abs_x["H2O"].fix(0.8495)
        blk.abs_x["PZ"].fix(0.0858)

        # add bounds
        blk.flash_tank.feed_properties[0].flow_mol.setlb(1e-1)
        blk.flash_tank.vapor_properties[0].flow_mol.setlb(1e-1)

        blk.flash_tank.del_component(blk.flash_tank.obj)

        # packing parameters for the stripper (random packing)
        blk.stripper.del_component(blk.stripper.a)
        blk.stripper.a = Param(
            initialize=250, mutable=True, doc="Specific area (m2/m3)"
        )
        blk.stripper.del_component(blk.stripper.void)
        blk.stripper.void = Param(initialize=0.97, mutable=True, doc="Void fraction")
        blk.stripper.del_component(blk.stripper.P_drop_z)
        blk.stripper.P_drop_z = Param(
            initialize=0.3,
            mutable=True,
        )

        # list of variables to be fixed for the first step of initialization
        base_vars = [
            "material_transfer_coefficient_tot",
            "liquid_velocity",
            "effective_vapor_velocity",
            "effective_liquid_velocity",
            "material_transfer_coefficient_gas",
            "heat_transfer_coefficient",
            "flux_mol",
        ]

        # list of equations to be deactivated for the first step of initialization
        base_eqs = [
            "flux_eq",
            "liq_velocity_eq",
            "effective_vapor_velocity_eq",
            "effective_liquid_velocity_eq",
            "material_transfer_coefficient_gas_eq",
            "material_transfer_coefficient_tot_eq",
            "heat_transfer_coefficient_eq",
        ]
        energy_balance_eqs = ["gas_energy_balance_eq", "liq_energy_balance_eq"]

        for v in blk.stripper.component_objects(Var):
            if v.local_name in base_vars:
                v.fix()
        blk.stripper.flux_mol.fix(0)

        for c in blk.stripper.component_objects(Constraint):
            if c.local_name in base_eqs:
                c.deactivate()
            if c.local_name in energy_balance_eqs:
                c.deactivate()
            if c.local_name == "pressure_drop_eq":
                c.deactivate()

        blk.stripper.flooding_velocity.fix()
        blk.stripper.flooding_velocity_eq.deactivate()

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            results = solver.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Flash tank and base column: {}.".format(idaeslog.condition(results))
        )

        for v in blk.stripper.component_objects(Var):
            if v.local_name in base_vars:
                v.unfix()

        for c in blk.stripper.component_objects(Constraint):
            if c.local_name in base_eqs:
                c.activate()

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            results = solver.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Step 2 - Mass transfer: {}.".format(idaeslog.condition(results))
        )

        # deactivate isothermal equations to add energy balance
        blk.stripper.isothermal_gas_eq.deactivate()
        blk.stripper.isothermal_liq_eq.deactivate()

        for c in blk.stripper.component_objects(Constraint):
            if c.local_name in energy_balance_eqs:
                c.activate()
            if c.local_name == "pressure_drop_eq":
                blk.stripper.isobaric_gas_eq.deactivate()
                c.activate()

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            results = solver.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Step 3 - Heat transfer: {}.".format(idaeslog.condition(results))
        )
